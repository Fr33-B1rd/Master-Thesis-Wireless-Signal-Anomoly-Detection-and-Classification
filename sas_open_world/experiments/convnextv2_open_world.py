"""Pseudo-online open-world experiment with ConvNeXtV2 classification and unknown-buffer clustering."""

import argparse
import json
import os
import random

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from timm.data import resolve_model_data_config
from timm.data.transforms_factory import create_transform
import torchvision
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from sas_open_world.clustering import cluster_unknown_buffer


class GaussianNoiseAugment:
    """Add zero-mean Gaussian noise to a normalized (C, H, W) tensor.

    Physical analogue for RF: simulates SNR fluctuations. Does not delete
    discriminative features — preserves all localized F/T patterns while
    teaching the model robustness to amplitude perturbations.
    """

    def __init__(self, std: float = 0.05, prob: float = 1.0) -> None:
        self.std = std
        self.prob = prob

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() >= self.prob:
            return tensor
        return tensor + torch.randn_like(tensor) * self.std
from sas_open_world.common import log, save_confusion
from sas_open_world.features import build_dino_feature_cache, build_feature_cache
from sas_open_world.experiments.convnextv2_open_set import RoiBox, build_roi_boxes
from sas_open_world.openset import select_anchor_indices
from sas_open_world.pipeline import compute_holdout_metrics, compute_phase_metrics


class SpectrogramDataset(Dataset):
    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
        transform,
        roi_boxes: dict[int, RoiBox] | None = None,
    ) -> None:
        self.images = images
        self.labels = labels
        self.indices = indices
        self.transform = transform
        self.roi_boxes = roi_boxes or {}

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int):
        sample_index = int(self.indices[idx])
        image = self.images[sample_index]
        roi_box = self.roi_boxes.get(sample_index)
        if roi_box is not None:
            image = image[roi_box.top : roi_box.bottom, roi_box.left : roi_box.right]
        tensor = self.transform(Image.fromarray(image).convert("RGB"))
        return tensor, int(self.labels[idx]), sample_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pseudo-online ConvNeXtV2 open-world experiment on SAS anomaly classes"
    )
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--known-classes", default="chirp,pulse,tone")
    parser.add_argument("--emerging-classes", default="comb,ofdm,fsk")
    parser.add_argument("--shot", type=int, default=80)
    parser.add_argument("--flow-per-class", type=int, default=600)
    parser.add_argument("--anchor-per-class", type=int, default=30)
    parser.add_argument("--model-name", default="convnextv2_tiny")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--gate-mode", default="energy_global", choices=["energy_global", "distance"])
    parser.add_argument("--threshold-quantile", type=float, default=95.0)
    parser.add_argument("--energy-calibration", default="balanced", choices=["balanced", "pooled"])
    parser.add_argument("--distance-quantile", type=float, default=95.0)
    parser.add_argument("--distance-metric", default="cosine", choices=["cosine", "l2"])
    parser.add_argument("--input-mode", default="full", choices=["full", "roi"])
    parser.add_argument("--roi-quantile", type=float, default=99.0)
    parser.add_argument("--roi-pad-ratio", type=float, default=0.10)
    parser.add_argument("--roi-min-side", type=int, default=32)
    parser.add_argument(
        "--clustering-feature-source",
        default="convnext",
        choices=["convnext", "patchcore_mean_plus_gram", "dino"],
    )
    parser.add_argument("--patchcore-output-dir")
    parser.add_argument("--dino-model-name", default="vit_small_patch16_dinov3")
    parser.add_argument("--dino-batch-size", type=int, default=32)
    parser.add_argument("--top-k-patches", type=int, default=64)
    parser.add_argument("--gram-dim", type=int, default=32)
    parser.add_argument("--cluster-pca-dim", type=int, default=8)
    parser.add_argument("--cluster-min-cluster-size", type=int, default=50)
    parser.add_argument("--cluster-min-samples", type=int, default=1)
    parser.add_argument("--cluster-accept-purity", type=float, default=0.9)
    parser.add_argument("--cluster-accept-min-size", type=int, default=40)
    parser.add_argument("--cluster-max-anchor-fraction", type=float, default=0.10)
    parser.add_argument("--cluster-dim-reduction", default="pca", choices=["pca", "umap"])
    parser.add_argument("--cluster-umap-n-neighbors", type=int, default=15)
    parser.add_argument("--cluster-umap-min-dist", type=float, default=0.0)
    parser.add_argument("--cluster-selection-method", default="eom", choices=["eom", "leaf"])
    parser.add_argument("--cluster-accept-min-prob", type=float, default=-1.0,
                        help="Unsupervised acceptance: min mean HDBSCAN membership probability. "
                             "Set >=0 to enable unsupervised mode (disables oracle purity check). "
                             "Default -1 = legacy oracle mode.")
    parser.add_argument("--cluster-anchor-null-alpha", type=float, default=-1.0,
                        help="Scale-free known-class-leakage test. If >=0, replaces "
                             "--cluster-max-anchor-fraction with a per-class hypergeometric "
                             "test: reject cluster if any known class anchor count is "
                             "significantly higher than random at level alpha/M (Bonferroni "
                             "over M known classes). Default -1 = use absolute fraction threshold.")
    parser.add_argument("--anchor-core-prob-threshold", type=float, default=0.5,
                        help="Under --cluster-anchor-null-alpha, reject leakage only when "
                             "mean HDBSCAN membership probability of the cluster's anchors "
                             "is >= this threshold (i.e. anchors are core members, not "
                             "periphery points dragged in by UMAP).")
    parser.add_argument("--cluster-anchor-quantile-alpha", type=float, default=-1.0,
                        help="Statistically rigorous anchor_fraction threshold. Per-cluster "
                             "threshold = hypergeom.ppf(1-alpha, N, K_a, n) / n, where K_a "
                             "is total anchor count and n is cluster size. Equivalent to a "
                             "one-sided permutation test on total anchor_fraction at level "
                             "alpha. Default -1 = disabled.")
    parser.add_argument("--cluster-anchor-fraction-scale", type=float, default=-1.0,
                        help="Scale-free reformulation of --cluster-max-anchor-fraction. "
                             "If >=0, reject cluster when anchor_fraction exceeds "
                             "scale x (n_anchor_total / n_total). Physical meaning: "
                             "'cluster has more than <scale>x the null-expected anchor "
                             "density'. Auto-scales with anchor count and buffer size. "
                             "Default -1 = use absolute --cluster-max-anchor-fraction.")
    parser.add_argument("--cluster-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_protocol(
    label_names: np.ndarray,
    known_classes: list[str],
    emerging_classes: list[str],
    shot: int,
    flow_per_class: int,
    rng: np.random.Generator,
):
    train_indices_by_label: dict[str, list[int]] = {}
    flow_indices_by_label: dict[str, list[int]] = {}
    holdout_indices_by_label: dict[str, list[int]] = {}
    split_summary: dict[str, dict[str, int]] = {}

    for class_name in [*known_classes, *emerging_classes]:
        class_indices = np.where(label_names == class_name)[0]
        shuffled = rng.permutation(class_indices).astype(np.int64)
        if class_name in known_classes:
            required = shot + flow_per_class
            if shuffled.shape[0] < required:
                raise ValueError(
                    f"{class_name} has only {shuffled.shape[0]} samples, fewer than shot+flow={required}"
                )
            train_part = shuffled[:shot]
            flow_part = shuffled[shot : shot + flow_per_class]
            holdout_part = shuffled[shot + flow_per_class :]
            train_indices_by_label[class_name] = train_part.tolist()
            split_summary[class_name] = {
                "train_count": int(train_part.shape[0]),
                "flow_count": int(flow_part.shape[0]),
                "holdout_count": int(holdout_part.shape[0]),
            }
        else:
            if shuffled.shape[0] < flow_per_class:
                raise ValueError(
                    f"{class_name} has only {shuffled.shape[0]} samples, fewer than flow={flow_per_class}"
                )
            flow_part = shuffled[:flow_per_class]
            holdout_part = shuffled[flow_per_class:]
            split_summary[class_name] = {
                "train_count": 0,
                "flow_count": int(flow_part.shape[0]),
                "holdout_count": int(holdout_part.shape[0]),
            }
        flow_indices_by_label[class_name] = flow_part.tolist()
        holdout_indices_by_label[class_name] = holdout_part.tolist()

    stream_pool = []
    for class_name in [*known_classes, *emerging_classes]:
        stream_pool.extend(flow_indices_by_label[class_name])
    stream_indices = rng.permutation(np.asarray(stream_pool, dtype=np.int64)).astype(np.int64).tolist()
    return train_indices_by_label, flow_indices_by_label, holdout_indices_by_label, split_summary, stream_indices


def create_model(model_name: str, num_classes: int, device: torch.device):
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    model = model.to(device)
    data_config = resolve_model_data_config(model)
    transform = create_transform(**data_config, is_training=False)
    return model, transform


def train_model(
    model,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> list[dict]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = []
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        for images, labels, _ in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * int(labels.size(0))
            preds = logits.argmax(dim=1)
            running_correct += int((preds == labels).sum().item())
            running_total += int(labels.size(0))
        history.append(
            {
                "epoch": epoch + 1,
                "loss": running_loss / max(running_total, 1),
                "train_accuracy": running_correct / max(running_total, 1),
            }
        )
        log(
            f"epoch {epoch + 1}/{epochs}: "
            f"loss={history[-1]['loss']:.4f}, "
            f"train_acc={history[-1]['train_accuracy']:.4f}"
        )
    return history


@torch.no_grad()
def infer_with_embeddings(
    model,
    loader: DataLoader,
    device: torch.device,
    temperature: float,
) -> pd.DataFrame:
    model.eval()
    rows = []
    for images, labels, sample_indices in loader:
        images = images.to(device, non_blocking=True)
        features = model.forward_features(images)
        embeddings = model.forward_head(features, pre_logits=True)
        logits = model.forward_head(features)
        preds = logits.argmax(dim=1)
        energy = (-temperature * torch.logsumexp(logits / temperature, dim=1)).detach()
        for i in range(images.size(0)):
            row = {
                "sample_index": int(sample_indices[i].item()),
                "label_id": int(labels[i].item()),
                "pred_id": int(preds[i].item()),
                "energy": float(energy[i].item()),
                "embedding": embeddings[i].detach().cpu().numpy().astype(np.float32),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def build_loader(
    images,
    indices: list[int],
    label_names: np.ndarray,
    class_to_id: dict[str, int],
    transform,
    batch_size: int,
    roi_boxes: dict[int, RoiBox] | None = None,
):
    indices_array = np.asarray(indices, dtype=np.int64)
    labels_array = np.asarray([class_to_id[str(label_names[idx])] for idx in indices_array], dtype=np.int64)
    dataset = SpectrogramDataset(
        images=images,
        labels=labels_array,
        indices=indices_array,
        transform=transform,
        roi_boxes=roi_boxes,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)


def _build_balanced_energy_values(
    calib_df: pd.DataFrame,
    effective_known_classes: list[str],
    class_to_id: dict[str, int],
) -> tuple[np.ndarray, dict[str, int]]:
    by_class = []
    available_counts = {}
    for class_name in effective_known_classes:
        class_id = class_to_id[class_name]
        class_energy = np.asarray(
            calib_df.loc[calib_df["label_id"] == class_id, "energy"].tolist(),
            dtype=np.float32,
        )
        available_counts[class_name] = int(class_energy.shape[0])
        by_class.append((class_name, class_energy))
    target_count = min(int(values.shape[0]) for _, values in by_class)
    balanced_values = np.concatenate([values[:target_count] for _, values in by_class], axis=0).astype(np.float32)
    balanced_counts = {class_name: target_count for class_name, _ in by_class}
    return balanced_values, balanced_counts


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms


def _compute_class_distances(
    embeddings: np.ndarray,
    prototype: np.ndarray,
    metric: str,
) -> np.ndarray:
    if metric == "cosine":
        norm_embeddings = _l2_normalize(embeddings)
        norm_prototype = prototype / max(float(np.linalg.norm(prototype)), 1e-12)
        return 1.0 - np.matmul(norm_embeddings, norm_prototype.astype(np.float32))
    if metric == "l2":
        return np.linalg.norm(embeddings - prototype[None, :], axis=1)
    raise ValueError(f"Unsupported distance metric: {metric}")


def train_convnext_open_world_model(
    images: np.ndarray,
    label_names: np.ndarray,
    train_indices_by_label: dict[str, list[int]],
    current_known_classes: list[str],
    model_name: str,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    temperature: float,
    gate_mode: str,
    threshold_quantile: float,
    energy_calibration: str,
    distance_quantile: float,
    distance_metric: str,
    device: torch.device,
    roi_boxes: dict[int, RoiBox] | None = None,
):
    train_indices = []
    for class_name in current_known_classes:
        train_indices.extend(train_indices_by_label[class_name])
    observed_train_labels = []
    for idx in train_indices:
        label_name = str(label_names[idx])
        if label_name not in observed_train_labels:
            observed_train_labels.append(label_name)
    effective_known_classes = list(current_known_classes)
    for label_name in observed_train_labels:
        if label_name not in effective_known_classes:
            effective_known_classes.append(label_name)

    model, transform = create_model(model_name=model_name, num_classes=len(effective_known_classes), device=device)
    class_to_id = {name: idx for idx, name in enumerate(effective_known_classes)}
    train_labels = np.asarray([class_to_id[str(label_names[idx])] for idx in train_indices], dtype=np.int64)
    class_counts = np.bincount(train_labels, minlength=len(effective_known_classes)).astype(np.float64)
    class_weights = np.divide(
        1.0,
        np.maximum(class_counts, 1.0),
    )
    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(train_indices),
        replacement=True,
    )
    noise_augment = GaussianNoiseAugment(std=0.05)
    train_transform = torchvision.transforms.Compose([transform, noise_augment])
    train_loader = DataLoader(
        SpectrogramDataset(
            images=images,
            labels=train_labels,
            indices=np.asarray(train_indices, dtype=np.int64),
            transform=train_transform,
            roi_boxes=roi_boxes,
        ),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
    )
    history = train_model(
        model=model,
        train_loader=train_loader,
        device=device,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
    )

    calib_loader = build_loader(
        images=images,
        indices=train_indices,
        label_names=label_names,
        class_to_id=class_to_id,
        transform=transform,
        batch_size=batch_size,
        roi_boxes=roi_boxes,
    )
    calib_df = infer_with_embeddings(
        model=model,
        loader=calib_loader,
        device=device,
        temperature=temperature,
    )
    if gate_mode == "energy_global":
        pooled_energy_values = np.asarray(calib_df["energy"].tolist(), dtype=np.float32)
        if energy_calibration == "balanced":
            energy_values, balanced_counts = _build_balanced_energy_values(
                calib_df=calib_df,
                effective_known_classes=effective_known_classes,
                class_to_id=class_to_id,
            )
        else:
            energy_values = pooled_energy_values
            balanced_counts = {}
        gate_info = {
            "mode": "energy_global",
            "metric": "energy",
            "global": float(np.percentile(energy_values, threshold_quantile)),
            "by_class": {},
            "prototypes": {},
            "quantile": float(threshold_quantile),
            "calibration": energy_calibration,
            "calibration_counts": balanced_counts,
        }
    else:
        embeddings = np.stack(calib_df["embedding"].tolist(), axis=0).astype(np.float32)
        gate_thresholds_by_class = {}
        prototypes_by_class = {}
        for class_name in effective_known_classes:
            class_id = class_to_id[class_name]
            class_embeddings = embeddings[np.asarray(calib_df["label_id"] == class_id)]
            if distance_metric == "cosine":
                normalized = _l2_normalize(class_embeddings)
                prototype = normalized.mean(axis=0).astype(np.float32)
                prototype = prototype / max(float(np.linalg.norm(prototype)), 1e-12)
            else:
                prototype = class_embeddings.mean(axis=0).astype(np.float32)
            distances = _compute_class_distances(
                embeddings=class_embeddings,
                prototype=prototype,
                metric=distance_metric,
            )
            prototypes_by_class[class_name] = prototype
            gate_thresholds_by_class[class_name] = float(np.percentile(distances, distance_quantile))
        all_class_thresholds = np.asarray(list(gate_thresholds_by_class.values()), dtype=np.float32)
        gate_info = {
            "mode": "per_class_distance",
            "metric": distance_metric,
            "global": float(np.mean(all_class_thresholds)),
            "by_class": gate_thresholds_by_class,
            "prototypes": prototypes_by_class,
            "quantile": float(distance_quantile),
            "calibration": "per_class_distance",
            "calibration_counts": {},
        }
    return (
        model,
        transform,
        class_to_id,
        gate_info,
        history,
        {
            class_name: int(class_counts[class_to_id[class_name]]) for class_name in effective_known_classes
        },
    )


def predict_sample(
    model,
    transform,
    image: np.ndarray,
    model_class_names: list[str],
    gate_info: dict,
    temperature: float,
    device: torch.device,
    roi_box: RoiBox | None = None,
):
    if roi_box is not None:
        image = image[roi_box.top : roi_box.bottom, roi_box.left : roi_box.right]
    tensor = transform(Image.fromarray(image).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.forward_features(tensor)
        embedding = model.forward_head(features, pre_logits=True)
        logits = model.forward_head(features)
        energy = float((-temperature * torch.logsumexp(logits / temperature, dim=1)).item())
        pred_id = int(logits.argmax(dim=1).item())
    embedding_np = embedding.squeeze(0).detach().cpu().numpy().astype(np.float32)
    pred_known_label = model_class_names[pred_id]
    if gate_info["mode"] == "energy_global":
        reject = energy > float(gate_info["global"])
    else:
        class_prototype = np.asarray(gate_info["prototypes"][pred_known_label], dtype=np.float32)
        class_threshold = float(gate_info["by_class"].get(pred_known_label, gate_info["global"]))
        sample_distance = float(
            _compute_class_distances(
                embeddings=embedding_np[None, :],
                prototype=class_prototype,
                metric=str(gate_info["metric"]),
            )[0]
        )
        reject = sample_distance > class_threshold
    pred_label = "unknown" if reject else pred_known_label
    return pred_label, energy, embedding_np


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    known_classes = [part.strip() for part in args.known_classes.split(",") if part.strip()]
    emerging_classes = [part.strip() for part in args.emerging_classes.split(",") if part.strip()]
    all_eval_classes = [*known_classes, *emerging_classes]

    images = np.load(os.path.join(args.dataset_dir, "test_spectrogram.npy"), mmap_mode="r")
    label_names = np.load(os.path.join(args.dataset_dir, "test_labels.npy")).astype(str)
    roi_boxes = None
    if args.input_mode == "roi":
        if not args.patchcore_output_dir:
            raise ValueError("--patchcore-output-dir is required when input-mode=roi")
        anomaly_outputs = np.load(os.path.join(args.patchcore_output_dir, "anomaly_outputs.npz"))
        anomaly_maps = anomaly_outputs["anomaly_maps"]
        roi_boxes = build_roi_boxes(
            anomaly_maps=anomaly_maps,
            sample_indices=np.arange(label_names.shape[0], dtype=np.int64),
            quantile=args.roi_quantile,
            pad_ratio=args.roi_pad_ratio,
            min_side=args.roi_min_side,
        )
        roi_heights = np.asarray([box.bottom - box.top for box in roi_boxes.values()], dtype=np.int64)
        roi_widths = np.asarray([box.right - box.left for box in roi_boxes.values()], dtype=np.int64)
        with open(os.path.join(args.output_dir, "roi_summary.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "num_boxes": int(len(roi_boxes)),
                    "quantile": args.roi_quantile,
                    "pad_ratio": args.roi_pad_ratio,
                    "min_side": args.roi_min_side,
                    "mean_height": float(np.mean(roi_heights)),
                    "mean_width": float(np.mean(roi_widths)),
                    "min_height": int(np.min(roi_heights)),
                    "max_height": int(np.max(roi_heights)),
                    "min_width": int(np.min(roi_widths)),
                    "max_width": int(np.max(roi_widths)),
                },
                f,
                indent=2,
            )
    x_cluster_all = None
    if args.clustering_feature_source == "patchcore_mean_plus_gram":
        if not args.patchcore_output_dir:
            raise ValueError("--patchcore-output-dir is required when clustering-feature-source=patchcore_mean_plus_gram")
        cache_path = os.path.join(
            args.output_dir,
            f"feature_cache_patchcore_mean_plus_gram_topk{args.top_k_patches}_gram{args.gram_dim}.npz",
        )
        x_cluster_all, cache_labels = build_feature_cache(
            dataset_dir=args.dataset_dir,
            patchcore_output_dir=args.patchcore_output_dir,
            feature_mode="mean_plus_gram",
            top_k_patches=args.top_k_patches,
            gram_dim=args.gram_dim,
            seed=args.seed,
            cache_path=cache_path,
            augment_temporal_diff=False,
            augment_dynamic_range=False,
            augment_entropy=False,
            augment_frequency_profile=False,
        )
        if not np.array_equal(label_names, cache_labels.astype(str)):
            raise ValueError("PatchCore feature cache labels do not align with dataset labels")
    elif args.clustering_feature_source == "dino":
        cache_path = os.path.join(
            args.output_dir,
            f"feature_cache_dino_{args.dino_model_name.replace('/', '_')}.npz",
        )
        x_cluster_all, cache_labels = build_dino_feature_cache(
            dataset_dir=args.dataset_dir,
            dino_model_name=args.dino_model_name,
            cache_path=cache_path,
            batch_size=args.dino_batch_size,
            device=device,
        )
        if not np.array_equal(label_names, cache_labels.astype(str)):
            raise ValueError("DINO feature cache labels do not align with dataset labels")

    (
        train_indices_by_label,
        flow_indices_by_label,
        holdout_indices_by_label,
        split_summary,
        stream_indices,
    ) = build_protocol(
        label_names=label_names,
        known_classes=known_classes,
        emerging_classes=emerging_classes,
        shot=args.shot,
        flow_per_class=args.flow_per_class,
        rng=rng,
    )
    with open(os.path.join(args.output_dir, "split_protocol.json"), "w", encoding="utf-8") as f:
        json.dump(split_summary, f, indent=2)

    current_known_classes = list(known_classes)
    (
        model,
        transform,
        class_to_id,
        gate_info,
        history,
        train_class_counts,
    ) = train_convnext_open_world_model(
        images=images,
        label_names=label_names,
        train_indices_by_label=train_indices_by_label,
        current_known_classes=current_known_classes,
        model_name=args.model_name,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        gate_mode=args.gate_mode,
        threshold_quantile=args.threshold_quantile,
        energy_calibration=args.energy_calibration,
        distance_quantile=args.distance_quantile,
        distance_metric=args.distance_metric,
        device=device,
        roi_boxes=roi_boxes,
    )
    model_class_names = [name for name, _ in sorted(class_to_id.items(), key=lambda item: item[1])]
    with open(os.path.join(args.output_dir, "initial_train_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(args.output_dir, "initial_train_class_counts.json"), "w", encoding="utf-8") as f:
        json.dump(train_class_counts, f, indent=2)

    processed_rows = []
    unknown_sample_indices: list[int] = []
    unknown_embeddings: list[np.ndarray] = []
    update_events = []

    # Pre-update stream pass: classify each sample; collect those flagged as unknown.
    for stream_pos, sample_index in enumerate(stream_indices):
        true_label = str(label_names[sample_index])
        pred_label, energy, embedding = predict_sample(
            model=model,
            transform=transform,
            image=np.asarray(images[sample_index]),
            model_class_names=model_class_names,
            gate_info=gate_info,
            temperature=args.temperature,
            device=device,
            roi_box=None if roi_boxes is None else roi_boxes.get(int(sample_index)),
        )
        processed_rows.append(
            {
                "stream_position": int(stream_pos),
                "sample_index": int(sample_index),
                "phase": "pre_update",
                "true_label": true_label,
                "pred_label": pred_label,
                "energy": energy,
            }
        )
        if pred_label == "unknown":
            unknown_sample_indices.append(int(sample_index))
            unknown_embeddings.append(embedding)

    # End-of-pre-update single clustering event with known-class anchors.
    if unknown_sample_indices:
        update_id = 1
        if x_cluster_all is None:
            x_buffer = np.stack(unknown_embeddings, axis=0).astype(np.float32)
        else:
            x_buffer = x_cluster_all[np.asarray(unknown_sample_indices, dtype=np.int64)].astype(np.float32, copy=False)
        anchor_indices = select_anchor_indices(
            train_indices_by_label=train_indices_by_label,
            current_known_classes=current_known_classes,
            excluded_indices=unknown_sample_indices,
            anchor_per_class=args.anchor_per_class,
            seed=args.seed + update_id,
        )
        if x_cluster_all is None:
            x_anchor_list = []
            for anchor_idx in anchor_indices:
                _, _, anchor_embedding = predict_sample(
                    model=model,
                    transform=transform,
                    image=np.asarray(images[anchor_idx]),
                    model_class_names=model_class_names,
                    gate_info=gate_info,
                    temperature=args.temperature,
                    device=device,
                    roi_box=None if roi_boxes is None else roi_boxes.get(int(anchor_idx)),
                )
                x_anchor_list.append(anchor_embedding)
            x_anchor = (
                np.stack(x_anchor_list, axis=0).astype(np.float32)
                if x_anchor_list
                else np.empty((0, x_buffer.shape[1]), dtype=np.float32)
            )
        else:
            x_anchor = x_cluster_all[np.asarray(anchor_indices, dtype=np.int64)].astype(np.float32, copy=False)

        assignment_df, accepted_by_label, cluster_summary = cluster_unknown_buffer(
            x_buffer=x_buffer,
            true_labels=label_names[np.asarray(unknown_sample_indices, dtype=np.int64)],
            buffer_sample_indices=np.asarray(unknown_sample_indices, dtype=np.int64),
            x_anchor=x_anchor,
            anchor_labels=label_names[np.asarray(anchor_indices, dtype=np.int64)],
            anchor_sample_indices=np.asarray(anchor_indices, dtype=np.int64),
            current_known_classes=current_known_classes,
            emerging_classes=emerging_classes,
            pca_dim=args.cluster_pca_dim,
            min_cluster_size=args.cluster_min_cluster_size,
            min_samples=args.cluster_min_samples,
            cluster_accept_purity=args.cluster_accept_purity,
            cluster_accept_min_size=args.cluster_accept_min_size,
            cluster_max_anchor_fraction=args.cluster_max_anchor_fraction,
            seed=args.seed,
            dim_reduction=args.cluster_dim_reduction,
            umap_n_neighbors=args.cluster_umap_n_neighbors,
            umap_min_dist=args.cluster_umap_min_dist,
            cluster_selection_method=args.cluster_selection_method,
            cluster_accept_min_prob=args.cluster_accept_min_prob,
            cluster_anchor_null_alpha=args.cluster_anchor_null_alpha,
            anchor_core_prob_threshold=args.anchor_core_prob_threshold,
            cluster_anchor_fraction_scale=args.cluster_anchor_fraction_scale,
            cluster_anchor_quantile_alpha=args.cluster_anchor_quantile_alpha,
        )

        event_dir = os.path.join(args.output_dir, f"update_{update_id:02d}")
        os.makedirs(event_dir, exist_ok=True)
        assignment_df.to_csv(os.path.join(event_dir, "cluster_assignments.csv"), index=False)

        discovered_classes = []
        accepted_counts = {}
        for class_name, local_indices in accepted_by_label.items():
            global_indices = [unknown_sample_indices[local_idx] for local_idx in sorted(set(local_indices))]
            train_indices_by_label.setdefault(class_name, [])
            train_indices_by_label[class_name].extend(global_indices)
            train_indices_by_label[class_name] = sorted(set(train_indices_by_label[class_name]))
            accepted_counts[class_name] = len(global_indices)
            if class_name not in current_known_classes:
                current_known_classes.append(class_name)
                discovered_classes.append(class_name)

        cluster_summary["accepted_counts"] = accepted_counts
        cluster_summary["discovered_classes"] = discovered_classes
        cluster_summary["buffer_sample_indices"] = list(unknown_sample_indices)
        with open(os.path.join(event_dir, "cluster_summary.json"), "w", encoding="utf-8") as f:
            json.dump(cluster_summary, f, indent=2)

        if args.cluster_only:
            log(f"cluster-only mode: stopping after clustering (discovered={discovered_classes}, accepted={accepted_counts})")
            return

        update_events.append(
            {
                "update_id": update_id,
                "trigger_stream_position": int(len(stream_indices) - 1),
                "unknown_count": len(unknown_sample_indices),
                "discovered_classes": discovered_classes,
                "accepted_counts": accepted_counts,
            }
        )

        if discovered_classes:
            (
                model,
                transform,
                class_to_id,
                gate_info,
                history,
                train_class_counts,
            ) = train_convnext_open_world_model(
                images=images,
                label_names=label_names,
                train_indices_by_label=train_indices_by_label,
                current_known_classes=current_known_classes,
                model_name=args.model_name,
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                temperature=args.temperature,
                gate_mode=args.gate_mode,
                threshold_quantile=args.threshold_quantile,
                energy_calibration=args.energy_calibration,
                distance_quantile=args.distance_quantile,
                distance_metric=args.distance_metric,
                device=device,
                roi_boxes=roi_boxes,
            )
            model_class_names = [name for name, _ in sorted(class_to_id.items(), key=lambda item: item[1])]
            with open(os.path.join(event_dir, "retrain_history.json"), "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
            with open(os.path.join(event_dir, "train_class_counts.json"), "w", encoding="utf-8") as f:
                json.dump(train_class_counts, f, indent=2)
            log(
                f"Update {update_id}: discovered {discovered_classes}, "
                f"retrained with classes {current_known_classes}, "
                f"gate_mode={gate_info['mode']}, "
                f"gate_metric={gate_info['metric']}, "
                f"global_gate_value={gate_info['global']:.4f}"
            )
        else:
            log(f"Update {update_id}: no new class accepted from unknown set")

    records_df = pd.DataFrame(processed_rows)
    records_df.to_csv(os.path.join(args.output_dir, "stream_predictions.csv"), index=False)
    with open(os.path.join(args.output_dir, "update_events.json"), "w", encoding="utf-8") as f:
        json.dump(update_events, f, indent=2)

    holdout_indices = []
    for class_name in all_eval_classes:
        holdout_indices.extend(holdout_indices_by_label[class_name])
    holdout_rows = []
    for sample_index in holdout_indices:
        pred_label, energy, _ = predict_sample(
            model=model,
            transform=transform,
            image=np.asarray(images[sample_index]),
            model_class_names=model_class_names,
            gate_info=gate_info,
            temperature=args.temperature,
            device=device,
            roi_box=None if roi_boxes is None else roi_boxes.get(int(sample_index)),
        )
        row = {
            "sample_index": int(sample_index),
            "true_label": str(label_names[sample_index]),
            "pred_label": pred_label,
            "energy": energy,
        }
        holdout_rows.append(row)
    holdout_df = pd.DataFrame(holdout_rows)
    holdout_df.to_csv(os.path.join(args.output_dir, "holdout_predictions.csv"), index=False)

    metrics = {
        "processed_samples": int(records_df.shape[0]),
        "initial_known_classes": known_classes,
        "final_known_classes": current_known_classes,
        "num_updates": len(update_events),
        "final_gate_mode": str(gate_info["mode"]),
        "final_gate_metric": str(gate_info["metric"]),
        "final_global_gate_value": float(gate_info["global"]),
        "final_gate_calibration": str(gate_info.get("calibration", "")),
        "final_gate_calibration_counts": {
            str(class_name): int(value)
            for class_name, value in gate_info.get("calibration_counts", {}).items()
        },
        "final_distance_thresholds_by_class": {
            str(class_name): float(value)
            for class_name, value in gate_info.get("by_class", {}).items()
        },
        "overall": compute_phase_metrics(
            records_df=records_df,
            known_classes=known_classes,
            emerging_classes=emerging_classes,
        ),
        "holdout": compute_holdout_metrics(
            records_df=holdout_df,
            final_known_classes=current_known_classes,
            all_eval_classes=all_eval_classes,
        ),
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    save_confusion(
        y_true=records_df["true_label"].tolist(),
        y_pred=records_df["pred_label"].tolist(),
        label_order=["unknown", *all_eval_classes],
        output_path=os.path.join(args.output_dir, "stream_confusion_matrix.png"),
        title="ConvNeXt Open-World Stream Confusion Matrix",
    )
    save_confusion(
        y_true=holdout_df["true_label"].tolist(),
        y_pred=holdout_df["pred_label"].tolist(),
        label_order=["unknown", *all_eval_classes],
        output_path=os.path.join(args.output_dir, "holdout_confusion_matrix.png"),
        title="ConvNeXt Open-World Holdout Confusion Matrix",
    )
    log(f"Finished ConvNeXtV2 open-world experiment with {len(update_events)} update(s)")


if __name__ == "__main__":
    main()
