"""ConvNeXtV2 classifier construction, training, and inference.

Top-level entry points:

- ``build_protocol``: split indices per class into train / stream / holdout.
  Called by the experiment driver before any model exists.
- ``train_convnext_open_world_model``: one-shot orchestrator that builds the
  model, fits it on the per-class train shot, runs calibration to compute the
  open-set gate, and returns (model, transform, class_to_id, gate_info,
  history, class_counts). Reused for the post-discovery retrain loop.
- ``predict_sample``: single-sample inference including the open-set gate.

Internal helpers (``_build_balanced_energy_values``, ``_l2_normalize``,
``_compute_class_distances``) stay leading-underscored to signal intra-module
use only.
"""

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from timm.data import resolve_model_data_config
from timm.data.transforms_factory import create_transform
from torch.utils.data import DataLoader, WeightedRandomSampler

from sas_open_world.common import log
from sas_open_world.experiments.convnextv2_open_set import RoiBox
from sas_open_world.models.dataset import SpectrogramDataset, build_loader
from sas_open_world.models.transforms import (
    GaussianNoiseAugment,
    MagOneChannelTransform,
    ThreeChannelTransform,
    _apply_stem_surgery_for_1channel,
    _apply_stem_surgery_for_3channel,
    transform_is_mag_1ch_native,
)


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
    stream_indices = (
        rng.permutation(np.asarray(stream_pool, dtype=np.int64)).astype(np.int64).tolist()
    )
    return (
        train_indices_by_label,
        flow_indices_by_label,
        holdout_indices_by_label,
        split_summary,
        stream_indices,
    )


def create_model(
    model_name: str,
    num_classes: int,
    device: torch.device,
    input_channels: str = "mag",
    channel_stats: dict | None = None,
    stem_perturb_std: float = 0.01,
    stem_surgery_seed: int = 0,
):
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    model = model.to(device)
    data_config = resolve_model_data_config(model)
    if input_channels in {"mag_if", "mag_gd", "mag_if_gd"}:
        if channel_stats is None:
            raise ValueError(
                f"channel_stats must be provided when input_channels='{input_channels}'"
            )
        out_size = int(data_config.get("input_size", (3, 224, 224))[-1])
        # zero_channels: which of (Ch0 log_mag, Ch1 IF, Ch2 GD) to force to 0
        # after normalization. mag_if keeps {0,1}, mag_gd keeps {0,2}, mag_if_gd
        # keeps all three.
        zero_channels = {
            "mag_if_gd": (),
            "mag_if": (2,),   # zero GD
            "mag_gd": (1,),   # zero IF
        }[input_channels]
        transform = ThreeChannelTransform(
            mean=channel_stats["mean"],
            std=channel_stats["std"],
            out_size=out_size,
            zero_channels=zero_channels,
        )
        _apply_stem_surgery_for_3channel(
            model=model,
            perturb_std_ratio=float(stem_perturb_std),
            seed=int(stem_surgery_seed),
        )
    elif input_channels == "mag_1ch":
        if channel_stats is None:
            raise ValueError(
                "channel_stats must be provided when input_channels='mag_1ch' "
                "(uses Ch0 mean/std from channel_stats.json)"
            )
        out_size = int(data_config.get("input_size", (3, 224, 224))[-1])
        transform = MagOneChannelTransform(
            mean=float(channel_stats["mean"][0]),
            std=float(channel_stats["std"][0]),
            out_size=out_size,
        )
        _apply_stem_surgery_for_1channel(
            model=model,
            perturb_std_ratio=float(stem_perturb_std),
            seed=int(stem_surgery_seed),
        )
    else:
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
    balanced_values = np.concatenate(
        [values[:target_count] for _, values in by_class], axis=0
    ).astype(np.float32)
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
    train_noise_std: float = 0.05,
    input_channels: str = "mag",
    channel_stats: dict | None = None,
    stem_perturb_std: float = 0.01,
    stem_surgery_seed: int = 0,
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

    model, transform = create_model(
        model_name=model_name,
        num_classes=len(effective_known_classes),
        device=device,
        input_channels=input_channels,
        channel_stats=channel_stats,
        stem_perturb_std=stem_perturb_std,
        stem_surgery_seed=stem_surgery_seed,
    )
    class_to_id = {name: idx for idx, name in enumerate(effective_known_classes)}
    train_labels = np.asarray(
        [class_to_id[str(label_names[idx])] for idx in train_indices], dtype=np.int64
    )
    class_counts = np.bincount(
        train_labels, minlength=len(effective_known_classes)
    ).astype(np.float64)
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
    if train_noise_std > 0.0:
        noise_augment = GaussianNoiseAugment(std=train_noise_std)
        train_transform = torchvision.transforms.Compose([transform, noise_augment])
    else:
        train_transform = transform
    train_dataset = SpectrogramDataset(
        images=images,
        labels=train_labels,
        indices=np.asarray(train_indices, dtype=np.int64),
        transform=train_transform,
        roi_boxes=roi_boxes,
    )
    train_dataset._mag_1ch_native = transform_is_mag_1ch_native(train_transform)
    train_loader = DataLoader(
        train_dataset,
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
            gate_thresholds_by_class[class_name] = float(
                np.percentile(distances, distance_quantile)
            )
        all_class_thresholds = np.asarray(
            list(gate_thresholds_by_class.values()), dtype=np.float32
        )
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
            class_name: int(class_counts[class_to_id[class_name]])
            for class_name in effective_known_classes
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
    if image.ndim == 3:
        if roi_box is not None:
            image = image[:, roi_box.top : roi_box.bottom, roi_box.left : roi_box.right]
        tensor = transform(image).unsqueeze(0).to(device)
    elif transform_is_mag_1ch_native(transform):
        if roi_box is not None:
            image = image[roi_box.top : roi_box.bottom, roi_box.left : roi_box.right]
        tensor = transform(image).unsqueeze(0).to(device)
    else:
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
        class_prototype = np.asarray(
            gate_info["prototypes"][pred_known_label], dtype=np.float32
        )
        class_threshold = float(
            gate_info["by_class"].get(pred_known_label, gate_info["global"])
        )
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
