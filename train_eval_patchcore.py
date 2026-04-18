# Trains and evaluates PatchCore on the original IAD PKL datasets, including
# anomaly scoring, localization maps, and optional patch-artifact export.

import argparse
import json
import math
import os
import pickle
import random
import sys
import time
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.random_projection import SparseRandomProjection
from torch.utils.data import DataLoader, Dataset
from torchvision.models import (
    ResNet18_Weights,
    Wide_ResNet50_2_Weights,
    resnet18,
    wide_resnet50_2,
)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def install_numpy_pickle_compat() -> None:
    # Some dataset pickles were serialized under NumPy 2.x where private module
    # paths differ from NumPy 1.x. Alias the old paths before unpickling so the
    # IIS environment can load the dataset without upgrading NumPy.
    compat_modules = {
        "numpy._core": np.core,
        "numpy._core.numeric": np.core.numeric,
        "numpy._core.multiarray": np.core.multiarray,
        "numpy._core.umath": np.core.umath,
        "numpy._core._multiarray_umath": np.core._multiarray_umath,
    }
    for module_name, module_obj in compat_modules.items():
        sys.modules.setdefault(module_name, module_obj)


@dataclass
class DatasetBundle:
    train_data: np.ndarray
    train_label: np.ndarray
    test_data: np.ndarray
    test_label: np.ndarray
    mean: float
    std: float


def load_iad_dataset(path: str) -> DatasetBundle:
    install_numpy_pickle_compat()
    with open(path, "rb") as f:
        obj = pickle.load(f)

    required_keys = {"train_data", "train_label", "test_data", "test_label"}
    if not isinstance(obj, dict) or not required_keys.issubset(obj.keys()):
        raise ValueError(f"{path} is not a valid IAD dataset pickle")

    train_data = np.asarray(obj["train_data"], dtype=np.float32)
    train_label = np.asarray(obj["train_label"], dtype=np.int64)
    test_data = np.asarray(obj["test_data"], dtype=np.float32)
    test_label = np.asarray(obj["test_label"], dtype=np.int64)

    if train_data.ndim != 4 or test_data.ndim != 4:
        raise ValueError("Expected train_data/test_data to have shape [N, C, H, W]")

    train_unique = sorted(np.unique(train_label).tolist())
    test_unique = sorted(np.unique(test_label).tolist())
    if train_unique != [0]:
        raise ValueError(
            f"PatchCore expects normal-only training data, got train labels: {train_unique}"
        )
    if test_unique != [0, 1]:
        raise ValueError(f"Expected binary test labels [0, 1], got {test_unique}")

    mean = float(train_data.mean())
    std = float(train_data.std())
    if std <= 0:
        raise ValueError("Training data std must be > 0")

    return DatasetBundle(
        train_data=train_data,
        train_label=train_label,
        test_data=test_data,
        test_label=test_label,
        mean=mean,
        std=std,
    )


class IADSpectrogramDataset(Dataset):
    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        mean: float,
        std: float,
        clip_z: float,
    ) -> None:
        self.data = data
        self.labels = labels
        self.mean = mean
        self.std = std
        self.clip_z = clip_z

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def __getitem__(self, index: int):
        x = torch.from_numpy(self.data[index]).float()
        x = (x - self.mean) / self.std
        if self.clip_z > 0:
            x = torch.clamp(x, -self.clip_z, self.clip_z)
        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1)
        label = int(self.labels[index])
        return x, label, index


class ResNetFeatureExtractor(nn.Module):
    def __init__(
        self, backbone: str, layers: list[str], pretrained: bool = True
    ) -> None:
        super().__init__()
        self.layers = layers
        if backbone == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            base = resnet18(weights=weights)
        elif backbone == "wide_resnet50_2":
            weights = Wide_ResNet50_2_Weights.DEFAULT if pretrained else None
            base = wide_resnet50_2(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = {}

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        if "layer1" in self.layers:
            outputs["layer1"] = x

        x = self.layer2(x)
        if "layer2" in self.layers:
            outputs["layer2"] = x

        x = self.layer3(x)
        if "layer3" in self.layers:
            outputs["layer3"] = x

        x = self.layer4(x)
        if "layer4" in self.layers:
            outputs["layer4"] = x

        return outputs


class PatchCoreModel:
    def __init__(
        self,
        backbone: str,
        layers: list[str],
        device: torch.device,
        coreset_ratio: float,
        max_memory_bank_size: int,
        candidate_pool_size: int,
        projection_dim: int,
        query_chunk_size: int,
        seed: int,
        pretrained: bool = True,
    ) -> None:
        self.backbone = backbone
        self.layers = layers
        self.device = device
        self.coreset_ratio = coreset_ratio
        self.max_memory_bank_size = max_memory_bank_size
        self.candidate_pool_size = candidate_pool_size
        self.projection_dim = projection_dim
        self.query_chunk_size = query_chunk_size
        self.seed = seed
        self.feature_extractor = ResNetFeatureExtractor(
            backbone=backbone, layers=layers, pretrained=pretrained
        ).to(device)
        self.feature_extractor.eval()
        self.memory_bank: torch.Tensor | None = None
        self.memory_bank_norms: torch.Tensor | None = None
        self.embedding_dim: int | None = None
        self.patch_grid_size: tuple[int, int] | None = None
        self.total_train_patches: int | None = None

    def _compute_patch_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(images)
        feature_maps = [features[layer] for layer in self.layers]
        reference_size = feature_maps[0].shape[-2:]

        aligned = []
        for feature_map in feature_maps:
            if feature_map.shape[-2:] != reference_size:
                feature_map = F.interpolate(
                    feature_map,
                    size=reference_size,
                    mode="bilinear",
                    align_corners=False,
                )
            aligned.append(feature_map)

        embedding_map = torch.cat(aligned, dim=1)
        self.patch_grid_size = tuple(int(v) for v in embedding_map.shape[-2:])
        patch_embeddings = embedding_map.permute(0, 2, 3, 1).reshape(
            embedding_map.shape[0], -1, embedding_map.shape[1]
        )
        self.embedding_dim = int(patch_embeddings.shape[-1])
        return patch_embeddings

    def extract_embeddings(self, loader: DataLoader) -> torch.Tensor:
        all_embeddings = []
        total_images = 0
        total_batches = len(loader)

        with torch.no_grad():
            for batch_idx, (images, _, _) in enumerate(loader, start=1):
                images = images.to(self.device, non_blocking=True)
                batch_embeddings = self._compute_patch_embeddings(images)
                all_embeddings.append(batch_embeddings.cpu())
                total_images += images.shape[0]
                if batch_idx == 1 or batch_idx % 10 == 0 or batch_idx == total_batches:
                    log(
                        f"Extracted train features for {total_images} images "
                        f"({batch_idx}/{total_batches} batches)"
                    )

        embeddings = torch.cat(all_embeddings, dim=0).reshape(-1, self.embedding_dim)
        self.total_train_patches = int(embeddings.shape[0])
        return embeddings.float()

    def _kcenter_greedy(
        self, candidate_embeddings: torch.Tensor, num_select: int
    ) -> torch.Tensor:
        candidate_embeddings = candidate_embeddings.to(self.device, non_blocking=True)
        num_candidates = candidate_embeddings.shape[0]

        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.seed)
        first_index = int(
            torch.randint(
                low=0,
                high=num_candidates,
                size=(1,),
                generator=generator,
                device=self.device,
            ).item()
        )

        selected = torch.empty(num_select, dtype=torch.long, device=self.device)
        selected[0] = first_index
        center = candidate_embeddings[first_index]
        min_distances = ((candidate_embeddings - center) ** 2).sum(dim=1)

        log(
            f"Running k-center greedy on {num_candidates} candidates to select "
            f"{num_select} memory vectors"
        )

        for i in range(1, num_select):
            next_index = torch.argmax(min_distances)
            selected[i] = next_index
            center = candidate_embeddings[next_index]
            distances = ((candidate_embeddings - center) ** 2).sum(dim=1)
            min_distances = torch.minimum(min_distances, distances)
            if i % 250 == 0 or i == num_select - 1:
                log(f"k-center progress: {i + 1}/{num_select}")

        return selected.cpu()

    def build_memory_bank(self, embeddings: torch.Tensor) -> None:
        num_embeddings = embeddings.shape[0]
        target_size = max(1, int(math.ceil(num_embeddings * self.coreset_ratio)))
        target_size = min(target_size, self.max_memory_bank_size, num_embeddings)

        if target_size >= num_embeddings:
            selected_embeddings = embeddings
            log("Skipping coreset subsampling because target size covers all patches")
        else:
            rng = np.random.default_rng(self.seed)
            if self.candidate_pool_size and self.candidate_pool_size < num_embeddings:
                candidate_indices = np.sort(
                    rng.choice(
                        num_embeddings,
                        size=self.candidate_pool_size,
                        replace=False,
                    )
                )
            else:
                candidate_indices = np.arange(num_embeddings)

            if target_size > candidate_indices.shape[0]:
                target_size = int(candidate_indices.shape[0])

            candidate_embeddings = embeddings[candidate_indices]

            if self.projection_dim > 0 and candidate_embeddings.shape[1] > self.projection_dim:
                projector = SparseRandomProjection(
                    n_components=self.projection_dim, random_state=self.seed
                )
                projected = projector.fit_transform(candidate_embeddings.numpy())
                candidate_for_selection = torch.from_numpy(
                    np.asarray(projected, dtype=np.float32)
                )
            else:
                candidate_for_selection = candidate_embeddings

            selected_relative = self._kcenter_greedy(
                candidate_for_selection, num_select=target_size
            ).numpy()
            selected_indices = candidate_indices[selected_relative]
            selected_embeddings = embeddings[selected_indices]

        self.memory_bank = selected_embeddings.to(self.device, non_blocking=True)
        self.memory_bank_norms = (self.memory_bank**2).sum(dim=1)
        log(
            f"Memory bank ready: {self.memory_bank.shape[0]} vectors "
            f"from {num_embeddings} total train patches"
        )

    def fit(self, train_loader: DataLoader) -> None:
        embeddings = self.extract_embeddings(train_loader)
        self.build_memory_bank(embeddings)

    def _nearest_neighbor_distances(self, queries: torch.Tensor) -> torch.Tensor:
        if self.memory_bank is None or self.memory_bank_norms is None:
            raise RuntimeError("Memory bank is not initialized")

        results = []
        memory_bank_t = self.memory_bank.t().contiguous()

        for start in range(0, queries.shape[0], self.query_chunk_size):
            end = min(start + self.query_chunk_size, queries.shape[0])
            query_chunk = queries[start:end].to(self.device, non_blocking=True)
            query_norms = (query_chunk**2).sum(dim=1, keepdim=True)
            distances = (
                query_norms
                + self.memory_bank_norms.unsqueeze(0)
                - 2.0 * query_chunk @ memory_bank_t
            )
            distances = torch.clamp(distances, min=0.0)
            min_distances = torch.sqrt(distances.min(dim=1).values)
            results.append(min_distances.cpu())

        return torch.cat(results, dim=0)

    def predict(self, test_loader: DataLoader):
        if self.memory_bank is None:
            raise RuntimeError("Call fit() before predict()")

        all_scores = []
        all_labels = []
        all_indices = []
        all_maps = []
        total_batches = len(test_loader)

        with torch.no_grad():
            for batch_idx, (images, labels, indices) in enumerate(test_loader, start=1):
                raw_height, raw_width = int(images.shape[-2]), int(images.shape[-1])
                images = images.to(self.device, non_blocking=True)
                patch_embeddings = self._compute_patch_embeddings(images)
                batch_size, num_patches, embedding_dim = patch_embeddings.shape
                flat_embeddings = patch_embeddings.reshape(-1, embedding_dim)
                patch_scores = self._nearest_neighbor_distances(flat_embeddings)
                patch_scores = patch_scores.reshape(batch_size, *self.patch_grid_size)
                image_scores = patch_scores.amax(dim=(1, 2))
                anomaly_maps = F.interpolate(
                    patch_scores.unsqueeze(1),
                    size=(raw_height, raw_width),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)

                all_scores.append(image_scores.cpu())
                all_labels.append(labels.cpu())
                all_indices.append(indices.cpu())
                all_maps.append(anomaly_maps.cpu())

                if batch_idx == 1 or batch_idx % 10 == 0 or batch_idx == total_batches:
                    log(
                        f"Scored test images for batch {batch_idx}/{total_batches}"
                    )

        scores = torch.cat(all_scores).numpy()
        labels = torch.cat(all_labels).numpy()
        indices = torch.cat(all_indices).numpy()
        anomaly_maps = torch.cat(all_maps).numpy()
        return scores, labels, indices, anomaly_maps

    def export_test_patch_artifacts(
        self,
        test_loader: DataLoader,
        output_path: str,
        selection: str = "label1",
        embedding_dtype: str = "float16",
    ) -> dict:
        if self.memory_bank is None:
            raise RuntimeError("Call fit() before export_test_patch_artifacts()")

        if selection not in {"all", "label1", "prediction1"}:
            raise ValueError(f"Unsupported export selection: {selection}")
        if embedding_dtype not in {"float16", "float32"}:
            raise ValueError(f"Unsupported embedding dtype: {embedding_dtype}")

        selected_indices = []
        selected_labels = []
        selected_scores = []
        selected_patch_scores = []
        selected_patch_embeddings = []

        total_batches = len(test_loader)
        with torch.no_grad():
            for batch_idx, (images, labels, indices) in enumerate(test_loader, start=1):
                images = images.to(self.device, non_blocking=True)
                labels_np = labels.cpu().numpy()
                indices_np = indices.cpu().numpy()

                patch_embeddings = self._compute_patch_embeddings(images)
                batch_size, _, embedding_dim = patch_embeddings.shape
                flat_embeddings = patch_embeddings.reshape(-1, embedding_dim)
                patch_scores = self._nearest_neighbor_distances(flat_embeddings)
                patch_scores = patch_scores.reshape(batch_size, *self.patch_grid_size)
                image_scores = patch_scores.amax(dim=(1, 2)).cpu().numpy()

                if selection == "all":
                    keep_mask = np.ones(batch_size, dtype=bool)
                elif selection == "label1":
                    keep_mask = labels_np == 1
                else:
                    keep_mask = image_scores >= 0
                    raise RuntimeError(
                        "prediction1 export requires predictions after threshold selection; "
                        "use label1 or all in this pipeline"
                    )

                keep_indices = np.where(keep_mask)[0]
                if keep_indices.size > 0:
                    embeddings_np = patch_embeddings[keep_indices].reshape(
                        keep_indices.size,
                        self.patch_grid_size[0],
                        self.patch_grid_size[1],
                        embedding_dim,
                    ).cpu().numpy()
                    patch_scores_np = patch_scores[keep_indices].cpu().numpy()
                    if embedding_dtype == "float16":
                        embeddings_np = embeddings_np.astype(np.float16, copy=False)
                        patch_scores_np = patch_scores_np.astype(np.float16, copy=False)
                    else:
                        embeddings_np = embeddings_np.astype(np.float32, copy=False)
                        patch_scores_np = patch_scores_np.astype(np.float32, copy=False)

                    selected_patch_embeddings.append(embeddings_np)
                    selected_patch_scores.append(patch_scores_np)
                    selected_labels.append(labels_np[keep_indices].astype(np.int64, copy=False))
                    selected_indices.append(indices_np[keep_indices].astype(np.int64, copy=False))
                    selected_scores.append(image_scores[keep_indices].astype(np.float32, copy=False))

                if batch_idx == 1 or batch_idx % 10 == 0 or batch_idx == total_batches:
                    log(
                        f"Prepared patch artifact export for batch {batch_idx}/{total_batches}"
                    )

        if selected_patch_embeddings:
            patch_embeddings_out = np.concatenate(selected_patch_embeddings, axis=0)
            patch_scores_out = np.concatenate(selected_patch_scores, axis=0)
            labels_out = np.concatenate(selected_labels, axis=0)
            indices_out = np.concatenate(selected_indices, axis=0)
            image_scores_out = np.concatenate(selected_scores, axis=0)
        else:
            patch_embeddings_out = np.empty(
                (0, self.patch_grid_size[0], self.patch_grid_size[1], self.embedding_dim),
                dtype=np.float16 if embedding_dtype == "float16" else np.float32,
            )
            patch_scores_out = np.empty(
                (0, self.patch_grid_size[0], self.patch_grid_size[1]),
                dtype=np.float16 if embedding_dtype == "float16" else np.float32,
            )
            labels_out = np.empty((0,), dtype=np.int64)
            indices_out = np.empty((0,), dtype=np.int64)
            image_scores_out = np.empty((0,), dtype=np.float32)

        np.savez_compressed(
            output_path,
            sample_indices=indices_out,
            labels=labels_out,
            image_scores=image_scores_out,
            patch_scores=patch_scores_out,
            patch_embeddings=patch_embeddings_out,
            patch_grid_size=np.asarray(self.patch_grid_size, dtype=np.int64),
            embedding_dim=np.asarray([self.embedding_dim], dtype=np.int64),
        )

        return {
            "num_selected_samples": int(indices_out.shape[0]),
            "patch_grid_size": self.patch_grid_size,
            "embedding_dim": int(self.embedding_dim),
            "selection": selection,
            "output_path": output_path,
        }


def save_roc_curve(y_true: np.ndarray, y_score: np.ndarray, output_path: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auroc = roc_auc_score(y_true, y_score)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUROC = {auroc:.4f}", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_confusion_matrix(cm: np.ndarray, output_path: str) -> None:
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred Normal", "Pred Anomaly"],
        yticklabels=["True Normal", "True Anomaly"],
    )
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_anomaly_montage(
    raw_test_data: np.ndarray,
    anomaly_maps: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    output_path: str,
    num_images: int,
) -> list[int]:
    anomaly_indices = np.where(labels == 1)[0]
    if anomaly_indices.size == 0:
        raise ValueError("No anomaly samples were found in the test set")

    ranked = anomaly_indices[np.argsort(scores[anomaly_indices])[::-1]]
    selected = ranked[: min(num_images, ranked.size)]
    cols = 5
    rows = int(math.ceil(selected.size / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for plot_idx, sample_idx in enumerate(selected):
        ax = axes[plot_idx]
        image = raw_test_data[sample_idx, 0]
        heatmap = anomaly_maps[sample_idx]
        ax.imshow(image, cmap="gray")
        ax.imshow(heatmap, cmap="jet", alpha=0.45)
        ax.set_title(f"idx={sample_idx} score={scores[sample_idx]:.3f}")
        ax.axis("off")

    fig.suptitle("Top Anomalous Test Spectrograms", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return selected.tolist()


def save_score_csv(
    scores: np.ndarray,
    labels: np.ndarray,
    preds: np.ndarray,
    output_path: str,
) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("index,label,score,prediction\n")
        for index, (label, score, pred) in enumerate(zip(labels, scores, preds)):
            f.write(f"{index},{int(label)},{float(score):.8f},{int(pred)}\n")


def evaluate_scores(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    auroc = float(roc_auc_score(y_true, y_score))
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    optimal_index = int(np.argmax(tpr - fpr))
    threshold = float(thresholds[optimal_index])
    predictions = (y_score >= threshold).astype(np.int64)
    cm = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    return {
        "auroc": auroc,
        "threshold": threshold,
        "confusion_matrix": cm,
        "predictions": predictions,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "tpr": float(tp / max(tp + fn, 1)),
        "tnr": float(tn / max(tn + fp, 1)),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate PatchCore on IAD spectrogram data"
    )
    parser.add_argument(
        "--dataset-pkl",
        required=True,
        help="Path to a *_Train_Test.pkl file",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where metrics and figures will be saved",
    )
    parser.add_argument(
        "--backbone",
        default="resnet18",
        choices=["resnet18", "wide_resnet50_2"],
        help="CNN backbone used for PatchCore feature extraction",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        default=["layer2", "layer3"],
        choices=["layer1", "layer2", "layer3", "layer4"],
        help="Backbone feature layers used to form PatchCore embeddings",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clip-z", type=float, default=5.0)
    parser.add_argument("--coreset-ratio", type=float, default=0.01)
    parser.add_argument("--max-memory-bank-size", type=int, default=2048)
    parser.add_argument("--candidate-pool-size", type=int, default=20000)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--query-chunk-size", type=int, default=2048)
    parser.add_argument("--num-vis", type=int, default=10)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on, e.g. cuda or cpu",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable ImageNet pretrained backbone weights",
    )
    parser.add_argument(
        "--export-test-patch-artifacts",
        action="store_true",
        help="Export per-sample patch embeddings and patch scores for selected test samples",
    )
    parser.add_argument(
        "--export-selection",
        default="label1",
        choices=["all", "label1"],
        help="Subset of test samples whose patch artifacts will be exported",
    )
    parser.add_argument(
        "--export-embedding-dtype",
        default="float16",
        choices=["float16", "float32"],
        help="Storage dtype for exported patch embeddings",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    log(f"Loading dataset from {args.dataset_pkl}")
    bundle = load_iad_dataset(args.dataset_pkl)
    log(
        f"Train shape: {bundle.train_data.shape}, Test shape: {bundle.test_data.shape}, "
        f"train mean={bundle.mean:.6f}, train std={bundle.std:.6f}"
    )

    train_dataset = IADSpectrogramDataset(
        bundle.train_data,
        bundle.train_label,
        mean=bundle.mean,
        std=bundle.std,
        clip_z=args.clip_z,
    )
    test_dataset = IADSpectrogramDataset(
        bundle.test_data,
        bundle.test_label,
        mean=bundle.mean,
        std=bundle.std,
        clip_z=args.clip_z,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = PatchCoreModel(
        backbone=args.backbone,
        layers=args.layers,
        device=device,
        coreset_ratio=args.coreset_ratio,
        max_memory_bank_size=args.max_memory_bank_size,
        candidate_pool_size=args.candidate_pool_size,
        projection_dim=args.projection_dim,
        query_chunk_size=args.query_chunk_size,
        seed=args.seed,
        pretrained=not args.no_pretrained,
    )

    log(f"Building PatchCore memory bank on device={device}")
    fit_start = time.time()
    model.fit(train_loader)
    fit_seconds = time.time() - fit_start

    log("Scoring test set")
    predict_start = time.time()
    scores, labels, indices, anomaly_maps = model.predict(test_loader)
    predict_seconds = time.time() - predict_start

    if not np.array_equal(indices, np.arange(indices.shape[0])):
        raise ValueError("Unexpected test index ordering")

    eval_result = evaluate_scores(labels, scores)

    save_roc_curve(
        labels,
        scores,
        output_path=os.path.join(args.output_dir, "roc_curve.png"),
    )
    save_confusion_matrix(
        eval_result["confusion_matrix"],
        output_path=os.path.join(args.output_dir, "confusion_matrix.png"),
    )
    selected_vis_indices = save_anomaly_montage(
        raw_test_data=bundle.test_data,
        anomaly_maps=anomaly_maps,
        scores=scores,
        labels=labels,
        output_path=os.path.join(args.output_dir, "anomaly_montage_top10.png"),
        num_images=args.num_vis,
    )
    save_score_csv(
        scores,
        labels,
        eval_result["predictions"],
        output_path=os.path.join(args.output_dir, "test_scores.csv"),
    )
    np.savez_compressed(
        os.path.join(args.output_dir, "anomaly_outputs.npz"),
        scores=scores,
        labels=labels,
        anomaly_maps=anomaly_maps,
    )

    patch_export_info = None
    if args.export_test_patch_artifacts:
        log("Exporting test patch embeddings and patch scores")
        patch_export_info = model.export_test_patch_artifacts(
            test_loader=test_loader,
            output_path=os.path.join(args.output_dir, "test_patch_artifacts.npz"),
            selection=args.export_selection,
            embedding_dtype=args.export_embedding_dtype,
        )

    metrics = {
        "dataset_pkl": args.dataset_pkl,
        "output_dir": args.output_dir,
        "device": args.device,
        "backbone": args.backbone,
        "layers": args.layers,
        "train_samples": int(bundle.train_data.shape[0]),
        "test_samples": int(bundle.test_data.shape[0]),
        "train_mean": bundle.mean,
        "train_std": bundle.std,
        "embedding_dim": model.embedding_dim,
        "patch_grid_size": model.patch_grid_size,
        "total_train_patches": model.total_train_patches,
        "memory_bank_size": int(model.memory_bank.shape[0]),
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "auroc": eval_result["auroc"],
        "threshold": eval_result["threshold"],
        "tn": eval_result["tn"],
        "fp": eval_result["fp"],
        "fn": eval_result["fn"],
        "tp": eval_result["tp"],
        "tpr": eval_result["tpr"],
        "tnr": eval_result["tnr"],
        "selected_visualization_indices": selected_vis_indices,
        "patch_export": patch_export_info,
    }

    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    torch.save(
        {
            "memory_bank": model.memory_bank.cpu(),
            "layers": args.layers,
            "embedding_dim": model.embedding_dim,
            "patch_grid_size": model.patch_grid_size,
            "mean": bundle.mean,
            "std": bundle.std,
            "clip_z": args.clip_z,
        },
        os.path.join(args.output_dir, "patchcore_memory_bank.pt"),
    )

    log(
        f"Finished. AUROC={metrics['auroc']:.4f}, threshold={metrics['threshold']:.6f}, "
        f"TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']} TP={metrics['tp']}"
    )
    log(f"Artifacts saved to {args.output_dir}")


if __name__ == "__main__":
    main()
