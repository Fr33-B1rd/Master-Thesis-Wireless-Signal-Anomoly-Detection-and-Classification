# Implements the PatchCore workflow for WSAD spectrogram arrays, including
# normalization, memory-bank construction, scoring, visualization, and exports.

import argparse
import json
import math
import os
import time
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
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
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class WSADBundle:
    train_data: np.ndarray
    test_parts: list[np.ndarray]
    test_labels: np.ndarray
    train_count: int
    test_count: int


def signed_log1p_np(array: np.ndarray) -> np.ndarray:
    return np.sign(array) * np.log1p(np.abs(array))


def compute_finite_mean_std(
    array: np.ndarray, chunk_size: int = 64, use_log_transform: bool = False
) -> tuple[float, float]:
    total_sum = 0.0
    total_sq_sum = 0.0
    total_count = 0
    num_samples = int(array.shape[0])
    for start in range(0, num_samples, chunk_size):
        end = min(start + chunk_size, num_samples)
        chunk = np.asarray(array[start:end], dtype=np.float32)
        finite_mask = np.isfinite(chunk)
        finite_values = chunk[finite_mask]
        if finite_values.size == 0:
            continue
        if use_log_transform:
            finite_values = signed_log1p_np(finite_values)
        total_sum += float(finite_values.sum(dtype=np.float64))
        total_sq_sum += float(np.square(finite_values, dtype=np.float64).sum(dtype=np.float64))
        total_count += int(finite_values.size)

    if total_count == 0:
        raise ValueError("No finite values found in dataset")

    mean = total_sum / total_count
    variance = max(total_sq_sum / total_count - mean * mean, 1e-12)
    return float(mean), float(np.sqrt(variance))


def load_wsad_bundle(dataset_dir: str) -> WSADBundle:
    train_data = np.load(os.path.join(dataset_dir, "train_normal.npy"), mmap_mode="r")
    test_normal = np.load(os.path.join(dataset_dir, "test_normal.npy"), mmap_mode="r")
    test_abnormal = np.load(os.path.join(dataset_dir, "test_abnormal.npy"), mmap_mode="r")

    train_csv = pd.read_csv(os.path.join(dataset_dir, "train_normal.csv"))
    test_normal_csv = pd.read_csv(os.path.join(dataset_dir, "test_normal.csv"))
    test_abnormal_csv = pd.read_csv(os.path.join(dataset_dir, "test_abnormal.csv"))

    train_labels = train_csv["label"].to_numpy(dtype=np.int64)
    if sorted(np.unique(train_labels).tolist()) != [0]:
        raise ValueError("train_normal split must contain only label 0")

    test_labels = np.concatenate(
        [
            test_normal_csv["label"].to_numpy(dtype=np.int64),
            test_abnormal_csv["label"].to_numpy(dtype=np.int64),
        ]
    )
    if sorted(np.unique(test_labels).tolist()) != [0, 1]:
        raise ValueError("Combined test split must contain labels 0 and 1")

    return WSADBundle(
        train_data=train_data,
        test_parts=[test_normal, test_abnormal],
        test_labels=test_labels,
        train_count=int(train_data.shape[0]),
        test_count=int(test_labels.shape[0]),
    )


def compute_wsad_normalization_stats(
    train_data: np.ndarray, normalization: str
) -> tuple[float, float]:
    if normalization == "zscore":
        return compute_finite_mean_std(train_data, use_log_transform=False)
    if normalization == "log_zscore":
        return compute_finite_mean_std(train_data, use_log_transform=True)
    if normalization in {"per_image_zscore", "log_per_image_zscore"}:
        return 0.0, 1.0
    raise ValueError(f"Unsupported normalization mode: {normalization}")


class WSADArrayDataset(Dataset):
    def __init__(
        self,
        data_parts: list[np.ndarray],
        labels: np.ndarray,
        mean: float,
        std: float,
        input_size: int,
        clip_z: float,
        normalization: str,
        mask_top_rows: int = 0,
        mask_bottom_rows: int = 0,
        mask_left_cols: int = 0,
        mask_right_cols: int = 0,
    ) -> None:
        self.data_parts = data_parts
        self.labels = labels
        self.mean = mean
        self.std = std
        self.input_size = input_size
        self.clip_z = clip_z
        self.normalization = normalization
        self.mask_top_rows = max(int(mask_top_rows), 0)
        self.mask_bottom_rows = max(int(mask_bottom_rows), 0)
        self.mask_left_cols = max(int(mask_left_cols), 0)
        self.mask_right_cols = max(int(mask_right_cols), 0)
        self.lengths = [int(part.shape[0]) for part in data_parts]
        self.cum_lengths = np.cumsum(self.lengths)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def _resolve_part(self, index: int):
        part_idx = int(np.searchsorted(self.cum_lengths, index, side="right"))
        prev = 0 if part_idx == 0 else int(self.cum_lengths[part_idx - 1])
        local_idx = index - prev
        return self.data_parts[part_idx], local_idx

    def _get_raw(self, index: int) -> np.ndarray:
        part, local_idx = self._resolve_part(index)
        return np.asarray(part[local_idx], dtype=np.float32)

    def get_processed_image(self, index: int) -> np.ndarray:
        x = self._prepare_tensor(self._get_raw(index))
        return x[0].cpu().numpy()

    def _prepare_tensor(self, array_2d: np.ndarray) -> torch.Tensor:
        array_2d = np.nan_to_num(
            array_2d,
            nan=self.mean,
            posinf=self.mean,
            neginf=self.mean,
        )
        if (
            self.mask_top_rows
            or self.mask_bottom_rows
            or self.mask_left_cols
            or self.mask_right_cols
        ):
            array_2d = np.array(array_2d, copy=True)
            if self.mask_top_rows:
                array_2d[: self.mask_top_rows, :] = self.mean
            if self.mask_bottom_rows:
                array_2d[-self.mask_bottom_rows :, :] = self.mean
            if self.mask_left_cols:
                array_2d[:, : self.mask_left_cols] = self.mean
            if self.mask_right_cols:
                array_2d[:, -self.mask_right_cols :] = self.mean
        if self.normalization in {"log_zscore", "log_per_image_zscore"}:
            array_2d = signed_log1p_np(array_2d).astype(np.float32, copy=False)
        x = torch.from_numpy(array_2d).float().unsqueeze(0)
        if self.normalization in {"zscore", "log_zscore"}:
            x = (x - self.mean) / self.std
        elif self.normalization in {"per_image_zscore", "log_per_image_zscore"}:
            img_mean = x.mean()
            img_std = torch.clamp(x.std(unbiased=False), min=1e-6)
            x = (x - img_mean) / img_std
        else:
            raise ValueError(f"Unsupported normalization mode: {self.normalization}")
        if self.clip_z > 0:
            x = torch.clamp(x, -self.clip_z, self.clip_z)
        if self.input_size > 0 and tuple(x.shape[-2:]) != (self.input_size, self.input_size):
            x = F.interpolate(
                x.unsqueeze(0),
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        x = x.repeat(3, 1, 1)
        return x

    def __getitem__(self, index: int):
        raw = self._get_raw(index)
        x = self._prepare_tensor(raw)
        label = int(self.labels[index])
        return x, label, index


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, backbone: str, layers: list[str], pretrained: bool = True) -> None:
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


class WSADPatchCore:
    def __init__(
        self,
        backbone: str,
        layers: list[str],
        device: torch.device,
        sample_patches_per_image: int,
        candidate_pool_size: int,
        coreset_ratio: float,
        max_memory_bank_size: int,
        projection_dim: int,
        query_chunk_size: int,
        seed: int,
        pretrained: bool = True,
    ) -> None:
        self.backbone = backbone
        self.layers = layers
        self.device = device
        self.sample_patches_per_image = sample_patches_per_image
        self.candidate_pool_size = candidate_pool_size
        self.coreset_ratio = coreset_ratio
        self.max_memory_bank_size = max_memory_bank_size
        self.projection_dim = projection_dim
        self.query_chunk_size = query_chunk_size
        self.seed = seed
        self.feature_extractor = ResNetFeatureExtractor(
            backbone=backbone, layers=layers, pretrained=pretrained
        ).to(device)
        self.feature_extractor.eval()
        self.patch_grid_size = None
        self.embedding_dim = None
        self.total_train_patches = 0
        self.memory_bank = None
        self.memory_bank_norms = None

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

    def _sample_candidate_patches(self, patch_embeddings: torch.Tensor) -> torch.Tensor:
        batch_size, num_patches, embedding_dim = patch_embeddings.shape
        sample_count = min(self.sample_patches_per_image, num_patches)
        sampled = []
        for batch_index in range(batch_size):
            perm = torch.randperm(num_patches, device=patch_embeddings.device)[:sample_count]
            sampled.append(patch_embeddings[batch_index, perm])
        return torch.stack(sampled, dim=0).reshape(-1, embedding_dim)

    def _kcenter_greedy(self, candidate_embeddings: torch.Tensor, num_select: int) -> torch.Tensor:
        candidate_embeddings = candidate_embeddings.to(self.device, non_blocking=True)
        num_candidates = candidate_embeddings.shape[0]
        first_index = int(torch.randint(0, num_candidates, (1,), device=self.device).item())
        selected = torch.empty(num_select, dtype=torch.long, device=self.device)
        selected[0] = first_index
        center = candidate_embeddings[first_index]
        min_distances = ((candidate_embeddings - center) ** 2).sum(dim=1)

        for i in range(1, num_select):
            next_index = torch.argmax(min_distances)
            selected[i] = next_index
            center = candidate_embeddings[next_index]
            distances = ((candidate_embeddings - center) ** 2).sum(dim=1)
            min_distances = torch.minimum(min_distances, distances)
            if i % 250 == 0 or i == num_select - 1:
                log(f"k-center progress: {i + 1}/{num_select}")
        return selected.cpu()

    def fit(self, train_loader: DataLoader) -> None:
        sampled_candidates = []
        total_batches = len(train_loader)
        with torch.no_grad():
            for batch_idx, (images, _, _) in enumerate(train_loader, start=1):
                images = images.to(self.device, non_blocking=True)
                patch_embeddings = self._compute_patch_embeddings(images)
                self.total_train_patches += int(
                    patch_embeddings.shape[0] * patch_embeddings.shape[1]
                )
                sampled = self._sample_candidate_patches(patch_embeddings)
                sampled_candidates.append(sampled.cpu().to(torch.float16))
                if batch_idx == 1 or batch_idx % 20 == 0 or batch_idx == total_batches:
                    log(
                        f"Collected candidate patches from batch {batch_idx}/{total_batches}"
                    )

        candidates = torch.cat(sampled_candidates, dim=0).float()
        if candidates.shape[0] > self.candidate_pool_size:
            perm = torch.randperm(candidates.shape[0])[: self.candidate_pool_size]
            candidates = candidates[perm]

        target_size = max(1, int(math.ceil(candidates.shape[0] * self.coreset_ratio)))
        target_size = min(target_size, self.max_memory_bank_size, candidates.shape[0])

        if self.projection_dim > 0 and candidates.shape[1] > self.projection_dim:
            projector = SparseRandomProjection(
                n_components=self.projection_dim, random_state=self.seed
            )
            projected = projector.fit_transform(candidates.numpy())
            projected = torch.from_numpy(np.asarray(projected, dtype=np.float32))
        else:
            projected = candidates

        log(
            f"Running k-center on {candidates.shape[0]} sampled candidates to select {target_size}"
        )
        selected_indices = self._kcenter_greedy(projected, target_size)
        self.memory_bank = candidates[selected_indices].to(self.device, non_blocking=True)
        self.memory_bank_norms = (self.memory_bank**2).sum(dim=1)
        log(
            f"Memory bank ready: {self.memory_bank.shape[0]} vectors from "
            f"{self.total_train_patches} estimated train patches"
        )

    def _nearest_neighbor_distances(self, queries: torch.Tensor) -> torch.Tensor:
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
            results.append(torch.sqrt(distances.min(dim=1).values).cpu())
        return torch.cat(results, dim=0)

    def predict(self, test_loader: DataLoader):
        all_scores = []
        all_labels = []
        all_indices = []
        all_maps = []
        total_batches = len(test_loader)
        with torch.no_grad():
            for batch_idx, (images, labels, indices) in enumerate(test_loader, start=1):
                raw_h, raw_w = int(images.shape[-2]), int(images.shape[-1])
                images = images.to(self.device, non_blocking=True)
                patch_embeddings = self._compute_patch_embeddings(images)
                flat_embeddings = patch_embeddings.reshape(-1, self.embedding_dim)
                patch_scores = self._nearest_neighbor_distances(flat_embeddings)
                patch_scores = patch_scores.reshape(images.shape[0], *self.patch_grid_size)
                image_scores = patch_scores.amax(dim=(1, 2))
                anomaly_maps = F.interpolate(
                    patch_scores.unsqueeze(1),
                    size=(raw_h, raw_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)

                all_scores.append(image_scores.cpu())
                all_labels.append(labels.cpu())
                all_indices.append(indices.cpu())
                all_maps.append(anomaly_maps.cpu())
                if batch_idx == 1 or batch_idx % 20 == 0 or batch_idx == total_batches:
                    log(f"Scored batch {batch_idx}/{total_batches}")

        return (
            torch.cat(all_scores).numpy(),
            torch.cat(all_labels).numpy(),
            torch.cat(all_indices).numpy(),
            torch.cat(all_maps).numpy(),
        )

    def export_test_patch_artifacts(
        self,
        test_loader: DataLoader,
        output_path: str,
        selection: str = "label1",
        embedding_dtype: str = "float16",
    ) -> dict:
        if self.memory_bank is None:
            raise RuntimeError("Call fit() before export_test_patch_artifacts()")

        if selection not in {"all", "label1"}:
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
                else:
                    keep_mask = labels_np == 1

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

                if batch_idx == 1 or batch_idx % 20 == 0 or batch_idx == total_batches:
                    log(f"Prepared patch artifact export for batch {batch_idx}/{total_batches}")

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


def evaluate_scores(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    auroc = float(roc_auc_score(y_true, y_score))
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    best_idx = int(np.argmax(tpr - fpr))
    threshold = float(thresholds[best_idx])
    predictions = (y_score >= threshold).astype(np.int64)
    cm = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    return {
        "auroc": auroc,
        "threshold": threshold,
        "predictions": predictions,
        "confusion_matrix": cm,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
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
    dataset: WSADArrayDataset,
    anomaly_maps: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    output_path: str,
    num_images: int,
    label_names: np.ndarray | None = None,
    title_prefix: str | None = None,
) -> list[int]:
    grouped_mode = label_names is not None
    selected: list[int] = []

    if grouped_mode:
        label_names = np.asarray(label_names).astype(str)
        abnormal_label_names = [
            name
            for name in sorted(np.unique(label_names).tolist())
            if name != "normal"
        ]
        cols = max(1, int(num_images))
        rows = max(1, len(abnormal_label_names))
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
        axes = np.asarray(axes, dtype=object).reshape(rows, cols)
        for ax in axes.reshape(-1):
            ax.axis("off")

        for row_idx, label_name in enumerate(abnormal_label_names):
            class_indices = np.where(label_names == label_name)[0]
            ranked = class_indices[np.argsort(scores[class_indices])[::-1]]
            chosen = ranked[: min(num_images, ranked.size)]
            selected.extend(int(idx) for idx in chosen.tolist())
            for col_idx, sample_idx in enumerate(chosen):
                ax = axes[row_idx, col_idx]
                image = dataset.get_processed_image(int(sample_idx))
                heatmap = anomaly_maps[sample_idx]
                ax.imshow(image, cmap="gray")
                ax.imshow(heatmap, cmap="jet", alpha=0.45)
                ax.set_title(
                    f"{label_name}\nidx={sample_idx} score={scores[sample_idx]:.3f}",
                    fontsize=10,
                )
                ax.axis("off")
    else:
        anomaly_indices = np.where(labels == 1)[0]
        ranked = anomaly_indices[np.argsort(scores[anomaly_indices])[::-1]]
        selected = ranked[: min(num_images, ranked.size)].tolist()
        cols = 5
        rows = int(math.ceil(max(1, len(selected)) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
        axes = np.array(axes).reshape(-1)
        for ax in axes:
            ax.axis("off")

        for i, sample_idx in enumerate(selected):
            ax = axes[i]
            image = dataset.get_processed_image(int(sample_idx))
            heatmap = anomaly_maps[sample_idx]
            ax.imshow(image, cmap="gray")
            ax.imshow(heatmap, cmap="jet", alpha=0.45)
            ax.set_title(f"idx={sample_idx} score={scores[sample_idx]:.3f}")
            ax.axis("off")

    if title_prefix is None:
        title_prefix = "Top Abnormal WSAD Test Samples"
    fig.suptitle(title_prefix, fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return selected


def save_score_csv(scores: np.ndarray, labels: np.ndarray, preds: np.ndarray, output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("index,label,score,prediction\n")
        for idx, (label, score, pred) in enumerate(zip(labels, scores, preds)):
            f.write(f"{idx},{int(label)},{float(score):.8f},{int(pred)}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="PatchCore evaluation on WSAD splits")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--backbone",
        default="wide_resnet50_2",
        choices=["resnet18", "wide_resnet50_2"],
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        default=["layer2", "layer3"],
        choices=["layer1", "layer2", "layer3", "layer4"],
    )
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--clip-z", type=float, default=5.0)
    parser.add_argument(
        "--normalization",
        default="zscore",
        choices=["zscore", "log_zscore", "per_image_zscore", "log_per_image_zscore"],
    )
    parser.add_argument("--sample-patches-per-image", type=int, default=32)
    parser.add_argument("--candidate-pool-size", type=int, default=20000)
    parser.add_argument("--coreset-ratio", type=float, default=0.1)
    parser.add_argument("--max-memory-bank-size", type=int, default=2048)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--query-chunk-size", type=int, default=1024)
    parser.add_argument("--num-vis", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)

    bundle = load_wsad_bundle(args.dataset_dir)
    train_mean, train_std = compute_wsad_normalization_stats(
        bundle.train_data, args.normalization
    )
    log(
        f"WSAD loaded: train={bundle.train_count}, test={bundle.test_count}, "
        f"raw_shape={tuple(bundle.train_data.shape[1:])}, normalization={args.normalization}, "
        f"norm_mean={train_mean:.6f}, norm_std={train_std:.6f}"
    )

    train_dataset = WSADArrayDataset(
        data_parts=[bundle.train_data],
        labels=np.zeros(bundle.train_count, dtype=np.int64),
        mean=train_mean,
        std=train_std,
        input_size=args.input_size,
        clip_z=args.clip_z,
        normalization=args.normalization,
    )
    test_dataset = WSADArrayDataset(
        data_parts=bundle.test_parts,
        labels=bundle.test_labels,
        mean=train_mean,
        std=train_std,
        input_size=args.input_size,
        clip_z=args.clip_z,
        normalization=args.normalization,
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

    model = WSADPatchCore(
        backbone=args.backbone,
        layers=args.layers,
        device=device,
        sample_patches_per_image=args.sample_patches_per_image,
        candidate_pool_size=args.candidate_pool_size,
        coreset_ratio=args.coreset_ratio,
        max_memory_bank_size=args.max_memory_bank_size,
        projection_dim=args.projection_dim,
        query_chunk_size=args.query_chunk_size,
        seed=args.seed,
        pretrained=not args.no_pretrained,
    )

    fit_start = time.time()
    model.fit(train_loader)
    fit_seconds = time.time() - fit_start

    predict_start = time.time()
    scores, labels, indices, anomaly_maps = model.predict(test_loader)
    predict_seconds = time.time() - predict_start

    if not np.array_equal(indices, np.arange(indices.shape[0])):
        raise ValueError("Unexpected test ordering")

    eval_result = evaluate_scores(labels, scores)
    save_roc_curve(labels, scores, os.path.join(args.output_dir, "roc_curve.png"))
    save_confusion_matrix(
        eval_result["confusion_matrix"],
        os.path.join(args.output_dir, "confusion_matrix.png"),
    )
    vis_indices = save_anomaly_montage(
        dataset=test_dataset,
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
        os.path.join(args.output_dir, "test_scores.csv"),
    )
    np.savez_compressed(
        os.path.join(args.output_dir, "anomaly_outputs.npz"),
        scores=scores,
        labels=labels,
        anomaly_maps=anomaly_maps.astype(np.float16),
    )

    metrics = {
        "dataset_dir": args.dataset_dir,
        "output_dir": args.output_dir,
        "backbone": args.backbone,
        "layers": args.layers,
        "device": args.device,
        "input_size": args.input_size,
        "train_count": bundle.train_count,
        "test_count": bundle.test_count,
        "raw_input_shape": list(bundle.train_data.shape[1:]),
        "normalization": args.normalization,
        "train_mean": train_mean,
        "train_std": train_std,
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
        "selected_visualization_indices": vis_indices,
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    torch.save(
        {
            "memory_bank": model.memory_bank.cpu(),
            "layers": args.layers,
            "embedding_dim": model.embedding_dim,
            "patch_grid_size": model.patch_grid_size,
            "input_size": args.input_size,
            "normalization": args.normalization,
            "train_mean": train_mean,
            "train_std": train_std,
        },
        os.path.join(args.output_dir, "patchcore_memory_bank.pt"),
    )

    log(
        f"Finished WSAD PatchCore. AUROC={metrics['auroc']:.4f}, "
        f"TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']} TP={metrics['tp']}"
    )
    log(f"Artifacts saved to {args.output_dir}")


if __name__ == "__main__":
    main()
