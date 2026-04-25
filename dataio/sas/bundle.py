# SAS packed-spectrogram dataset loader.
#
# Expects a directory containing:
#   train_spectrogram.npy      (N, H, W) float32/uint8 — training signals (normal only)
#   test_spectrogram.npy       (M, H, W) uint8 [0,255]  — 1-channel test signals
#   train_labels.npy           (N,) string labels
#   test_labels.npy            (M,) string labels
#   test_spectrogram_3c.npy    (M, 3, H, W) float16    — optional, 3-channel pack
#   train_spectrogram_3c.npy   (N, 3, H, W) float16    — optional, 3-channel pack
#   channel_stats.json                                 — optional, per-channel mean/std
#
# Three access patterns:
#   - load_sas_bundle(dir)              train + test, for Stage 1 PatchCore
#   - load_sas_test_split(dir)          1-channel test split (Stage 2 default)
#   - load_sas_test_split_3c(dir)       3-channel test split (Stage 2 mag_if/mag_gd/mag_if_gd)
# Plus:
#   - load_channel_stats(dir)           channel_stats.json (needed by mag_1ch + 3c modes)

import json
import os
from dataclasses import dataclass

import numpy as np

from core.utils import compute_finite_mean_std


@dataclass
class SASBundle:
    train_data: np.ndarray
    test_data: np.ndarray
    train_labels: np.ndarray
    test_labels: np.ndarray
    test_label_names: np.ndarray
    train_count: int
    test_count: int
    mean: float
    std: float


@dataclass
class SASTestSplit:
    label_names: np.ndarray
    images: np.ndarray | None
    count: int


@dataclass
class SASTestSplit3C:
    label_names: np.ndarray
    images: np.ndarray
    count: int


def _load_test_labels(dataset_dir: str) -> np.ndarray:
    """Shared: load test_labels.npy and cast to str."""
    return np.load(os.path.join(dataset_dir, "test_labels.npy")).astype(str)


def load_sas_bundle(dataset_dir: str) -> SASBundle:
    """Full train+test bundle used by Stage 1 PatchCore. Enforces normal-only training."""
    train_data = np.load(
        os.path.join(dataset_dir, "train_spectrogram.npy"),
        mmap_mode="r",
    )
    test_data = np.load(
        os.path.join(dataset_dir, "test_spectrogram.npy"),
        mmap_mode="r",
    )
    train_label_names = np.load(os.path.join(dataset_dir, "train_labels.npy"))
    test_label_names = _load_test_labels(dataset_dir)

    if train_data.ndim != 3 or test_data.ndim != 3:
        raise ValueError("Expected SAS spectrogram arrays with shape [N, H, W]")

    train_label_names = np.asarray(train_label_names).astype(str)
    train_labels = (train_label_names != "normal").astype(np.int64)
    test_labels = (test_label_names != "normal").astype(np.int64)

    train_unique = sorted(np.unique(train_label_names).tolist())
    test_unique = sorted(np.unique(test_label_names).tolist())
    if train_unique != ["normal"]:
        raise ValueError(
            f"SAS PatchCore expects normal-only training data, got train labels: {train_unique}"
        )
    if "normal" not in test_unique or len(test_unique) < 2:
        raise ValueError(
            "Expected SAS test split to contain normal plus at least one abnormal class"
        )

    mean, std = compute_finite_mean_std(train_data, use_log_transform=False)
    if std <= 0:
        raise ValueError("Training data std must be > 0")

    return SASBundle(
        train_data=train_data,
        test_data=test_data,
        train_labels=train_labels,
        test_labels=test_labels,
        test_label_names=test_label_names,
        train_count=int(train_data.shape[0]),
        test_count=int(test_data.shape[0]),
        mean=mean,
        std=std,
    )


def load_sas_test_split(
    dataset_dir: str,
    *,
    load_images: bool = True,
) -> SASTestSplit:
    """Test-side only: 1-channel `test_spectrogram.npy` (uint8 H×W) + `test_labels.npy`.

    Set load_images=False when only labels are needed (e.g. few-shot training
    setup that reads patch artifacts instead of raw spectrograms).
    """
    label_names = _load_test_labels(dataset_dir)
    images: np.ndarray | None = None
    if load_images:
        images = np.load(os.path.join(dataset_dir, "test_spectrogram.npy"), mmap_mode="r")
    return SASTestSplit(
        label_names=label_names,
        images=images,
        count=int(label_names.shape[0]),
    )


def load_sas_test_split_3c(dataset_dir: str) -> SASTestSplit3C:
    """Test-side only: 3-channel `test_spectrogram_3c.npy` (M×3×H×W float16) + labels.

    Produced by `Datasets/SAS/build_rf_3channel.py`; file is not shipped with
    the packed directory by default. Raises with producer guidance if missing.
    """
    spec_path = os.path.join(dataset_dir, "test_spectrogram_3c.npy")
    if not os.path.exists(spec_path):
        raise FileNotFoundError(
            f"Missing {spec_path}. Run `Datasets/SAS/build_rf_3channel.py "
            f"--packed-dir {dataset_dir}` first to produce the 3-channel test pack."
        )
    label_names = _load_test_labels(dataset_dir)
    images = np.load(spec_path, mmap_mode="r")
    return SASTestSplit3C(
        label_names=label_names,
        images=images,
        count=int(label_names.shape[0]),
    )


def load_channel_stats(dataset_dir: str) -> dict:
    """Loads `channel_stats.json` produced by `Datasets/SAS/build_rf_3channel.py`.

    Used by Stage 2 in both the 3-channel modes (mag_if/mag_gd/mag_if_gd) and
    the true 1-channel mag_1ch mode (which needs Ch0 mean/std for normalization).
    """
    stats_path = os.path.join(dataset_dir, "channel_stats.json")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(
            f"Missing {stats_path}. Run `Datasets/SAS/build_rf_3channel.py "
            f"--packed-dir {dataset_dir}` first to produce per-channel statistics."
        )
    with open(stats_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_label_breakdown(
    label_names: np.ndarray,
    scores: np.ndarray,
    output_path: str,
) -> None:
    unique_labels = sorted(np.unique(label_names).tolist())
    rows = []
    for label_name in unique_labels:
        mask = label_names == label_name
        rows.append(
            {
                "label_name": label_name,
                "count": int(mask.sum()),
                "mean_score": float(scores[mask].mean()),
                "std_score": float(scores[mask].std()),
            }
        )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
