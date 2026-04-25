import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def compute_scale_gamma(x: np.ndarray) -> float:
    variance = float(np.var(x))
    if variance <= 0:
        return 1.0
    return 1.0 / (float(x.shape[1]) * variance)


def select_anchor_indices(
    train_indices_by_label: dict[str, list[int]],
    current_known_classes: list[str],
    excluded_indices: list[int],
    anchor_per_class: int,
    seed: int,
) -> list[int]:
    """Sample per-class anchor indices for HDBSCAN-based unknown-buffer analysis."""
    anchor_indices = []
    excluded = set(int(idx) for idx in excluded_indices)
    anchor_rng = np.random.default_rng(seed)
    for class_name in current_known_classes:
        available = [idx for idx in train_indices_by_label.get(class_name, []) if idx not in excluded]
        if not available:
            continue
        sample_count = min(anchor_per_class, len(available))
        selected = anchor_rng.choice(np.asarray(available, dtype=np.int64), size=sample_count, replace=False)
        anchor_indices.extend(selected.tolist())
    return sorted(set(int(idx) for idx in anchor_indices))


def save_rectangular_confusion(
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    output_path: str,
    title: str,
) -> None:
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=col_labels,
        yticklabels=row_labels,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_confusion(
    y_true: list[str],
    y_pred: list[str],
    label_order: list[str],
    output_path: str,
    title: str,
) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=label_order)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_order,
        yticklabels=label_order,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()
