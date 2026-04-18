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
