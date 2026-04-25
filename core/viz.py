# AUROC/CM evaluation, ROC plot, confusion-matrix plot, anomaly-montage,
# score CSV writer. Shared across datasets.

import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from core.patchcore import ArrayDataset2D


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
    dataset: ArrayDataset2D,
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
        title_prefix = "Top Abnormal Test Samples"
    fig.suptitle(title_prefix, fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return selected


def save_score_csv(
    scores: np.ndarray, labels: np.ndarray, preds: np.ndarray, output_path: str
) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("index,label,score,prediction\n")
        for idx, (label, score, pred) in enumerate(zip(labels, scores, preds)):
            f.write(f"{idx},{int(label)},{float(score):.8f},{int(pred)}\n")
