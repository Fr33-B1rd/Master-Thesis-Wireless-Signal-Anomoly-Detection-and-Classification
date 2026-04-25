"""Protocol and metric helpers for SAS open-world simulation."""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def build_stream_indices(
    flow_indices_by_label: dict[str, list[int]],
    split_classes: list[str],
    initial_train_indices: list[int],
    normal_stream_indices: np.ndarray,
    rng: np.random.Generator,
) -> list[int]:
    initial_train_set = set(int(idx) for idx in initial_train_indices)
    abnormal_stream_indices = []
    for class_name in split_classes:
        for idx in flow_indices_by_label[class_name]:
            if int(idx) not in initial_train_set:
                abnormal_stream_indices.append(int(idx))
    stream_indices = np.concatenate(
        [
            np.asarray(abnormal_stream_indices, dtype=np.int64),
            normal_stream_indices,
        ]
    )
    return rng.permutation(stream_indices).astype(np.int64).tolist()


def compute_phase_metrics(
    records_df: pd.DataFrame,
    known_classes: list[str],
    emerging_classes: list[str],
) -> dict:
    metrics = {}
    known_mask = records_df["true_label"].isin(known_classes)
    emerging_mask = records_df["true_label"].isin(emerging_classes)

    if known_mask.any():
        known_true = records_df.loc[known_mask, "true_label"].tolist()
        known_pred = records_df.loc[known_mask, "pred_label"].tolist()
        accepted_mask = [pred in known_classes for pred in known_pred]
        accepted_true = [t for t, keep in zip(known_true, accepted_mask) if keep]
        accepted_pred = [p for p, keep in zip(known_pred, accepted_mask) if keep]
        coverage = float(np.mean(accepted_mask))
        if accepted_true:
            macro_f1 = float(
                f1_score(accepted_true, accepted_pred, labels=known_classes, average="macro", zero_division=0)
            )
        else:
            macro_f1 = 0.0
        metrics["known_coverage"] = coverage
        metrics["known_macro_f1"] = macro_f1

    if emerging_mask.any():
        per_class_unknown_rates = []
        for class_name in emerging_classes:
            class_mask = records_df["true_label"] == class_name
            if class_mask.any():
                unknown_rate = float(np.mean(records_df.loc[class_mask, "pred_label"].values == "unknown"))
                recognized_rate = float(np.mean(records_df.loc[class_mask, "pred_label"].values == class_name))
                metrics[f"{class_name}_unknown_rate"] = unknown_rate
                metrics[f"{class_name}_recognized_rate"] = recognized_rate
                per_class_unknown_rates.append(unknown_rate)
        if per_class_unknown_rates:
            metrics["avg_emerging_unknown_rate"] = float(np.mean(per_class_unknown_rates))

    metrics["overall_accuracy"] = float(accuracy_score(records_df["true_label"], records_df["pred_label"]))
    return metrics


def compute_holdout_metrics(
    records_df: pd.DataFrame,
    final_known_classes: list[str],
    all_eval_classes: list[str],
) -> dict:
    metrics = {
        "overall_accuracy": float(accuracy_score(records_df["true_label"], records_df["pred_label"])),
        "macro_f1_all_classes": float(
            f1_score(
                records_df["true_label"],
                records_df["pred_label"],
                labels=all_eval_classes,
                average="macro",
                zero_division=0,
            )
        ),
        "final_known_classes": final_known_classes,
    }
    for class_name in all_eval_classes:
        class_mask = records_df["true_label"] == class_name
        if class_mask.any():
            metrics[f"{class_name}_recognized_rate"] = float(
                np.mean(records_df.loc[class_mask, "pred_label"].values == class_name)
            )
            metrics[f"{class_name}_unknown_rate"] = float(
                np.mean(records_df.loc[class_mask, "pred_label"].values == "unknown")
            )
    return metrics
