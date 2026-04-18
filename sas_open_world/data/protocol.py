"""Flow/holdout split utilities for SAS open-world protocols."""

import numpy as np


def build_flow_holdout_split(
    label_array: np.ndarray,
    split_classes: list[str],
    flow_per_class: int,
    holdout_per_class: int,
    rng: np.random.Generator,
) -> tuple[dict[str, list[int]], dict[str, list[int]], dict[str, dict[str, int]], np.ndarray]:
    flow_indices_by_label: dict[str, list[int]] = {}
    holdout_indices_by_label: dict[str, list[int]] = {}
    split_summary: dict[str, dict[str, int]] = {}

    for class_name in split_classes:
        class_indices = np.where(label_array == class_name)[0]
        shuffled = rng.permutation(class_indices)
        required = flow_per_class + holdout_per_class
        if shuffled.shape[0] < required:
            raise ValueError(
                f"{class_name} has only {shuffled.shape[0]} samples, fewer than flow+holdout={required}"
            )
        flow_part = shuffled[:flow_per_class].astype(np.int64)
        holdout_part = shuffled[flow_per_class:required].astype(np.int64)
        flow_indices_by_label[class_name] = flow_part.tolist()
        holdout_indices_by_label[class_name] = holdout_part.tolist()
        split_summary[class_name] = {
            "flow_count": int(flow_part.shape[0]),
            "holdout_count": int(holdout_part.shape[0]),
        }

    normal_indices = np.where(label_array == "normal")[0].astype(np.int64)
    normal_stream_indices = rng.permutation(normal_indices).astype(np.int64)
    split_summary["normal"] = {
        "flow_count": int(normal_stream_indices.shape[0]),
        "holdout_count": 0,
    }
    return flow_indices_by_label, holdout_indices_by_label, split_summary, normal_stream_indices


def build_initial_training_indices(
    flow_indices_by_label: dict[str, list[int]],
    known_classes: list[str],
    shot: int,
) -> tuple[dict[str, list[int]], list[int]]:
    train_indices_by_label: dict[str, list[int]] = {}
    initial_train_indices: list[int] = []
    for class_name in known_classes:
        class_indices = np.asarray(flow_indices_by_label[class_name], dtype=np.int64)
        selected = class_indices[:shot]
        if selected.shape[0] < shot:
            raise ValueError(f"{class_name} has only {selected.shape[0]} samples, fewer than shot={shot}")
        train_indices_by_label[class_name] = selected.astype(np.int64).tolist()
        initial_train_indices.extend(train_indices_by_label[class_name])
    return train_indices_by_label, initial_train_indices
