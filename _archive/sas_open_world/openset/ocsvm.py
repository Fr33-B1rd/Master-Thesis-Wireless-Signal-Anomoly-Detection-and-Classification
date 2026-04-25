"""Classifier plus OCSVM-gate helpers for SAS open-set experiments."""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from sas_open_world.common import compute_scale_gamma
from support.fewshot_patchcore_classifier_sas import build_classifier


def train_open_set_model(
    x_all: np.ndarray,
    train_indices_by_label: dict[str, list[int]],
    known_classes: list[str],
    classifier_name: str,
    classifier_c: float,
    class_weight_name: str,
    target_class: str,
    target_gamma_multiplier: float,
    nu: float,
    seed: int,
) -> dict:
    ordered_train_indices = []
    train_labels = []
    class_to_id = {name: idx for idx, name in enumerate(known_classes)}
    for class_name in known_classes:
        class_indices = sorted(set(train_indices_by_label.get(class_name, [])))
        ordered_train_indices.extend(class_indices)
        train_labels.extend([class_to_id[class_name]] * len(class_indices))

    train_indices = np.asarray(ordered_train_indices, dtype=np.int64)
    y_train = np.asarray(train_labels, dtype=np.int64)
    clf = build_classifier(classifier_name, seed=seed)
    if classifier_name == "logreg":
        class_weight = None if class_weight_name == "none" else "balanced"
        clf.named_steps["clf"].set_params(C=classifier_c, class_weight=class_weight)
    clf.fit(x_all[train_indices], y_train)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_all[train_indices]).astype(np.float32)

    ocsvm_models = {}
    for class_name in known_classes:
        class_id = class_to_id[class_name]
        class_vectors = x_train_scaled[y_train == class_id]
        gamma_value = compute_scale_gamma(class_vectors)
        if class_name == target_class:
            gamma_value *= target_gamma_multiplier
        model = OneClassSVM(kernel="rbf", gamma=gamma_value, nu=nu)
        model.fit(class_vectors)
        ocsvm_models[class_name] = model

    return {
        "classifier": clf,
        "scaler": scaler,
        "ocsvm_models": ocsvm_models,
        "known_classes": known_classes,
        "train_indices": train_indices,
        "train_indices_by_label": {key: sorted(set(value)) for key, value in train_indices_by_label.items()},
    }


def select_anchor_indices(
    train_indices_by_label: dict[str, list[int]],
    current_known_classes: list[str],
    excluded_indices: list[int],
    anchor_per_class: int,
    seed: int,
) -> list[int]:
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


def predict_label(
    model: dict,
    x_vector: np.ndarray,
    detector_prediction: int,
    gate_threshold: float,
) -> tuple[str, float]:
    if detector_prediction == 0:
        return "normal", float("-inf")

    known_classes = model["known_classes"]
    clf = model["classifier"]
    scaler = model["scaler"]
    x_batch = x_vector[np.newaxis, :]
    pred_known_id = int(clf.predict(x_batch)[0])
    x_scaled = scaler.transform(x_batch).astype(np.float32)
    decision_scores = np.asarray(
        [
            model_name.decision_function(x_scaled).reshape(-1)[0]
            for model_name in [model["ocsvm_models"][class_name] for class_name in known_classes]
        ],
        dtype=np.float32,
    )
    max_decision = float(decision_scores.max())
    if max_decision < gate_threshold:
        return "unknown", max_decision
    return known_classes[pred_known_id], max_decision
