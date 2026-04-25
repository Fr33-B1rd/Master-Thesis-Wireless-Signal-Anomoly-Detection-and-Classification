import argparse
import json
import os
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score

from cluster_patchcore_sas import build_random_projection
from fewshot_patchcore_classifier_sas import build_classifier, build_semantic_feature


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open-set SAS abnormal subtype classification with confidence-based reject"
    )
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--patchcore-output-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--known-classes", default="chirp,pulse,tone")
    parser.add_argument("--unknown-classes", default="comb,fsk,ofdm")
    parser.add_argument("--shot", type=int, default=60)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--feature-mode", default="mean_plus_gram", choices=["mean", "mean_plus_gram"])
    parser.add_argument("--top-k-patches", type=int, default=64)
    parser.add_argument("--gram-dim", type=int, default=32)
    parser.add_argument("--classifier", default="logreg", choices=["logreg", "linear_svm"])
    parser.add_argument("--classifier-c", type=float, default=1.0)
    parser.add_argument("--class-weight", default="none", choices=["none", "balanced"])
    parser.add_argument(
        "--reject-thresholds",
        default="0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    known_classes = [part.strip() for part in args.known_classes.split(",") if part.strip()]
    unknown_classes = [part.strip() for part in args.unknown_classes.split(",") if part.strip()]
    true_label_names = ["normal", *known_classes, *unknown_classes]
    pred_label_names = ["unknown", *known_classes]

    test_label_names = np.load(os.path.join(args.dataset_dir, "test_labels.npy")).astype(str)
    patch_npz = np.load(os.path.join(args.patchcore_output_dir, "test_patch_artifacts.npz"))

    sample_indices = patch_npz["sample_indices"].astype(np.int64)
    patch_scores = patch_npz["patch_scores"]
    patch_embeddings = patch_npz["patch_embeddings"]
    embedding_dim = int(patch_npz["embedding_dim"][0])

    if sample_indices.shape[0] != test_label_names.shape[0]:
        raise ValueError("Expected export_selection=all patch artifacts to cover the full test set")

    gram_projection = None
    if args.feature_mode == "mean_plus_gram":
        gram_projection = build_random_projection(
            input_dim=embedding_dim,
            output_dim=args.gram_dim,
            seed=args.seed,
        )

    log(f"Building {args.feature_mode} features for {sample_indices.shape[0]} test samples")
    features = []
    for i in range(sample_indices.shape[0]):
        features.append(
            build_semantic_feature(
                patch_entry={
                    "patch_scores": patch_scores[i],
                    "patch_embeddings": patch_embeddings[i],
                },
                feature_mode=args.feature_mode,
                top_k_patches=args.top_k_patches,
                gram_projection=gram_projection,
            )
        )
    X_all = np.stack(features).astype(np.float32)

    label_series = pd.Series(test_label_names)
    known_index_map = {
        label_name: np.where(label_series.to_numpy() == label_name)[0]
        for label_name in known_classes
    }
    for label_name, indices in known_index_map.items():
        if indices.shape[0] <= args.shot:
            raise ValueError(f"{label_name} has only {indices.shape[0]} samples, fewer than shot={args.shot}")

    known_to_id = {name: idx for idx, name in enumerate(known_classes)}
    threshold_values = [float(part.strip()) for part in args.reject_thresholds.split(",") if part.strip()]

    results = []
    confusion_sums = {
        threshold: np.zeros((len(true_label_names), len(pred_label_names)), dtype=np.float64)
        for threshold in threshold_values
    }

    for repeat in range(args.repeats):
        rng = np.random.default_rng(args.seed + repeat)
        train_parts = []
        for label_name in known_classes:
            shuffled = rng.permutation(known_index_map[label_name])
            train_parts.append(shuffled[: args.shot])
        train_idx = np.concatenate(train_parts)

        train_mask = np.zeros(test_label_names.shape[0], dtype=bool)
        train_mask[train_idx] = True
        eval_idx = np.where(~train_mask)[0]

        y_train = np.asarray([known_to_id[str(test_label_names[idx])] for idx in train_idx], dtype=np.int64)
        model = build_classifier(args.classifier, seed=args.seed + repeat)
        if args.classifier == "logreg":
            class_weight = None if args.class_weight == "none" else "balanced"
            model.named_steps["clf"].set_params(C=args.classifier_c, class_weight=class_weight)

        model.fit(X_all[train_idx], y_train)
        if not hasattr(model, "predict_proba"):
            raise RuntimeError("Open-set reject evaluation requires a probabilistic classifier")

        proba = model.predict_proba(X_all[eval_idx])
        pred_known_ids = model.classes_[np.argmax(proba, axis=1)]
        pred_conf = np.max(proba, axis=1)

        for threshold in threshold_values:
            true_names = []
            pred_names = []

            known_true = []
            known_pred = []
            known_rejected = 0

            normal_total = 0
            normal_rejected = 0
            unknown_total = {name: 0 for name in unknown_classes}
            unknown_rejected = {name: 0 for name in unknown_classes}

            for local_pos, global_idx in enumerate(eval_idx):
                true_name = str(test_label_names[global_idx])
                if pred_conf[local_pos] < threshold:
                    pred_name = "unknown"
                else:
                    pred_name = known_classes[int(pred_known_ids[local_pos])]

                true_names.append(true_name)
                pred_names.append(pred_name)

                if true_name in known_classes:
                    known_true.append(true_name)
                    known_pred.append(pred_name)
                    if pred_name == "unknown":
                        known_rejected += 1
                elif true_name == "normal":
                    normal_total += 1
                    if pred_name == "unknown":
                        normal_rejected += 1
                elif true_name in unknown_classes:
                    unknown_total[true_name] += 1
                    if pred_name == "unknown":
                        unknown_rejected[true_name] += 1

            known_accept_mask = [pred != "unknown" for pred in known_pred]
            accepted_true = [t for t, keep in zip(known_true, known_accept_mask) if keep]
            accepted_pred = [p for p, keep in zip(known_pred, known_accept_mask) if keep]

            if accepted_true:
                macro_f1_known = float(
                    f1_score(
                        accepted_true,
                        accepted_pred,
                        labels=known_classes,
                        average="macro",
                    )
                )
            else:
                macro_f1_known = 0.0

            known_coverage = 1.0 - (known_rejected / max(len(known_true), 1))
            normal_reject_rate = normal_rejected / max(normal_total, 1)
            unknown_reject_rates = {
                name: unknown_rejected[name] / max(unknown_total[name], 1)
                for name in unknown_classes
            }
            avg_unknown_reject_rate = float(np.mean(list(unknown_reject_rates.values())))

            row_to_idx = {name: idx for idx, name in enumerate(true_label_names)}
            col_to_idx = {name: idx for idx, name in enumerate(pred_label_names)}
            cm = np.zeros((len(true_label_names), len(pred_label_names)), dtype=np.float64)
            for true_name, pred_name in zip(true_names, pred_names):
                cm[row_to_idx[true_name], col_to_idx[pred_name]] += 1
            confusion_sums[threshold] += cm

            row = {
                "repeat": repeat,
                "reject_threshold": threshold,
                "macro_f1_known": macro_f1_known,
                "known_coverage": known_coverage,
                "normal_reject_rate": normal_reject_rate,
                "avg_unknown_reject_rate": avg_unknown_reject_rate,
            }
            for name in unknown_classes:
                row[f"{name}_reject_rate"] = unknown_reject_rates[name]
            results.append(row)

    detail_df = pd.DataFrame(results)
    detail_path = os.path.join(args.output_dir, "reject_threshold_detail.csv")
    detail_df.to_csv(detail_path, index=False)

    agg_spec = {
        "macro_f1_known_mean": ("macro_f1_known", "mean"),
        "macro_f1_known_std": ("macro_f1_known", "std"),
        "known_coverage_mean": ("known_coverage", "mean"),
        "known_coverage_std": ("known_coverage", "std"),
        "normal_reject_rate_mean": ("normal_reject_rate", "mean"),
        "normal_reject_rate_std": ("normal_reject_rate", "std"),
        "avg_unknown_reject_rate_mean": ("avg_unknown_reject_rate", "mean"),
        "avg_unknown_reject_rate_std": ("avg_unknown_reject_rate", "std"),
    }
    for name in unknown_classes:
        agg_spec[f"{name}_reject_rate_mean"] = (f"{name}_reject_rate", "mean")
        agg_spec[f"{name}_reject_rate_std"] = (f"{name}_reject_rate", "std")

    summary_df = (
        detail_df.groupby("reject_threshold", as_index=False)
        .agg(**agg_spec)
        .sort_values(
            ["macro_f1_known_mean", "avg_unknown_reject_rate_mean"],
            ascending=[False, False],
        )
    )

    summary_path = os.path.join(args.output_dir, "reject_threshold_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    best_row = summary_df.iloc[0]
    best_threshold = float(best_row["reject_threshold"])
    best_json_path = os.path.join(args.output_dir, "best_threshold.json")
    with open(best_json_path, "w", encoding="utf-8") as f:
        json.dump(best_row.to_dict(), f, indent=2)

    avg_cm = confusion_sums[best_threshold] / float(args.repeats)
    save_rectangular_confusion(
        matrix=avg_cm,
        row_labels=true_label_names,
        col_labels=pred_label_names,
        output_path=os.path.join(args.output_dir, "best_threshold_confusion_matrix.png"),
        title=f"SAS Open-Set Reject (shot={args.shot}, thr={best_threshold:.2f})",
    )

    log(
        f"Finished open-set SAS reject eval. "
        f"Best threshold={best_threshold:.2f}, "
        f"macro-F1-known={best_row['macro_f1_known_mean']:.4f}, "
        f"avg-unknown-reject={best_row['avg_unknown_reject_rate_mean']:.4f}"
    )


if __name__ == "__main__":
    main()
