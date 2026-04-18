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
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from evaluate_sas_open_set_ocsvm_gate import build_feature_cache
from fewshot_patchcore_classifier_sas import build_classifier


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open-set SAS evaluation with logreg + Mahalanobis reject"
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
    parser.add_argument("--distance-percentiles", default="90,95,97,98,99,99.5")
    parser.add_argument("--var-eps", type=float, default=1e-3)
    parser.add_argument("--coverage-constraint", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    known_classes = [part.strip() for part in args.known_classes.split(",") if part.strip()]
    unknown_classes = [part.strip() for part in args.unknown_classes.split(",") if part.strip()]
    true_label_names = ["normal", *known_classes, *unknown_classes]
    pred_label_names = ["unknown", *known_classes]
    cache_path = os.path.join(args.output_dir, f"feature_cache_{args.feature_mode}.npz")

    X_all, label_names = build_feature_cache(
        dataset_dir=args.dataset_dir,
        patchcore_output_dir=args.patchcore_output_dir,
        feature_mode=args.feature_mode,
        top_k_patches=args.top_k_patches,
        gram_dim=args.gram_dim,
        seed=args.seed,
        cache_path=cache_path,
        augment_temporal_diff=False,
        augment_dynamic_range=False,
    )

    label_array = np.asarray(label_names).astype(str)
    known_index_map = {
        label_name: np.where(label_array == label_name)[0]
        for label_name in known_classes
    }
    for label_name, indices in known_index_map.items():
        if indices.shape[0] <= args.shot:
            raise ValueError(f"{label_name} has only {indices.shape[0]} samples, fewer than shot={args.shot}")

    known_to_id = {name: idx for idx, name in enumerate(known_classes)}
    percentile_values = [float(part.strip()) for part in args.distance_percentiles.split(",") if part.strip()]

    results = []
    confusion_sums = {
        percentile: np.zeros((len(true_label_names), len(pred_label_names)), dtype=np.float64)
        for percentile in percentile_values
    }

    for repeat in range(args.repeats):
        rng = np.random.default_rng(args.seed + repeat)
        train_parts = []
        for label_name in known_classes:
            shuffled = rng.permutation(known_index_map[label_name])
            train_parts.append(shuffled[: args.shot])
        train_idx = np.concatenate(train_parts)

        train_mask = np.zeros(label_array.shape[0], dtype=bool)
        train_mask[train_idx] = True
        eval_idx = np.where(~train_mask)[0]

        y_train = np.asarray([known_to_id[str(label_array[idx])] for idx in train_idx], dtype=np.int64)
        clf = build_classifier(args.classifier, seed=args.seed + repeat)
        if args.classifier == "logreg":
            class_weight = None if args.class_weight == "none" else "balanced"
            clf.named_steps["clf"].set_params(C=args.classifier_c, class_weight=class_weight)
        clf.fit(X_all[train_idx], y_train)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_all[train_idx]).astype(np.float32)
        X_eval_scaled = scaler.transform(X_all[eval_idx]).astype(np.float32)
        eval_pred_known_ids = clf.predict(X_all[eval_idx]).astype(np.int64)

        class_stats = {}
        for label_name in known_classes:
            class_id = known_to_id[label_name]
            class_vectors = X_train_scaled[y_train == class_id]
            mean_vec = class_vectors.mean(axis=0).astype(np.float32)
            var_vec = class_vectors.var(axis=0).astype(np.float32)
            var_vec = np.maximum(var_vec, np.float32(args.var_eps))
            train_dist = np.sqrt(np.sum(((class_vectors - mean_vec[None, :]) ** 2) / var_vec[None, :], axis=1))
            class_stats[label_name] = {
                "mean": mean_vec,
                "var": var_vec,
                "train_dist": train_dist.astype(np.float32),
            }

        predicted_distances = np.zeros(eval_idx.shape[0], dtype=np.float32)
        for local_pos, class_id in enumerate(eval_pred_known_ids.tolist()):
            class_name = known_classes[int(class_id)]
            stats = class_stats[class_name]
            delta = X_eval_scaled[local_pos] - stats["mean"]
            predicted_distances[local_pos] = float(
                np.sqrt(np.sum((delta * delta) / stats["var"]))
            )

        percentile_thresholds = {}
        for percentile in percentile_values:
            percentile_thresholds[percentile] = {
                class_name: float(np.percentile(class_stats[class_name]["train_dist"], percentile))
                for class_name in known_classes
            }

        for percentile in percentile_values:
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
                true_name = str(label_array[global_idx])
                pred_class_name = known_classes[int(eval_pred_known_ids[local_pos])]
                threshold = percentile_thresholds[percentile][pred_class_name]
                if predicted_distances[local_pos] > threshold:
                    pred_name = "unknown"
                else:
                    pred_name = pred_class_name

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
            confusion_sums[percentile] += cm

            row = {
                "repeat": repeat,
                "distance_percentile": percentile,
                "macro_f1_known": macro_f1_known,
                "known_coverage": known_coverage,
                "normal_reject_rate": normal_reject_rate,
                "avg_unknown_reject_rate": avg_unknown_reject_rate,
            }
            for name in unknown_classes:
                row[f"{name}_reject_rate"] = unknown_reject_rates[name]
            results.append(row)

    detail_df = pd.DataFrame(results)
    detail_df.to_csv(os.path.join(args.output_dir, "mahalanobis_detail.csv"), index=False)

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
        detail_df.groupby("distance_percentile", as_index=False)
        .agg(**agg_spec)
        .sort_values(
            ["avg_unknown_reject_rate_mean", "macro_f1_known_mean", "known_coverage_mean"],
            ascending=[False, False, False],
        )
    )
    summary_df.to_csv(os.path.join(args.output_dir, "mahalanobis_summary.csv"), index=False)

    best_overall = summary_df.iloc[0].to_dict()
    with open(os.path.join(args.output_dir, "best_overall.json"), "w", encoding="utf-8") as f:
        json.dump(best_overall, f, indent=2)

    feasible_df = summary_df[summary_df["known_coverage_mean"] >= args.coverage_constraint]
    if feasible_df.empty:
        constrained = {
            "feasible": False,
            "coverage_constraint": args.coverage_constraint,
        }
    else:
        constrained = feasible_df.iloc[0].to_dict()
        constrained["feasible"] = True
        constrained["coverage_constraint"] = args.coverage_constraint
        key = float(constrained["distance_percentile"])
        avg_cm = confusion_sums[key] / float(args.repeats)
        save_rectangular_confusion(
            matrix=avg_cm,
            row_labels=true_label_names,
            col_labels=pred_label_names,
            output_path=os.path.join(args.output_dir, "best_under_constraint_confusion_matrix.png"),
            title=f"SAS Mahalanobis Reject (pct={key:.1f})",
        )
    with open(os.path.join(args.output_dir, "best_under_coverage_constraint.json"), "w", encoding="utf-8") as f:
        json.dump(constrained, f, indent=2)

    log(
        f"Finished Mahalanobis evaluation. "
        f"Best overall avg-unknown-reject={best_overall['avg_unknown_reject_rate_mean']:.4f}, "
        f"best feasible={bool(constrained.get('feasible', False))}"
    )


if __name__ == "__main__":
    main()
