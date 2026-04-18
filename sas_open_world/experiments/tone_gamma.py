"""Entry point for SAS tone-specific OCSVM gamma sweeps."""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from sas_open_world.common import compute_scale_gamma, log, save_rectangular_confusion
from sas_open_world.features import build_feature_cache
from support.fewshot_patchcore_classifier_sas import build_classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune per-class gamma for the tone OCSVM gate on SAS open-set classification"
    )
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--patchcore-output-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--known-classes", default="chirp,pulse,tone")
    parser.add_argument("--unknown-classes", default="comb,fsk,ofdm")
    parser.add_argument("--target-class", default="tone")
    parser.add_argument("--shot", type=int, default=60)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--feature-mode", default="mean_plus_gram", choices=["mean", "gram", "mean_plus_gram"])
    parser.add_argument("--top-k-patches", type=int, default=64)
    parser.add_argument("--gram-dim", type=int, default=32)
    parser.add_argument("--augment-entropy", action="store_true")
    parser.add_argument("--augment-frequency-profile", action="store_true")
    parser.add_argument("--classifier", default="logreg", choices=["logreg", "linear_svm"])
    parser.add_argument("--classifier-c", type=float, default=1.0)
    parser.add_argument("--class-weight", default="none", choices=["none", "balanced"])
    parser.add_argument("--nu", type=float, default=0.06)
    parser.add_argument("--gamma-multipliers", default="0.5,1,2,4,8,12,16")
    parser.add_argument("--gate-thresholds", default="-0.50,-0.45,-0.40,-0.35,-0.30,-0.25,-0.20,-0.15")
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
    if args.target_class not in known_classes:
        raise ValueError(f"target_class must be one of {known_classes}")

    true_label_names = ["normal", *known_classes, *unknown_classes]
    pred_label_names = ["unknown", *known_classes]
    cache_suffix_parts = [args.feature_mode]
    if args.augment_entropy:
        cache_suffix_parts.append("entropy")
    if args.augment_frequency_profile:
        cache_suffix_parts.append("fprofile")
    cache_path = os.path.join(args.output_dir, f"feature_cache_{'_'.join(cache_suffix_parts)}.npz")

    x_all, label_names = build_feature_cache(
        dataset_dir=args.dataset_dir,
        patchcore_output_dir=args.patchcore_output_dir,
        feature_mode=args.feature_mode,
        top_k_patches=args.top_k_patches,
        gram_dim=args.gram_dim,
        seed=args.seed,
        cache_path=cache_path,
        augment_temporal_diff=False,
        augment_dynamic_range=False,
        augment_entropy=args.augment_entropy,
        augment_frequency_profile=args.augment_frequency_profile,
    )

    label_array = np.asarray(label_names).astype(str)
    known_index_map = {label_name: np.where(label_array == label_name)[0] for label_name in known_classes}
    for label_name, indices in known_index_map.items():
        if indices.shape[0] <= args.shot:
            raise ValueError(f"{label_name} has only {indices.shape[0]} samples, fewer than shot={args.shot}")

    known_to_id = {name: idx for idx, name in enumerate(known_classes)}
    gamma_multipliers = [float(part.strip()) for part in args.gamma_multipliers.split(",") if part.strip()]
    gate_thresholds = [float(part.strip()) for part in args.gate_thresholds.split(",") if part.strip()]

    results = []
    confusion_sums: dict[tuple[float, float], np.ndarray] = {}
    for gamma_multiplier in gamma_multipliers:
        for gate_threshold in gate_thresholds:
            confusion_sums[(gamma_multiplier, gate_threshold)] = np.zeros(
                (len(true_label_names), len(pred_label_names)),
                dtype=np.float64,
            )

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
        clf: Pipeline = build_classifier(args.classifier, seed=args.seed + repeat)
        if args.classifier == "logreg":
            class_weight = None if args.class_weight == "none" else "balanced"
            clf.named_steps["clf"].set_params(C=args.classifier_c, class_weight=class_weight)
        clf.fit(x_all[train_idx], y_train)

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_all[train_idx]).astype(np.float32)
        x_eval_scaled = scaler.transform(x_all[eval_idx]).astype(np.float32)

        class_train_scaled = {}
        class_scale_gamma = {}
        for label_name in known_classes:
            class_id = known_to_id[label_name]
            class_vectors = x_train_scaled[y_train == class_id]
            class_train_scaled[label_name] = class_vectors
            class_scale_gamma[label_name] = compute_scale_gamma(class_vectors)

        eval_pred_known_ids = clf.predict(x_all[eval_idx])

        for gamma_multiplier in gamma_multipliers:
            ocsvm_models = {}
            for label_name in known_classes:
                gamma_value = class_scale_gamma[label_name]
                if label_name == args.target_class:
                    gamma_value *= gamma_multiplier
                ocsvm_models[label_name] = OneClassSVM(kernel="rbf", gamma=gamma_value, nu=args.nu)
                ocsvm_models[label_name].fit(class_train_scaled[label_name])

            decision_matrix = np.stack(
                [ocsvm_models[label_name].decision_function(x_eval_scaled).reshape(-1) for label_name in known_classes],
                axis=1,
            ).astype(np.float32)
            max_decision = decision_matrix.max(axis=1)

            for gate_threshold in gate_thresholds:
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
                    pred_name = "unknown" if max_decision[local_pos] < gate_threshold else known_classes[int(eval_pred_known_ids[local_pos])]
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
                macro_f1_known = float(
                    f1_score(accepted_true, accepted_pred, labels=known_classes, average="macro")
                ) if accepted_true else 0.0

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
                confusion_sums[(gamma_multiplier, gate_threshold)] += cm

                row = {
                    "repeat": repeat,
                    "gamma_multiplier": gamma_multiplier,
                    "gate_threshold": gate_threshold,
                    "macro_f1_known": macro_f1_known,
                    "known_coverage": known_coverage,
                    "normal_reject_rate": normal_reject_rate,
                    "avg_unknown_reject_rate": avg_unknown_reject_rate,
                }
                for name in unknown_classes:
                    row[f"{name}_reject_rate"] = unknown_reject_rates[name]
                results.append(row)

    detail_df = pd.DataFrame(results)
    detail_df.to_csv(os.path.join(args.output_dir, "tone_gamma_detail.csv"), index=False)

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
        detail_df.groupby(["gamma_multiplier", "gate_threshold"], as_index=False)
        .agg(**agg_spec)
        .sort_values(
            ["avg_unknown_reject_rate_mean", "macro_f1_known_mean", "known_coverage_mean"],
            ascending=[False, False, False],
        )
    )
    summary_df.to_csv(os.path.join(args.output_dir, "tone_gamma_summary.csv"), index=False)

    best_overall = summary_df.iloc[0].to_dict()
    with open(os.path.join(args.output_dir, "best_overall.json"), "w", encoding="utf-8") as f:
        json.dump(best_overall, f, indent=2)

    feasible_df = summary_df[summary_df["known_coverage_mean"] >= args.coverage_constraint]
    if feasible_df.empty:
        constrained = {"feasible": False, "coverage_constraint": args.coverage_constraint}
    else:
        constrained = feasible_df.iloc[0].to_dict()
        constrained["feasible"] = True
        constrained["coverage_constraint"] = args.coverage_constraint
        key = (float(constrained["gamma_multiplier"]), float(constrained["gate_threshold"]))
        avg_cm = confusion_sums[key] / float(args.repeats)
        save_rectangular_confusion(
            matrix=avg_cm,
            row_labels=true_label_names,
            col_labels=pred_label_names,
            output_path=os.path.join(args.output_dir, "best_under_constraint_confusion_matrix.png"),
            title=(
                f"SAS Tone-Gamma OCSVM (x{constrained['gamma_multiplier']:.2f}, "
                f"thr={constrained['gate_threshold']:.2f})"
            ),
        )
    with open(os.path.join(args.output_dir, "best_under_coverage_constraint.json"), "w", encoding="utf-8") as f:
        json.dump(constrained, f, indent=2)

    log(
        f"Finished tone-gamma evaluation. "
        f"Best overall avg-unknown-reject={best_overall['avg_unknown_reject_rate_mean']:.4f}, "
        f"best feasible={bool(constrained.get('feasible', False))}"
    )


if __name__ == "__main__":
    main()
