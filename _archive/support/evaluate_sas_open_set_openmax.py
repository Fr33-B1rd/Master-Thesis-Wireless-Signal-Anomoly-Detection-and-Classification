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
from scipy.stats import weibull_min
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from cluster_patchcore_sas import build_random_projection
from fewshot_patchcore_classifier_sas import build_classifier, build_semantic_feature


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.maximum(exp_logits.sum(axis=1, keepdims=True), 1e-8)


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
        description="Open-set SAS evaluation with lightweight OpenMax on PatchCore features"
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
    parser.add_argument("--classifier-c", type=float, default=1.0)
    parser.add_argument("--class-weight", default="none", choices=["none", "balanced"])
    parser.add_argument("--tail-sizes", default="10,20,40,60")
    parser.add_argument("--alpha-values", default="1,2,3")
    parser.add_argument("--unknown-thresholds", default="0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50")
    parser.add_argument("--coverage-constraint", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_feature_cache(
    dataset_dir: str,
    patchcore_output_dir: str,
    feature_mode: str,
    top_k_patches: int,
    gram_dim: int,
    seed: int,
    cache_path: str,
) -> tuple[np.ndarray, np.ndarray]:
    if os.path.exists(cache_path):
        cache = np.load(cache_path)
        return cache["X"].astype(np.float32), cache["label_names"].astype(str)

    test_label_names = np.load(os.path.join(dataset_dir, "test_labels.npy")).astype(str)
    patch_npz = np.load(os.path.join(patchcore_output_dir, "test_patch_artifacts.npz"))
    sample_indices = patch_npz["sample_indices"].astype(np.int64)
    patch_scores = patch_npz["patch_scores"]
    patch_embeddings = patch_npz["patch_embeddings"]
    embedding_dim = int(patch_npz["embedding_dim"][0])

    if sample_indices.shape[0] != test_label_names.shape[0]:
        raise ValueError("Expected export_selection=all patch artifacts to cover the full test set")

    gram_projection = None
    if feature_mode == "mean_plus_gram":
        gram_projection = build_random_projection(
            input_dim=embedding_dim,
            output_dim=gram_dim,
            seed=seed,
        )

    log(f"Building {feature_mode} features for {sample_indices.shape[0]} test samples")
    features = []
    for i in range(sample_indices.shape[0]):
        features.append(
            build_semantic_feature(
                patch_entry={
                    "patch_scores": patch_scores[i],
                    "patch_embeddings": patch_embeddings[i],
                },
                feature_mode=feature_mode,
                top_k_patches=top_k_patches,
                gram_projection=gram_projection,
            )
        )
    X_all = np.stack(features).astype(np.float32)
    np.savez_compressed(cache_path, X=X_all, label_names=test_label_names.astype("U32"))
    return X_all, test_label_names


def fit_weibull_tail(distances: np.ndarray, tail_size: int) -> tuple[float, float, float]:
    tail_size = min(int(tail_size), distances.shape[0])
    if tail_size <= 1:
        raise ValueError("tail_size must be > 1")
    tail = np.sort(distances)[-tail_size:]
    tail = np.maximum(tail, 1e-8)
    shape, loc, scale = weibull_min.fit(tail, floc=0)
    return float(shape), float(loc), float(scale)


def openmax_recalibrate(
    logits: np.ndarray,
    distances: np.ndarray,
    weibull_params: list[tuple[float, float, float]],
    alpha: int,
) -> np.ndarray:
    alpha = min(int(alpha), logits.shape[1])
    rank_idx = np.argsort(-logits, axis=1)
    modified_logits = logits.copy()
    unknown_logits = np.zeros((logits.shape[0], 1), dtype=np.float32)

    for i in range(logits.shape[0]):
        for rank in range(alpha):
            class_idx = int(rank_idx[i, rank])
            shape, loc, scale = weibull_params[class_idx]
            distance = max(float(distances[i, class_idx]), 1e-8)
            wscore = float(weibull_min.cdf(distance, shape, loc=loc, scale=scale))
            alpha_weight = float((alpha - rank) / alpha)
            omega = alpha_weight * wscore
            removed = modified_logits[i, class_idx] * omega
            modified_logits[i, class_idx] -= removed
            unknown_logits[i, 0] += removed

    openmax_logits = np.concatenate([modified_logits, unknown_logits], axis=1)
    return softmax(openmax_logits)


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
    tail_sizes = [int(part.strip()) for part in args.tail_sizes.split(",") if part.strip()]
    alpha_values = [int(part.strip()) for part in args.alpha_values.split(",") if part.strip()]
    unknown_thresholds = [float(part.strip()) for part in args.unknown_thresholds.split(",") if part.strip()]

    results = []
    confusion_sums: dict[tuple[int, int, float], np.ndarray] = {}
    for tail_size in tail_sizes:
        for alpha in alpha_values:
            for unknown_threshold in unknown_thresholds:
                confusion_sums[(tail_size, alpha, unknown_threshold)] = np.zeros(
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
        class_weight = None if args.class_weight == "none" else "balanced"
        clf = build_classifier("logreg", seed=args.seed + repeat)
        clf.named_steps["clf"].set_params(C=args.classifier_c, class_weight=class_weight)
        clf.fit(X_all[train_idx], y_train)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_all[train_idx])
        X_eval_scaled = scaler.transform(X_all[eval_idx])

        class_means = []
        class_train_distances = []
        for class_name in known_classes:
            class_id = known_to_id[class_name]
            class_vectors = X_train_scaled[y_train == class_id]
            mean_vec = class_vectors.mean(axis=0)
            class_means.append(mean_vec)
            distances = np.linalg.norm(class_vectors - mean_vec[None, :], axis=1)
            class_train_distances.append(distances.astype(np.float32))
        class_means = np.stack(class_means).astype(np.float32)

        eval_logits = clf.decision_function(X_all[eval_idx]).astype(np.float32)
        if eval_logits.ndim == 1:
            eval_logits = eval_logits[:, None]
        eval_distances = np.stack(
            [np.linalg.norm(X_eval_scaled - mean_vec[None, :], axis=1) for mean_vec in class_means],
            axis=1,
        ).astype(np.float32)

        for tail_size in tail_sizes:
            weibull_params = [fit_weibull_tail(dist, tail_size) for dist in class_train_distances]
            for alpha in alpha_values:
                openmax_probs = openmax_recalibrate(
                    logits=eval_logits,
                    distances=eval_distances,
                    weibull_params=weibull_params,
                    alpha=alpha,
                )
                unknown_prob = openmax_probs[:, -1]
                pred_known_ids = np.argmax(openmax_probs[:, :-1], axis=1)

                for unknown_threshold in unknown_thresholds:
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
                        if unknown_prob[local_pos] >= unknown_threshold:
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
                    confusion_sums[(tail_size, alpha, unknown_threshold)] += cm

                    row = {
                        "repeat": repeat,
                        "tail_size": tail_size,
                        "alpha": alpha,
                        "unknown_threshold": unknown_threshold,
                        "macro_f1_known": macro_f1_known,
                        "known_coverage": known_coverage,
                        "normal_reject_rate": normal_reject_rate,
                        "avg_unknown_reject_rate": avg_unknown_reject_rate,
                    }
                    for name in unknown_classes:
                        row[f"{name}_reject_rate"] = unknown_reject_rates[name]
                    results.append(row)

    detail_df = pd.DataFrame(results)
    detail_df.to_csv(os.path.join(args.output_dir, "openmax_detail.csv"), index=False)

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
        detail_df.groupby(["tail_size", "alpha", "unknown_threshold"], as_index=False)
        .agg(**agg_spec)
        .sort_values(
            ["avg_unknown_reject_rate_mean", "macro_f1_known_mean", "known_coverage_mean"],
            ascending=[False, False, False],
        )
    )
    summary_df.to_csv(os.path.join(args.output_dir, "openmax_summary.csv"), index=False)

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
        key = (
            int(constrained["tail_size"]),
            int(constrained["alpha"]),
            float(constrained["unknown_threshold"]),
        )
        avg_cm = confusion_sums[key] / float(args.repeats)
        save_rectangular_confusion(
            matrix=avg_cm,
            row_labels=true_label_names,
            col_labels=pred_label_names,
            output_path=os.path.join(args.output_dir, "best_under_constraint_confusion_matrix.png"),
            title=(
                f"SAS OpenMax (tail={constrained['tail_size']}, alpha={constrained['alpha']}, "
                f"thr={constrained['unknown_threshold']:.2f})"
            ),
        )
    with open(os.path.join(args.output_dir, "best_under_coverage_constraint.json"), "w", encoding="utf-8") as f:
        json.dump(constrained, f, indent=2)

    log(
        f"Finished OpenMax evaluation. "
        f"Best overall avg-unknown-reject={best_overall['avg_unknown_reject_rate_mean']:.4f}, "
        f"best feasible={bool(constrained.get('feasible', False))}"
    )


if __name__ == "__main__":
    main()
