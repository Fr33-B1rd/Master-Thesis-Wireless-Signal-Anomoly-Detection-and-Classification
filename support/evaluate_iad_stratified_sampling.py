import argparse
import json
import math
import os
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

from fewshot_patchcore_classifier_iad import build_classifier


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def save_confusion_heatmap(
    matrix: np.ndarray,
    class_names: list[str],
    output_path: str,
    title: str,
) -> None:
    plt.figure(figsize=(6.5, 5.5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare random vs cluster-stratified few-shot sampling on IAD cached features"
    )
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--class-names",
        default="16QAM,CHIRP,GMSK,QPSK",
        help="Comma-separated class names in label-id order",
    )
    parser.add_argument("--shot", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--classifier", default="logreg")
    parser.add_argument(
        "--cluster-counts",
        default="2,3,4,5",
        help="Comma-separated KMeans cluster counts to test for stratified sampling",
    )
    parser.add_argument("--cluster-pca-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def allocate_quota(cluster_sizes: np.ndarray, shot: int) -> np.ndarray:
    num_clusters = int(cluster_sizes.shape[0])
    if shot < num_clusters:
        raise ValueError("shot must be >= num_clusters for stratified allocation")

    quotas = np.ones(num_clusters, dtype=int)
    remaining = int(shot - num_clusters)
    if remaining <= 0:
        return quotas

    free_sizes = np.maximum(cluster_sizes - 1, 0)
    total_free = int(free_sizes.sum())
    if total_free == 0:
        quotas[:shot] = 1
        return quotas

    proportional = free_sizes / float(total_free) * remaining
    add = np.floor(proportional).astype(int)
    quotas += add
    leftover = remaining - int(add.sum())
    if leftover > 0:
        order = np.argsort(-(proportional - add))
        for idx in order[:leftover]:
            quotas[idx] += 1

    quotas = np.minimum(quotas, cluster_sizes)
    deficit = shot - int(quotas.sum())
    if deficit > 0:
        spare = cluster_sizes - quotas
        order = np.argsort(-spare)
        for idx in order:
            if deficit <= 0:
                break
            take = min(deficit, int(spare[idx]))
            quotas[idx] += take
            deficit -= take
    return quotas


def select_random(indices: np.ndarray, shot: int, rng: np.random.Generator) -> np.ndarray:
    shuffled = rng.permutation(indices)
    return shuffled[:shot]


def select_cluster_stratified(
    X_class: np.ndarray,
    global_indices: np.ndarray,
    shot: int,
    cluster_count: int,
    pca_dim: int,
    seed: int,
    repeat: int,
) -> np.ndarray:
    cluster_count = min(int(cluster_count), int(shot), int(global_indices.shape[0]))
    if cluster_count <= 1:
        rng = np.random.default_rng(seed + repeat)
        return select_random(global_indices, shot, rng)

    scaled = StandardScaler().fit_transform(X_class)
    reduced_dim = min(int(pca_dim), scaled.shape[1], scaled.shape[0])
    reduced = PCA(n_components=reduced_dim, random_state=seed).fit_transform(scaled)
    kmeans = KMeans(n_clusters=cluster_count, random_state=seed + repeat, n_init=20)
    cluster_labels = kmeans.fit_predict(reduced)

    cluster_sizes = np.bincount(cluster_labels, minlength=cluster_count)
    quotas = allocate_quota(cluster_sizes, shot)
    rng = np.random.default_rng(seed + repeat)

    selected = []
    for cluster_id in range(cluster_count):
        member_pos = np.where(cluster_labels == cluster_id)[0]
        member_global = global_indices[member_pos]
        chosen = select_random(member_global, int(quotas[cluster_id]), rng)
        selected.append(chosen)
    return np.concatenate(selected)


def evaluate_strategy(
    X: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    shot: int,
    repeats: int,
    classifier_name: str,
    strategy: str,
    cluster_count: int | None,
    cluster_pca_dim: int,
    seed: int,
) -> tuple[dict, np.ndarray]:
    class_index_map = {class_id: np.where(y == class_id)[0] for class_id in range(len(class_names))}
    confusion_sum = np.zeros((len(class_names), len(class_names)), dtype=np.float64)
    rows = []

    for repeat in range(repeats):
        train_parts = []
        test_parts = []
        for class_id, indices in class_index_map.items():
            if strategy == "random":
                rng = np.random.default_rng(seed + 1000 * class_id + repeat)
                train_idx = select_random(indices, shot, rng)
            else:
                train_idx = select_cluster_stratified(
                    X_class=X[indices],
                    global_indices=indices,
                    shot=shot,
                    cluster_count=int(cluster_count),
                    pca_dim=cluster_pca_dim,
                    seed=seed + 1000 * class_id,
                    repeat=repeat,
                )
            train_mask = np.zeros(indices.shape[0], dtype=bool)
            lookup = {value: pos for pos, value in enumerate(indices.tolist())}
            for value in train_idx.tolist():
                train_mask[lookup[value]] = True
            test_idx = indices[~train_mask]
            train_parts.append(train_idx)
            test_parts.append(test_idx)

        train_idx = np.concatenate(train_parts)
        test_idx = np.concatenate(test_parts)
        model = build_classifier(classifier_name, seed + repeat)
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[test_idx])

        acc = float(accuracy_score(y[test_idx], y_pred))
        macro_f1 = float(f1_score(y[test_idx], y_pred, average="macro"))
        rows.append({"accuracy": acc, "macro_f1": macro_f1})
        confusion_sum += confusion_matrix(
            y[test_idx],
            y_pred,
            labels=np.arange(len(class_names)),
        )

    df = pd.DataFrame(rows)
    summary = {
        "strategy": strategy,
        "cluster_count": None if cluster_count is None else int(cluster_count),
        "accuracy_mean": float(df["accuracy"].mean()),
        "accuracy_std": float(df["accuracy"].std(ddof=0)),
        "macro_f1_mean": float(df["macro_f1"].mean()),
        "macro_f1_std": float(df["macro_f1"].std(ddof=0)),
    }
    return summary, confusion_sum / float(repeats)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    cache = np.load(args.feature_cache)
    X = cache["X"].astype(np.float32)
    y = cache["y"].astype(np.int64)
    class_names = [part.strip() for part in args.class_names.split(",") if part.strip()]
    cluster_counts = [int(part.strip()) for part in args.cluster_counts.split(",") if part.strip()]

    results = []
    best = None

    random_summary, random_cm = evaluate_strategy(
        X=X,
        y=y,
        class_names=class_names,
        shot=args.shot,
        repeats=args.repeats,
        classifier_name=args.classifier,
        strategy="random",
        cluster_count=None,
        cluster_pca_dim=args.cluster_pca_dim,
        seed=args.seed,
    )
    results.append(random_summary)
    best = {"summary": random_summary, "confusion_matrix": random_cm}

    for cluster_count in cluster_counts:
        log(f"Evaluating cluster-stratified sampling with k={cluster_count}")
        summary, avg_cm = evaluate_strategy(
            X=X,
            y=y,
            class_names=class_names,
            shot=args.shot,
            repeats=args.repeats,
            classifier_name=args.classifier,
            strategy="cluster_stratified",
            cluster_count=cluster_count,
            cluster_pca_dim=args.cluster_pca_dim,
            seed=args.seed,
        )
        results.append(summary)
        if summary["macro_f1_mean"] > best["summary"]["macro_f1_mean"]:
            best = {"summary": summary, "confusion_matrix": avg_cm}

    result_df = pd.DataFrame(results).sort_values(
        ["macro_f1_mean", "accuracy_mean"],
        ascending=[False, False],
    )
    result_df.to_csv(os.path.join(args.output_dir, "sampling_comparison.csv"), index=False)

    with open(os.path.join(args.output_dir, "best_sampling.json"), "w", encoding="utf-8") as f:
        json.dump(best["summary"], f, indent=2)

    save_confusion_heatmap(
        best["confusion_matrix"],
        class_names=class_names,
        output_path=os.path.join(args.output_dir, "best_sampling_confusion_matrix.png"),
        title=(
            f"Best IAD {args.shot}-shot Sampling\n"
            f"{best['summary']['strategy']} "
            + (
                f"(k={best['summary']['cluster_count']})"
                if best["summary"]["cluster_count"] is not None
                else ""
            )
        ),
    )

    log(f"Sampling comparison saved to {args.output_dir}")


if __name__ == "__main__":
    main()
