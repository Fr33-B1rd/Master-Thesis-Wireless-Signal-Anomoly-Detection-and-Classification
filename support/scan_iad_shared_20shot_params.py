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

from cluster_patchcore_sas import build_random_projection
from evaluate_iad_stratified_sampling import evaluate_strategy
from fewshot_patchcore_classifier_iad import build_semantic_feature, parse_dataset_specs


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
        description="Scan shared-space IAD 20-shot parameters with stratified sampling"
    )
    parser.add_argument(
        "--dataset-specs",
        nargs="+",
        required=True,
        help="Repeated specs of the form CLASS_NAME::OUTPUT_DIR",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shot", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--classifier", default="logreg")
    parser.add_argument(
        "--topk-list",
        default="12,16,20,24",
        help="Comma-separated top-k patch counts",
    )
    parser.add_argument(
        "--gram-dims",
        default="16,32,64",
        help="Comma-separated gram projection dims",
    )
    parser.add_argument(
        "--cluster-counts",
        default="2,3,4,5",
        help="Comma-separated class-internal cluster counts",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_samples(dataset_specs: list[tuple[str, str]]) -> tuple[list[str], list[dict], int]:
    class_names = [name for name, _ in dataset_specs]
    class_to_id = {name: i for i, name in enumerate(class_names)}
    samples = []
    embedding_dim = None

    for class_name, output_dir in dataset_specs:
        patch_path = os.path.join(output_dir, "test_patch_artifacts.npz")
        patch_npz = np.load(patch_path)
        if embedding_dim is None:
            embedding_dim = int(patch_npz["embedding_dim"][0])

        sample_indices = patch_npz["sample_indices"].astype(np.int64)
        patch_scores = patch_npz["patch_scores"]
        patch_embeddings = patch_npz["patch_embeddings"]
        image_scores = patch_npz["image_scores"].astype(np.float32)

        for i in range(sample_indices.shape[0]):
            samples.append(
                {
                    "class_name": class_name,
                    "label_id": class_to_id[class_name],
                    "sample_index": int(sample_indices[i]),
                    "score": float(image_scores[i]),
                    "patch_scores": patch_scores[i],
                    "patch_embeddings": patch_embeddings[i],
                }
            )

    return class_names, samples, int(embedding_dim)


def build_feature_matrix(
    samples: list[dict],
    top_k: int,
    gram_dim: int,
    embedding_dim: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    gram_projection = build_random_projection(
        input_dim=embedding_dim,
        output_dim=gram_dim,
        seed=seed,
    )
    X_parts = []
    y_parts = []
    for sample in samples:
        X_parts.append(
            build_semantic_feature(
                patch_embeddings=sample["patch_embeddings"],
                patch_scores=sample["patch_scores"],
                feature_mode="mean_plus_gram",
                top_k_patches=top_k,
                gram_projection=gram_projection,
            )
        )
        y_parts.append(int(sample["label_id"]))
    return np.stack(X_parts).astype(np.float32), np.asarray(y_parts, dtype=np.int64)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    dataset_specs = parse_dataset_specs(args.dataset_specs)
    class_names, samples, embedding_dim = load_samples(dataset_specs)

    topk_list = [int(part.strip()) for part in args.topk_list.split(",") if part.strip()]
    gram_dims = [int(part.strip()) for part in args.gram_dims.split(",") if part.strip()]
    cluster_counts = [int(part.strip()) for part in args.cluster_counts.split(",") if part.strip()]

    feature_cache_dir = os.path.join(args.output_dir, "feature_cache")
    os.makedirs(feature_cache_dir, exist_ok=True)

    results = []
    best = None

    for top_k in topk_list:
        for gram_dim in gram_dims:
            cache_path = os.path.join(feature_cache_dir, f"topk{top_k}_gram{gram_dim}.npz")
            if os.path.exists(cache_path):
                cache = np.load(cache_path)
                X = cache["X"].astype(np.float32)
                y = cache["y"].astype(np.int64)
                log(f"Loaded cached features for topk={top_k}, gram_dim={gram_dim}")
            else:
                log(f"Building features for topk={top_k}, gram_dim={gram_dim}")
                X, y = build_feature_matrix(
                    samples=samples,
                    top_k=top_k,
                    gram_dim=gram_dim,
                    embedding_dim=embedding_dim,
                    seed=args.seed + gram_dim + top_k,
                )
                np.savez_compressed(cache_path, X=X, y=y)

            for cluster_count in cluster_counts:
                log(
                    f"Evaluating topk={top_k}, gram_dim={gram_dim}, cluster_k={cluster_count}"
                )
                summary, avg_cm = evaluate_strategy(
                    X=X,
                    y=y,
                    class_names=class_names,
                    shot=args.shot,
                    repeats=args.repeats,
                    classifier_name=args.classifier,
                    strategy="cluster_stratified",
                    cluster_count=cluster_count,
                    cluster_pca_dim=32,
                    seed=args.seed,
                )
                row = {
                    "top_k": int(top_k),
                    "gram_dim": int(gram_dim),
                    "cluster_count": int(cluster_count),
                    "classifier": args.classifier,
                    "shot": int(args.shot),
                    "repeats": int(args.repeats),
                    **summary,
                }
                results.append(row)
                if best is None or row["macro_f1_mean"] > best["row"]["macro_f1_mean"]:
                    best = {"row": row, "confusion_matrix": avg_cm}

    result_df = pd.DataFrame(results).sort_values(
        ["macro_f1_mean", "accuracy_mean"],
        ascending=[False, False],
    )
    result_df.to_csv(os.path.join(args.output_dir, "scan_results.csv"), index=False)
    result_df.head(20).to_csv(os.path.join(args.output_dir, "top_configs.csv"), index=False)

    if best is not None:
        with open(os.path.join(args.output_dir, "best_config.json"), "w", encoding="utf-8") as f:
            json.dump(best["row"], f, indent=2)
        save_confusion_heatmap(
            best["confusion_matrix"],
            class_names=class_names,
            output_path=os.path.join(args.output_dir, "best_confusion_matrix.png"),
            title=(
                f"Best Shared IAD {args.shot}-shot\n"
                f"logreg | topk={best['row']['top_k']} | gram={best['row']['gram_dim']} | "
                f"k={best['row']['cluster_count']}"
            ),
        )

    log(f"Parameter scan saved to {args.output_dir}")


if __name__ == "__main__":
    main()
