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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from cluster_patchcore_sas import build_random_projection
from fewshot_patchcore_classifier_iad import (
    build_classifier,
    build_semantic_feature,
    parse_dataset_specs,
)


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
        description="Scan IAD few-shot classifier configs on top of PatchCore features"
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
    parser.add_argument(
        "--feature-modes",
        default="mean,mean_plus_gram",
        help="Comma-separated feature modes",
    )
    parser.add_argument(
        "--topk-list",
        default="4,8,16,32",
        help="Comma-separated top-k patch counts",
    )
    parser.add_argument(
        "--classifiers",
        default="linear_svm,logreg",
        help="Comma-separated classifier names",
    )
    parser.add_argument("--gram-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_iad_patch_sets(dataset_specs: list[tuple[str, str]]) -> tuple[list[str], list[dict]]:
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

    return class_names, samples


def build_feature_matrix(
    samples: list[dict],
    feature_mode: str,
    top_k: int,
    gram_projection: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []
    for sample in samples:
        features.append(
            build_semantic_feature(
                patch_embeddings=sample["patch_embeddings"],
                patch_scores=sample["patch_scores"],
                feature_mode=feature_mode,
                top_k_patches=top_k,
                gram_projection=gram_projection,
            )
        )
        labels.append(int(sample["label_id"]))
    return np.stack(features).astype(np.float32), np.asarray(labels, dtype=np.int64)


def evaluate_config(
    X: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    classifier_name: str,
    shot: int,
    repeats: int,
    seed: int,
) -> tuple[dict, np.ndarray]:
    class_index_map = {class_id: np.where(y == class_id)[0] for class_id in range(len(class_names))}
    confusion_sum = np.zeros((len(class_names), len(class_names)), dtype=np.float64)
    metrics = []

    for repeat in range(repeats):
        rng = np.random.default_rng(seed + repeat)
        train_parts = []
        test_parts = []
        for class_id, indices in class_index_map.items():
            shuffled = rng.permutation(indices)
            train_parts.append(shuffled[:shot])
            test_parts.append(shuffled[shot:])

        train_idx = np.concatenate(train_parts)
        test_idx = np.concatenate(test_parts)
        model = build_classifier(classifier_name, seed + repeat)
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[test_idx])

        metrics.append(
            {
                "accuracy": float(accuracy_score(y[test_idx], y_pred)),
                "macro_f1": float(f1_score(y[test_idx], y_pred, average="macro")),
            }
        )
        confusion_sum += confusion_matrix(
            y[test_idx],
            y_pred,
            labels=np.arange(len(class_names)),
        )

    metric_df = pd.DataFrame(metrics)
    summary = {
        "accuracy_mean": float(metric_df["accuracy"].mean()),
        "accuracy_std": float(metric_df["accuracy"].std(ddof=0)),
        "macro_f1_mean": float(metric_df["macro_f1"].mean()),
        "macro_f1_std": float(metric_df["macro_f1"].std(ddof=0)),
    }
    return summary, confusion_sum / float(repeats)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    dataset_specs = parse_dataset_specs(args.dataset_specs)
    class_names, samples = load_iad_patch_sets(dataset_specs)
    feature_modes = [part.strip() for part in args.feature_modes.split(",") if part.strip()]
    topk_list = [int(part.strip()) for part in args.topk_list.split(",") if part.strip()]
    classifiers = [part.strip() for part in args.classifiers.split(",") if part.strip()]

    embedding_dim = int(samples[0]["patch_embeddings"].shape[-1])
    gram_projection = build_random_projection(
        input_dim=embedding_dim,
        output_dim=args.gram_dim,
        seed=args.seed,
    )

    results = []
    best = None
    cache_dir = os.path.join(args.output_dir, "feature_cache")
    os.makedirs(cache_dir, exist_ok=True)

    for feature_mode in feature_modes:
        for top_k in topk_list:
            cache_path = os.path.join(cache_dir, f"{feature_mode}_topk{top_k}.npz")
            if os.path.exists(cache_path):
                cache = np.load(cache_path)
                X = cache["X"].astype(np.float32)
                y = cache["y"].astype(np.int64)
                log(f"Loaded cached features: mode={feature_mode}, topk={top_k}")
            else:
                log(f"Building features: mode={feature_mode}, topk={top_k}")
                X, y = build_feature_matrix(
                    samples=samples,
                    feature_mode=feature_mode,
                    top_k=top_k,
                    gram_projection=gram_projection if feature_mode == "mean_plus_gram" else None,
                )
                np.savez_compressed(cache_path, X=X, y=y)

            for classifier_name in classifiers:
                log(
                    f"Evaluating classifier={classifier_name}, mode={feature_mode}, topk={top_k}"
                )
                summary, avg_cm = evaluate_config(
                    X=X,
                    y=y,
                    class_names=class_names,
                    classifier_name=classifier_name,
                    shot=args.shot,
                    repeats=args.repeats,
                    seed=args.seed + top_k,
                )
                row = {
                    "feature_mode": feature_mode,
                    "top_k": int(top_k),
                    "classifier": classifier_name,
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
                f"Best IAD {args.shot}-shot Confusion Matrix\n"
                f"{best['row']['classifier']} | {best['row']['feature_mode']} | topk={best['row']['top_k']}"
            ),
        )

    log(f"Scan complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
