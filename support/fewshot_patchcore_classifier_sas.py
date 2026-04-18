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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from cluster_patchcore_sas import build_patch_lookup, build_random_projection


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def build_classifier(classifier_name: str, seed: int):
    if classifier_name == "linear_svm":
        model = LinearSVC(
            random_state=seed,
            class_weight="balanced",
            dual="auto",
            max_iter=20000,
        )
    elif classifier_name == "logreg":
        model = LogisticRegression(
            random_state=seed,
            class_weight="balanced",
            max_iter=10000,
            multi_class="auto",
        )
    else:
        raise ValueError(f"Unsupported classifier: {classifier_name}")

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", model),
        ]
    )


def save_confusion_heatmap(
    matrix: np.ndarray,
    class_names: list[str],
    output_path: str,
    title: str,
) -> None:
    plt.figure(figsize=(6, 5))
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


def build_semantic_feature(
    patch_entry: dict,
    feature_mode: str,
    top_k_patches: int,
    gram_projection: np.ndarray | None,
) -> np.ndarray:
    patch_scores = patch_entry["patch_scores"].astype(np.float32, copy=False).reshape(-1)
    patch_embeddings = patch_entry["patch_embeddings"].astype(np.float32, copy=False).reshape(
        -1,
        patch_entry["patch_embeddings"].shape[-1],
    )

    top_k = min(int(top_k_patches), int(patch_scores.shape[0]))
    if top_k <= 0:
        raise ValueError("top_k_patches must be > 0")

    selected_idx = np.argpartition(patch_scores, -top_k)[-top_k:]
    selected_scores = patch_scores[selected_idx]
    selected_embeddings = patch_embeddings[selected_idx]

    selected_scores = selected_scores - float(selected_scores.min())
    if float(selected_scores.max()) > 0:
        selected_scores = selected_scores / float(selected_scores.max())
    weights = selected_scores + 1e-6

    mean_embedding = np.average(selected_embeddings, axis=0, weights=weights)
    mean_norm = float(np.linalg.norm(mean_embedding))
    if mean_norm > 0:
        mean_embedding = mean_embedding / mean_norm

    if feature_mode == "mean":
        return mean_embedding.astype(np.float32, copy=False)

    if gram_projection is None:
        raise ValueError("gram_projection is required for mean_plus_gram")
    projected = selected_embeddings @ gram_projection
    projected = projected.astype(np.float32, copy=False)
    gram = (projected * weights[:, None]).T @ projected / max(float(weights.sum()), 1e-6)
    upper = gram[np.triu_indices_from(gram)].astype(np.float32, copy=False)
    if feature_mode == "gram":
        return upper
    if feature_mode != "mean_plus_gram":
        raise ValueError(f"Unsupported feature mode: {feature_mode}")
    return np.concatenate([mean_embedding.astype(np.float32, copy=False), upper], axis=0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Few-shot abnormal subtype classification on top of SAS PatchCore features"
    )
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--patchcore-output-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--feature-mode",
        default="mean_plus_gram",
        choices=["mean", "gram", "mean_plus_gram"],
    )
    parser.add_argument("--threshold-percentile", type=float, default=90.0)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--gram-dim", type=int, default=32)
    parser.add_argument("--top-k-patches", type=int, default=64)
    parser.add_argument(
        "--classifier",
        default="linear_svm",
        choices=["linear_svm", "logreg"],
    )
    parser.add_argument(
        "--feature-cache",
        default="",
        help="Optional path to a .npz cache for abnormal sample features",
    )
    parser.add_argument(
        "--shots",
        default="1,5,10,20",
        help="Comma-separated shots per class",
    )
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    dataset_dir = args.dataset_dir
    patchcore_output_dir = args.patchcore_output_dir

    test_label_names = np.load(os.path.join(dataset_dir, "test_labels.npy")).astype(str)
    patch_npz = np.load(os.path.join(patchcore_output_dir, "test_patch_artifacts.npz"))

    abnormal_mask = test_label_names != "normal"
    abnormal_indices = np.where(abnormal_mask)[0]
    class_names = sorted(np.unique(test_label_names[abnormal_mask]).tolist())
    class_to_id = {name: i for i, name in enumerate(class_names)}

    cache_path = args.feature_cache or os.path.join(
        args.output_dir,
        f"feature_cache_{args.feature_mode}.npz",
    )

    if os.path.exists(cache_path):
        log(f"Loading cached features from {cache_path}")
        cache = np.load(cache_path, allow_pickle=True)
        X = cache["X"].astype(np.float32)
        y = cache["y"].astype(np.int64)
        sample_indices = cache["sample_indices"].astype(np.int64)
        label_names = cache["label_names"].astype(str)
        meta_df = pd.DataFrame(
            {
                "sample_index": sample_indices,
                "label_name": label_names,
                "label_id": y,
            }
        )
    else:
        patch_lookup = build_patch_lookup(patch_npz)
        gram_projection = None
        if args.feature_mode == "mean_plus_gram":
            gram_projection = build_random_projection(
                input_dim=int(patch_npz["embedding_dim"][0]),
                output_dim=args.gram_dim,
                seed=args.seed,
            )

        log(
            f"Building {args.feature_mode} features for {abnormal_indices.shape[0]} abnormal samples"
        )
        feature_rows = []
        features = []
        labels = []
        sample_indices = []
        label_names = []
        for sample_index in abnormal_indices:
            feature_vector = build_semantic_feature(
                patch_entry=patch_lookup[int(sample_index)],
                feature_mode=args.feature_mode,
                top_k_patches=args.top_k_patches,
                gram_projection=gram_projection,
            )
            features.append(feature_vector)
            label_name = str(test_label_names[sample_index])
            label_id = class_to_id[label_name]
            labels.append(label_id)
            sample_indices.append(int(sample_index))
            label_names.append(label_name)
            feature_rows.append(
                {
                    "sample_index": int(sample_index),
                    "label_name": label_name,
                    "label_id": label_id,
                }
            )

        X = np.stack(features).astype(np.float32)
        y = np.asarray(labels, dtype=np.int64)
        sample_indices = np.asarray(sample_indices, dtype=np.int64)
        label_names = np.asarray(label_names, dtype="U32")
        np.savez_compressed(
            cache_path,
            X=X,
            y=y,
            sample_indices=sample_indices,
            label_names=label_names,
        )
        log(f"Saved feature cache to {cache_path}")
        meta_df = pd.DataFrame(feature_rows)

    meta_df.to_csv(os.path.join(args.output_dir, "feature_index.csv"), index=False)

    class_index_map = {
        class_id: np.where(y == class_id)[0] for class_id in range(len(class_names))
    }
    shots = [int(part.strip()) for part in args.shots.split(",") if part.strip()]

    detail_rows = []
    summary_rows = []

    for shot in shots:
        if any(indices.shape[0] <= shot for indices in class_index_map.values()):
            log(f"Skipping shot={shot} because at least one class has too few samples")
            continue

        confusion_sum = np.zeros((len(class_names), len(class_names)), dtype=np.float64)
        metrics_for_shot = []

        for repeat in range(args.repeats):
            rng = np.random.default_rng(args.seed + shot * 1000 + repeat)
            train_parts = []
            test_parts = []
            for class_id, indices in class_index_map.items():
                shuffled = rng.permutation(indices)
                train_parts.append(shuffled[:shot])
                test_parts.append(shuffled[shot:])
            train_idx = np.concatenate(train_parts)
            test_idx = np.concatenate(test_parts)

            X_train = X[train_idx]
            y_train = y[train_idx]
            X_test = X[test_idx]
            y_test = y[test_idx]

            model = build_classifier(args.classifier, seed=args.seed + repeat)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = float(accuracy_score(y_test, y_pred))
            macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
            cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(class_names)))
            confusion_sum += cm

            row = {
                "shot": shot,
                "repeat": repeat,
                "train_size": int(train_idx.shape[0]),
                "test_size": int(test_idx.shape[0]),
                "accuracy": acc,
                "macro_f1": macro_f1,
            }
            detail_rows.append(row)
            metrics_for_shot.append(row)

        avg_cm = confusion_sum / float(args.repeats)
        save_confusion_heatmap(
            avg_cm,
            class_names=class_names,
            output_path=os.path.join(args.output_dir, f"confusion_matrix_{shot}shot.png"),
            title=f"{shot}-shot Average Confusion Matrix",
        )

        shot_df = pd.DataFrame(metrics_for_shot)
        summary_rows.append(
            {
                "shot": shot,
                "repeats": int(args.repeats),
                "train_size_per_repeat": int(shot * len(class_names)),
                "test_size_per_repeat": int(len(y) - shot * len(class_names)),
                "accuracy_mean": float(shot_df["accuracy"].mean()),
                "accuracy_std": float(shot_df["accuracy"].std(ddof=0)),
                "macro_f1_mean": float(shot_df["macro_f1"].mean()),
                "macro_f1_std": float(shot_df["macro_f1"].std(ddof=0)),
            }
        )
        log(
            f"{shot}-shot done: "
            f"acc={summary_rows[-1]['accuracy_mean']:.4f}±{summary_rows[-1]['accuracy_std']:.4f}, "
            f"macro_f1={summary_rows[-1]['macro_f1_mean']:.4f}±{summary_rows[-1]['macro_f1_std']:.4f}"
        )

    detail_df = pd.DataFrame(detail_rows)
    summary_df = pd.DataFrame(summary_rows)
    detail_df.to_csv(os.path.join(args.output_dir, "fewshot_detail_results.csv"), index=False)
    summary_df.to_csv(os.path.join(args.output_dir, "fewshot_summary.csv"), index=False)

    with open(os.path.join(args.output_dir, "fewshot_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_dir": args.dataset_dir,
                "patchcore_output_dir": args.patchcore_output_dir,
                "feature_mode": args.feature_mode,
                "classifier": args.classifier,
                "class_names": class_names,
                "num_abnormal_samples": int(len(y)),
                "results": summary_rows,
            },
            f,
            indent=2,
        )

    log(f"Few-shot classification results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
