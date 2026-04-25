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
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

from cluster_patchcore_sas import build_random_projection


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
    elif classifier_name == "rbf_svm":
        model = SVC(
            kernel="rbf",
            C=3.0,
            gamma="scale",
            class_weight="balanced",
            random_state=seed,
        )
    elif classifier_name == "logreg":
        model = LogisticRegression(
            random_state=seed,
            class_weight="balanced",
            max_iter=10000,
            multi_class="auto",
        )
    elif classifier_name == "mlp":
        model = MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=600,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=20,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unsupported classifier: {classifier_name}")

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", model),
        ]
    )


def build_semantic_feature(
    patch_embeddings: np.ndarray,
    patch_scores: np.ndarray,
    feature_mode: str,
    top_k_patches: int,
    gram_projection: np.ndarray | None,
) -> np.ndarray:
    patch_scores = patch_scores.astype(np.float32, copy=False).reshape(-1)
    patch_embeddings = patch_embeddings.astype(np.float32, copy=False).reshape(
        -1,
        patch_embeddings.shape[-1],
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
    gram = (projected * weights[:, None]).T @ projected / max(float(weights.sum()), 1e-6)
    upper = gram[np.triu_indices_from(gram)].astype(np.float32, copy=False)
    return np.concatenate([mean_embedding.astype(np.float32, copy=False), upper], axis=0)


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
        description="Few-shot modulation classification on top of IAD PatchCore features"
    )
    parser.add_argument(
        "--dataset-specs",
        nargs="+",
        required=True,
        help="Repeated specs of the form CLASS_NAME::OUTPUT_DIR",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--feature-mode",
        default="mean_plus_gram",
        choices=["mean", "mean_plus_gram"],
    )
    parser.add_argument("--gram-dim", type=int, default=32)
    parser.add_argument("--top-k-patches", type=int, default=16)
    parser.add_argument(
        "--classifier",
        default="linear_svm",
        choices=["linear_svm", "rbf_svm", "logreg", "mlp"],
    )
    parser.add_argument("--shot", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--feature-cache",
        default="",
        help="Optional path to a .npz cache for abnormal sample features",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parse_dataset_specs(specs: list[str]) -> list[tuple[str, str]]:
    parsed = []
    for spec in specs:
        if "::" not in spec:
            raise ValueError(f"Invalid dataset spec: {spec}")
        class_name, output_dir = spec.split("::", 1)
        parsed.append((class_name, output_dir))
    return parsed


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    dataset_specs = parse_dataset_specs(args.dataset_specs)
    class_names = [name for name, _ in dataset_specs]
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
        feature_index = pd.DataFrame(
            {
                "class_name": cache["class_names"].astype(str),
                "sample_index": cache["sample_indices"].astype(np.int64),
                "score": cache["scores"].astype(np.float32),
            }
        )
    else:
        log(f"Building {args.feature_mode} features from {len(dataset_specs)} IAD outputs")
        X_parts = []
        y_parts = []
        feature_rows = []
        gram_projection = None

        for class_name, output_dir in dataset_specs:
            patch_path = os.path.join(output_dir, "test_patch_artifacts.npz")
            patch_npz = np.load(patch_path)
            if gram_projection is None and args.feature_mode == "mean_plus_gram":
                gram_projection = build_random_projection(
                    input_dim=int(patch_npz["embedding_dim"][0]),
                    output_dim=args.gram_dim,
                    seed=args.seed,
                )

            sample_indices = patch_npz["sample_indices"].astype(np.int64)
            patch_scores = patch_npz["patch_scores"]
            patch_embeddings = patch_npz["patch_embeddings"]
            image_scores = patch_npz["image_scores"].astype(np.float32)

            for i in range(sample_indices.shape[0]):
                feature_vector = build_semantic_feature(
                    patch_embeddings=patch_embeddings[i],
                    patch_scores=patch_scores[i],
                    feature_mode=args.feature_mode,
                    top_k_patches=args.top_k_patches,
                    gram_projection=gram_projection,
                )
                X_parts.append(feature_vector)
                y_parts.append(class_to_id[class_name])
                feature_rows.append(
                    {
                        "class_name": class_name,
                        "sample_index": int(sample_indices[i]),
                        "score": float(image_scores[i]),
                    }
                )

        X = np.stack(X_parts).astype(np.float32)
        y = np.asarray(y_parts, dtype=np.int64)
        feature_index = pd.DataFrame(feature_rows)
        np.savez_compressed(
            cache_path,
            X=X,
            y=y,
            class_names=feature_index["class_name"].to_numpy(dtype="U32"),
            sample_indices=feature_index["sample_index"].to_numpy(dtype=np.int64),
            scores=feature_index["score"].to_numpy(dtype=np.float32),
        )
        log(f"Saved feature cache to {cache_path}")

    feature_index.to_csv(os.path.join(args.output_dir, "feature_index.csv"), index=False)

    class_index_map = {class_id: np.where(y == class_id)[0] for class_id in range(len(class_names))}
    if any(indices.shape[0] <= args.shot for indices in class_index_map.values()):
        raise ValueError("At least one class has fewer samples than the requested shot count")

    confusion_sum = np.zeros((len(class_names), len(class_names)), dtype=np.float64)
    detail_rows = []

    for repeat in range(args.repeats):
        rng = np.random.default_rng(args.seed + repeat)
        train_parts = []
        test_parts = []
        for class_id, indices in class_index_map.items():
            shuffled = rng.permutation(indices)
            train_parts.append(shuffled[: args.shot])
            test_parts.append(shuffled[args.shot :])
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

        detail_rows.append(
            {
                "repeat": repeat,
                "train_size": int(train_idx.shape[0]),
                "test_size": int(test_idx.shape[0]),
                "accuracy": acc,
                "macro_f1": macro_f1,
            }
        )

    avg_cm = confusion_sum / float(args.repeats)
    save_confusion_heatmap(
        avg_cm,
        class_names=class_names,
        output_path=os.path.join(args.output_dir, f"confusion_matrix_{args.shot}shot.png"),
        title=f"IAD {args.shot}-shot Average Confusion Matrix",
    )

    detail_df = pd.DataFrame(detail_rows)
    summary = {
        "shot": int(args.shot),
        "repeats": int(args.repeats),
        "train_size_per_repeat": int(args.shot * len(class_names)),
        "test_size_per_repeat": int(len(y) - args.shot * len(class_names)),
        "accuracy_mean": float(detail_df["accuracy"].mean()),
        "accuracy_std": float(detail_df["accuracy"].std(ddof=0)),
        "macro_f1_mean": float(detail_df["macro_f1"].mean()),
        "macro_f1_std": float(detail_df["macro_f1"].std(ddof=0)),
        "class_names": class_names,
        "num_samples_per_class": {name: int(class_index_map[i].shape[0]) for i, name in enumerate(class_names)},
    }

    detail_df.to_csv(os.path.join(args.output_dir, "fewshot_detail_results.csv"), index=False)
    pd.DataFrame([summary]).to_csv(os.path.join(args.output_dir, "fewshot_summary.csv"), index=False)
    with open(os.path.join(args.output_dir, "fewshot_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log(
        f"{args.shot}-shot done: "
        f"acc={summary['accuracy_mean']:.4f}±{summary['accuracy_std']:.4f}, "
        f"macro_f1={summary['macro_f1_mean']:.4f}±{summary['macro_f1_std']:.4f}"
    )
    log(f"Few-shot classification results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
