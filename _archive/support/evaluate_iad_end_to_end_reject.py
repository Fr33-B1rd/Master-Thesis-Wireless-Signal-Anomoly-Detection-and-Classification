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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

from fewshot_patchcore_classifier_iad import build_classifier, build_semantic_feature
from cluster_patchcore_sas import build_random_projection


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def allocate_quota(cluster_sizes: np.ndarray, shot: int) -> np.ndarray:
    num_clusters = int(cluster_sizes.shape[0])
    quotas = np.ones(num_clusters, dtype=int)
    remaining = int(shot - num_clusters)
    if remaining <= 0:
        return quotas
    free_sizes = np.maximum(cluster_sizes - 1, 0)
    total_free = int(free_sizes.sum())
    if total_free == 0:
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


def select_random(indices: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    shuffled = rng.permutation(indices)
    return shuffled[:count]


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
    scaled = StandardScaler().fit_transform(X_class)
    reduced_dim = min(int(pca_dim), scaled.shape[1], scaled.shape[0])
    reduced = PCA(n_components=reduced_dim, random_state=seed).fit_transform(scaled)
    labels = KMeans(n_clusters=cluster_count, random_state=seed + repeat, n_init=20).fit_predict(reduced)
    cluster_sizes = np.bincount(labels, minlength=cluster_count)
    quotas = allocate_quota(cluster_sizes, shot)
    rng = np.random.default_rng(seed + repeat)
    selected = []
    for cid in range(cluster_count):
        member_pos = np.where(labels == cid)[0]
        member_global = global_indices[member_pos]
        selected.append(select_random(member_global, int(quotas[cid]), rng))
    return np.concatenate(selected)


def parse_dataset_specs(specs: list[str]) -> list[tuple[str, str]]:
    parsed = []
    for spec in specs:
        if "::" not in spec:
            raise ValueError(f"Invalid dataset spec: {spec}")
        name, output_dir = spec.split("::", 1)
        parsed.append((name, output_dir))
    return parsed


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate end-to-end IAD PatchCore + subtype classifier with reject threshold"
    )
    parser.add_argument(
        "--dataset-specs",
        nargs="+",
        required=True,
        help="Repeated specs of the form CLASS_NAME::OUTPUT_DIR with shared PatchCore outputs",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shot", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--classifier", default="logreg")
    parser.add_argument("--classifier-c", type=float, default=0.1)
    parser.add_argument("--class-weight", default="none", choices=["none", "balanced"])
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--gram-dim", type=int, default=64)
    parser.add_argument("--cluster-count", type=int, default=3)
    parser.add_argument("--cluster-pca-dim", type=int, default=32)
    parser.add_argument(
        "--reject-thresholds",
        default="0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    dataset_specs = parse_dataset_specs(args.dataset_specs)
    anomaly_class_names = [name for name, _ in dataset_specs]
    true_label_names = ["normal", *anomaly_class_names]
    pred_label_names = ["normal", "unknown", *anomaly_class_names]
    pred_to_col = {name: idx for idx, name in enumerate(pred_label_names)}

    class_to_id = {name: i for i, name in enumerate(anomaly_class_names)}
    samples = []
    embedding_dim = None

    for class_name, output_dir in dataset_specs:
        patch_npz = np.load(os.path.join(output_dir, "test_patch_artifacts.npz"))
        score_df = pd.read_csv(os.path.join(output_dir, "test_scores.csv"))
        if embedding_dim is None:
            embedding_dim = int(patch_npz["embedding_dim"][0])

        sample_indices = patch_npz["sample_indices"].astype(np.int64)
        patch_scores = patch_npz["patch_scores"]
        patch_embeddings = patch_npz["patch_embeddings"]
        image_scores = patch_npz["image_scores"].astype(np.float32)
        labels = patch_npz["labels"].astype(np.int64)

        predictions = score_df["prediction"].to_numpy(dtype=np.int64)
        if predictions.shape[0] != sample_indices.shape[0]:
            raise ValueError(f"{class_name}: expected export_selection=all patch artifacts")

        for i in range(sample_indices.shape[0]):
            if int(labels[i]) == 1:
                true_name = class_name
                anomaly_label_id = class_to_id[class_name]
            else:
                true_name = "normal"
                anomaly_label_id = -1
            samples.append(
                {
                    "source_class": class_name,
                    "sample_index": int(sample_indices[i]),
                    "true_name": true_name,
                    "is_abnormal": int(labels[i]) == 1,
                    "anomaly_label_id": anomaly_label_id,
                    "patchcore_prediction": int(predictions[i]),
                    "score": float(image_scores[i]),
                    "patch_scores": patch_scores[i],
                    "patch_embeddings": patch_embeddings[i],
                }
            )

    gram_projection = build_random_projection(
        input_dim=int(embedding_dim),
        output_dim=args.gram_dim,
        seed=args.seed,
    )

    log("Building semantic features for all exported samples")
    X_all = []
    anomaly_ids = []
    for sample in samples:
        X_all.append(
            build_semantic_feature(
                patch_embeddings=sample["patch_embeddings"],
                patch_scores=sample["patch_scores"],
                feature_mode="mean_plus_gram",
                top_k_patches=args.top_k,
                gram_projection=gram_projection,
            )
        )
        anomaly_ids.append(int(sample["anomaly_label_id"]))
    X_all = np.stack(X_all).astype(np.float32)
    anomaly_ids = np.asarray(anomaly_ids, dtype=np.int64)

    abnormal_global_indices = np.where(anomaly_ids >= 0)[0]
    class_index_map = {
        class_id: abnormal_global_indices[anomaly_ids[abnormal_global_indices] == class_id]
        for class_id in range(len(anomaly_class_names))
    }

    threshold_values = [float(part.strip()) for part in args.reject_thresholds.split(",") if part.strip()]
    results = []
    confusion_sums = {
        threshold: np.zeros((len(true_label_names), len(pred_label_names)), dtype=np.float64)
        for threshold in threshold_values
    }

    for repeat in range(args.repeats):
        train_parts = []
        test_abnormal_parts = []
        for class_id, indices in class_index_map.items():
            train_idx = select_cluster_stratified(
                X_class=X_all[indices],
                global_indices=indices,
                shot=args.shot,
                cluster_count=args.cluster_count,
                pca_dim=args.cluster_pca_dim,
                seed=args.seed + 1000 * class_id,
                repeat=repeat,
            )
            train_mask = np.zeros(indices.shape[0], dtype=bool)
            lookup = {value: pos for pos, value in enumerate(indices.tolist())}
            for value in train_idx.tolist():
                train_mask[lookup[value]] = True
            test_idx = indices[~train_mask]
            train_parts.append(train_idx)
            test_abnormal_parts.append(test_idx)

        train_idx = np.concatenate(train_parts)
        test_abnormal_idx = np.concatenate(test_abnormal_parts)
        test_normal_idx = np.where(anomaly_ids < 0)[0]
        eval_idx = np.concatenate([test_normal_idx, test_abnormal_idx])

        class_weight = None if args.class_weight == "none" else "balanced"
        model = build_classifier(args.classifier, seed=args.seed + repeat)
        if args.classifier == "logreg":
            model.named_steps["clf"].set_params(C=args.classifier_c, class_weight=class_weight)
        model.fit(X_all[train_idx], anomaly_ids[train_idx])

        if not hasattr(model, "predict_proba"):
            raise RuntimeError("Reject evaluation currently requires probabilistic classifier output")

        proba = model.predict_proba(X_all[eval_idx])
        pred_class_ids = model.classes_[np.argmax(proba, axis=1)]
        pred_conf = np.max(proba, axis=1)

        for threshold in threshold_values:
            true_names = []
            pred_names = []
            fp_normals = 0
            rejected_normals = 0
            rejected_abnormals = 0
            accepted_abnormals = 0
            accepted_abnormal_correct = 0

            for local_pos, global_idx in enumerate(eval_idx):
                sample = samples[int(global_idx)]
                true_name = sample["true_name"]
                patchcore_pred = int(sample["patchcore_prediction"])

                if patchcore_pred == 0:
                    pred_name = "normal"
                else:
                    if pred_conf[local_pos] < threshold:
                        pred_name = "unknown"
                    else:
                        pred_name = anomaly_class_names[int(pred_class_ids[local_pos])]

                true_names.append(true_name)
                pred_names.append(pred_name)

                if true_name == "normal" and patchcore_pred == 1:
                    fp_normals += 1
                    if pred_name == "unknown":
                        rejected_normals += 1

                if true_name != "normal":
                    if pred_name == "unknown":
                        rejected_abnormals += 1
                    elif pred_name != "normal":
                        accepted_abnormals += 1
                        if pred_name == true_name:
                            accepted_abnormal_correct += 1

            row_to_idx = {name: idx for idx, name in enumerate(true_label_names)}
            cm = np.zeros((len(true_label_names), len(pred_label_names)), dtype=np.float64)
            for true_name, pred_name in zip(true_names, pred_names):
                cm[row_to_idx[true_name], pred_to_col[pred_name]] += 1
            confusion_sums[threshold] += cm

            true_known = np.array(true_names, dtype=object)
            pred_known = np.array(pred_names, dtype=object)
            accuracy = float(np.mean(true_known == pred_known))
            macro_f1 = float(
                f1_score(
                    true_known,
                    pred_known,
                    labels=true_label_names,
                    average="macro",
                    zero_division=0,
                )
            )
            results.append(
                {
                    "repeat": repeat,
                    "reject_threshold": threshold,
                    "overall_accuracy": accuracy,
                    "macro_f1_known": macro_f1,
                    "patchcore_false_positive_normals": int(fp_normals),
                    "rejected_false_positive_normals": int(rejected_normals),
                    "fp_normal_reject_rate": float(rejected_normals / max(fp_normals, 1)),
                    "rejected_true_abnormals": int(rejected_abnormals),
                    "accepted_true_abnormals": int(accepted_abnormals),
                    "abnormal_coverage": float(accepted_abnormals / max(np.sum(true_known != "normal"), 1)),
                    "accepted_abnormal_accuracy": float(
                        accepted_abnormal_correct / max(accepted_abnormals, 1)
                    ),
                }
            )

    detail_df = pd.DataFrame(results)
    summary_df = (
        detail_df.groupby("reject_threshold", as_index=False)
        .agg(
            overall_accuracy_mean=("overall_accuracy", "mean"),
            overall_accuracy_std=("overall_accuracy", "std"),
            macro_f1_known_mean=("macro_f1_known", "mean"),
            macro_f1_known_std=("macro_f1_known", "std"),
            fp_normal_reject_rate_mean=("fp_normal_reject_rate", "mean"),
            fp_normal_reject_rate_std=("fp_normal_reject_rate", "std"),
            abnormal_coverage_mean=("abnormal_coverage", "mean"),
            abnormal_coverage_std=("abnormal_coverage", "std"),
            accepted_abnormal_accuracy_mean=("accepted_abnormal_accuracy", "mean"),
            accepted_abnormal_accuracy_std=("accepted_abnormal_accuracy", "std"),
        )
        .sort_values("macro_f1_known_mean", ascending=False)
    )

    detail_df.to_csv(os.path.join(args.output_dir, "reject_threshold_detail.csv"), index=False)
    summary_df.to_csv(os.path.join(args.output_dir, "reject_threshold_summary.csv"), index=False)

    best_threshold = float(summary_df.iloc[0]["reject_threshold"])
    with open(os.path.join(args.output_dir, "best_threshold.json"), "w", encoding="utf-8") as f:
        json.dump(summary_df.iloc[0].to_dict(), f, indent=2)

    avg_cm = confusion_sums[best_threshold] / float(args.repeats)
    save_rectangular_confusion(
        avg_cm,
        row_labels=true_label_names,
        col_labels=pred_label_names,
        output_path=os.path.join(args.output_dir, "best_threshold_confusion_matrix.png"),
        title=f"Best Reject Threshold = {best_threshold:.2f}",
    )

    log(f"Reject-threshold evaluation saved to {args.output_dir}")


if __name__ == "__main__":
    main()
