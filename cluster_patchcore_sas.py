# Clusters SAS abnormal samples from PatchCore outputs with anomaly-region features
# and optional mean/gram semantic embeddings.

import argparse
import json
import os
import time
from dataclasses import dataclass

import hdbscan
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from PIL import Image
from scipy import ndimage
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from dataio.sas import load_sas_test_split


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def normalize_map(anomaly_map: np.ndarray) -> np.ndarray:
    anomaly_map = anomaly_map.astype(np.float32, copy=False)
    min_value = float(anomaly_map.min())
    max_value = float(anomaly_map.max())
    if max_value <= min_value:
        return np.zeros_like(anomaly_map, dtype=np.float32)
    return (anomaly_map - min_value) / (max_value - min_value)


def resize_array(array: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if array.ndim == 1:
        array = array[np.newaxis, :]
    pil = Image.fromarray(array.astype(np.float32))
    resized = pil.resize((size[1], size[0]), resample=Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def largest_connected_region(
    normalized_map: np.ndarray, threshold_percentile: float
) -> tuple[np.ndarray, dict]:
    threshold_value = float(np.percentile(normalized_map, threshold_percentile))
    binary_mask = normalized_map >= threshold_value
    if not np.any(binary_mask):
        flat_index = int(np.argmax(normalized_map))
        binary_mask = np.zeros_like(normalized_map, dtype=bool)
        binary_mask.flat[flat_index] = True

    labeled, num_components = ndimage.label(binary_mask)
    if num_components == 0:
        flat_index = int(np.argmax(normalized_map))
        binary_mask = np.zeros_like(normalized_map, dtype=bool)
        binary_mask.flat[flat_index] = True
        labeled, num_components = ndimage.label(binary_mask)

    component_sizes = ndimage.sum(binary_mask, labeled, index=np.arange(1, num_components + 1))
    largest_component = int(np.argmax(component_sizes)) + 1
    mask = labeled == largest_component

    y_coords, x_coords = np.where(mask)
    y_min, y_max = int(y_coords.min()), int(y_coords.max())
    x_min, x_max = int(x_coords.min()), int(x_coords.max())
    props = {
        "threshold_value": threshold_value,
        "num_components": int(num_components),
        "y_min": y_min,
        "y_max": y_max,
        "x_min": x_min,
        "x_max": x_max,
    }
    return mask, props


@dataclass
class SampleFeature:
    sample_index: int
    label_name: str
    score: float
    prediction: int
    feature_vector: np.ndarray
    region_props: dict


def build_random_projection(
    input_dim: int, output_dim: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    projection = rng.standard_normal((input_dim, output_dim), dtype=np.float32)
    projection /= np.sqrt(float(input_dim))
    return projection.astype(np.float32, copy=False)


def build_patch_lookup(patch_npz: np.lib.npyio.NpzFile) -> dict[int, dict]:
    sample_indices = patch_npz["sample_indices"].astype(np.int64)
    labels = patch_npz["labels"].astype(np.int64)
    patch_scores = patch_npz["patch_scores"]
    patch_embeddings = patch_npz["patch_embeddings"]

    lookup = {}
    for i, sample_index in enumerate(sample_indices):
        lookup[int(sample_index)] = {
            "label": int(labels[i]),
            "patch_scores": patch_scores[i].astype(np.float32, copy=False),
            "patch_embeddings": patch_embeddings[i].astype(np.float32, copy=False),
        }
    return lookup


def extract_feature_vector(
    raw_image: np.ndarray,
    anomaly_map: np.ndarray,
    threshold_percentile: float,
    patch_size: int,
    patch_entry: dict | None = None,
    feature_mode: str = "base",
    gram_projection: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    raw_image = raw_image.astype(np.float32, copy=False)
    normalized_map = normalize_map(anomaly_map)
    mask, props = largest_connected_region(normalized_map, threshold_percentile)

    y_min = props["y_min"]
    y_max = props["y_max"]
    x_min = props["x_min"]
    x_max = props["x_max"]

    map_crop = normalized_map[y_min : y_max + 1, x_min : x_max + 1]
    raw_crop = raw_image[y_min : y_max + 1, x_min : x_max + 1]
    mask_crop = mask[y_min : y_max + 1, x_min : x_max + 1]

    masked_map_values = normalized_map[mask]
    masked_raw_values = raw_image[mask]

    image_height, image_width = raw_image.shape
    bbox_height = y_max - y_min + 1
    bbox_width = x_max - x_min + 1
    area = int(mask.sum())
    area_ratio = area / float(image_height * image_width)

    center_y, center_x = ndimage.center_of_mass(mask.astype(np.float32))
    center_y = float(center_y / image_height)
    center_x = float(center_x / image_width)

    map_resized = resize_array(map_crop, (patch_size, patch_size))
    raw_resized = resize_array(raw_crop, (patch_size, patch_size))
    masked_crop = raw_crop * mask_crop.astype(np.float32)
    masked_resized = resize_array(masked_crop, (patch_size, patch_size))

    row_profile = resize_array(mask.astype(np.float32).mean(axis=1), (1, patch_size)).reshape(-1)
    col_profile = resize_array(mask.astype(np.float32).mean(axis=0), (1, patch_size)).reshape(-1)

    stats = np.array(
        [
            center_y,
            center_x,
            bbox_height / image_height,
            bbox_width / image_width,
            area_ratio,
            bbox_width / max(bbox_height, 1),
            float(masked_map_values.mean()),
            float(masked_map_values.std()),
            float(masked_map_values.max()),
            float(normalized_map.mean()),
            float(raw_crop.mean()),
            float(raw_crop.std()),
            float(masked_raw_values.mean()),
            float(masked_raw_values.std()),
            float(masked_raw_values.max()),
            float(masked_raw_values.min()),
            float(np.square(masked_raw_values).mean()),
            float(props["num_components"]),
        ],
        dtype=np.float32,
    )

    base_features = np.concatenate(
        [
            stats,
            map_resized.reshape(-1),
            raw_resized.reshape(-1),
            masked_resized.reshape(-1),
            row_profile,
            col_profile,
        ]
    ).astype(np.float32)

    if feature_mode == "base" or patch_entry is None:
        return base_features, props

    patch_scores = patch_entry["patch_scores"]
    patch_embeddings = patch_entry["patch_embeddings"]
    patch_mask = resize_array(mask.astype(np.float32), patch_scores.shape)
    patch_mask = patch_mask >= 0.25
    if not np.any(patch_mask):
        flat_index = int(np.argmax(patch_scores))
        patch_mask = np.zeros_like(patch_scores, dtype=bool)
        patch_mask.flat[flat_index] = True

    region_scores = patch_scores[patch_mask].astype(np.float32, copy=False)
    region_embeddings = patch_embeddings[patch_mask].astype(np.float32, copy=False)

    region_scores = region_scores - float(region_scores.min())
    if float(region_scores.max()) > 0:
        region_scores = region_scores / float(region_scores.max())
    weights = region_scores + 1e-6

    mean_embedding = np.average(region_embeddings, axis=0, weights=weights)
    mean_norm = float(np.linalg.norm(mean_embedding))
    if mean_norm > 0:
        mean_embedding = mean_embedding / mean_norm
    semantic_features = [mean_embedding.astype(np.float32, copy=False)]

    if feature_mode == "mean_plus_gram":
        if gram_projection is None:
            raise ValueError("gram_projection is required for mean_plus_gram mode")
        projected = region_embeddings @ gram_projection
        projected = projected.astype(np.float32, copy=False)
        weight_sum = float(weights.sum())
        gram = (projected * weights[:, None]).T @ projected / max(weight_sum, 1e-6)
        upper = gram[np.triu_indices_from(gram)].astype(np.float32, copy=False)
        semantic_features.append(upper)

    feature_vector = np.concatenate([base_features, *semantic_features]).astype(np.float32)
    return feature_vector, props


def plot_umap_projection(df: pd.DataFrame, output_path: str) -> None:
    plt.figure(figsize=(10, 8))
    clusters = sorted(df["cluster"].unique().tolist())
    palette = sns.color_palette("tab10", n_colors=max(1, len(clusters)))
    color_map = {cluster: palette[i % len(palette)] for i, cluster in enumerate(clusters)}

    for cluster in clusters:
        cluster_df = df[df["cluster"] == cluster]
        label = "noise" if cluster == -1 else f"cluster {cluster}"
        plt.scatter(
            cluster_df["umap_x"],
            cluster_df["umap_y"],
            s=20 if cluster == -1 else 28,
            alpha=0.55 if cluster == -1 else 0.8,
            c=[color_map[cluster]],
            label=label,
            edgecolors="none",
        )

    plt.title("SAS Anomaly Clustering Projection (UMAP)")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_cluster_composition(df: pd.DataFrame, output_path: str) -> None:
    clustered = df[df["cluster"] != -1].copy()
    if clustered.empty:
        return

    composition = (
        clustered.groupby(["cluster", "label_name"])
        .size()
        .reset_index(name="count")
        .sort_values(["cluster", "label_name"])
    )
    pivot = composition.pivot(index="cluster", columns="label_name", values="count").fillna(0)
    pivot = pivot.sort_index()

    noise_df = df[df["cluster"] == -1]
    noise_counts = noise_df["label_name"].value_counts().sort_index()
    noise_text = ", ".join(f"{name}={int(count)}" for name, count in noise_counts.items())

    ax = pivot.plot(kind="bar", stacked=True, figsize=(10, 6), colormap="tab20")
    ax.set_title("Cluster Composition by Anomaly Type")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Sample Count")
    ax.legend(title="Label", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.figtext(
        0.02,
        0.01,
        f"Noise count: {int(noise_df.shape[0])}" + (f" ({noise_text})" if noise_text else ""),
        ha="left",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def save_cluster_montages(
    result_df: pd.DataFrame,
    test_images: np.ndarray,
    anomaly_maps: np.ndarray,
    output_dir: str,
    num_images: int,
) -> None:
    clustered = result_df[result_df["cluster"] != -1].copy()
    if clustered.empty:
        return

    for cluster_id in sorted(clustered["cluster"].unique().tolist()):
        cluster_df = clustered[clustered["cluster"] == cluster_id].copy()
        cluster_df = cluster_df.sort_values("score", ascending=False).head(num_images)
        selected = cluster_df["sample_index"].astype(int).tolist()

        cols = 5
        rows = int(np.ceil(len(selected) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
        axes = np.array(axes).reshape(-1)
        for ax in axes:
            ax.axis("off")

        for plot_idx, sample_index in enumerate(selected):
            ax = axes[plot_idx]
            image = test_images[sample_index]
            heatmap = anomaly_maps[sample_index]
            label_name = str(cluster_df.iloc[plot_idx]["label_name"])
            score = float(cluster_df.iloc[plot_idx]["score"])
            ax.imshow(image, cmap="gray")
            ax.imshow(heatmap, cmap="jet", alpha=0.45)
            ax.set_title(f"{label_name} idx={sample_index} s={score:.2f}")
            ax.axis("off")

        fig.suptitle(f"Cluster {cluster_id} Top Samples", fontsize=16)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"cluster_{int(cluster_id):02d}_montage.png"),
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cluster abnormal SAS PatchCore outputs with region or semantic features"
    )
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--patchcore-output-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--feature-mode",
        default="base",
        choices=["base", "mean", "mean_plus_gram"],
    )
    parser.add_argument("--threshold-percentile", type=float, default=90.0)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--gram-dim", type=int, default=32)
    parser.add_argument("--pca-dim", type=int, default=10)
    parser.add_argument("--min-cluster-size", type=int, default=20)
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--umap-neighbors", type=int, default=25)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--num-montage-images", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    # Pre-materialize the test spectrogram (no mmap): downstream code indexes
    # it repeatedly and random access on a 100MB+ mmap is slower than a full load.
    split = load_sas_test_split(args.dataset_dir)
    test_images = np.asarray(split.images)
    test_label_names = split.label_names
    anomaly_npz = np.load(os.path.join(args.patchcore_output_dir, "anomaly_outputs.npz"))
    score_df = pd.read_csv(os.path.join(args.patchcore_output_dir, "test_scores.csv"))

    scores = anomaly_npz["scores"].astype(np.float32)
    labels = anomaly_npz["labels"].astype(np.int64)
    anomaly_maps = anomaly_npz["anomaly_maps"].astype(np.float32)

    patch_lookup = None
    gram_projection = None
    if args.feature_mode != "base":
        patch_path = os.path.join(args.patchcore_output_dir, "test_patch_artifacts.npz")
        if not os.path.exists(patch_path):
            raise FileNotFoundError(
                f"Missing patch artifact export at {patch_path}. Re-run PatchCore with "
                "--export-test-patch-artifacts."
            )
        patch_npz = np.load(patch_path)
        patch_lookup = build_patch_lookup(patch_npz)
        embedding_dim = int(patch_npz["embedding_dim"][0])
        if args.feature_mode == "mean_plus_gram":
            gram_projection = build_random_projection(
                input_dim=embedding_dim,
                output_dim=args.gram_dim,
                seed=args.seed,
            )

    if not (
        test_images.shape[0]
        == test_label_names.shape[0]
        == scores.shape[0]
        == labels.shape[0]
        == anomaly_maps.shape[0]
        == score_df.shape[0]
    ):
        raise ValueError("Dataset and PatchCore output sizes do not match")

    abnormal_mask = test_label_names != "normal"
    abnormal_indices = np.where(abnormal_mask)[0]
    log(
        f"Preparing {args.feature_mode} features for {abnormal_indices.shape[0]} abnormal samples"
    )

    sample_features: list[SampleFeature] = []
    for sample_index in abnormal_indices:
        patch_entry = None if patch_lookup is None else patch_lookup.get(int(sample_index))
        feature_vector, props = extract_feature_vector(
            raw_image=test_images[sample_index],
            anomaly_map=anomaly_maps[sample_index],
            threshold_percentile=args.threshold_percentile,
            patch_size=args.patch_size,
            patch_entry=patch_entry,
            feature_mode=args.feature_mode,
            gram_projection=gram_projection,
        )
        sample_features.append(
            SampleFeature(
                sample_index=int(sample_index),
                label_name=str(test_label_names[sample_index]),
                score=float(scores[sample_index]),
                prediction=int(score_df.iloc[sample_index]["prediction"]),
                feature_vector=feature_vector,
                region_props=props,
            )
        )

    feature_matrix = np.stack([row.feature_vector for row in sample_features], axis=0)
    scaled = StandardScaler().fit_transform(feature_matrix)

    pca_dim = int(min(args.pca_dim, scaled.shape[0], scaled.shape[1]))
    reducer = PCA(n_components=pca_dim, random_state=args.seed)
    reduced = reducer.fit_transform(scaled)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
    )
    cluster_labels = clusterer.fit_predict(reduced)

    umap_reducer = umap.UMAP(
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        random_state=args.seed,
    )
    projection = umap_reducer.fit_transform(reduced)

    num_clusters = len(set(cluster_labels.tolist()) - {-1})
    noise_points = int(np.sum(cluster_labels == -1))
    clustered_points = int(cluster_labels.shape[0] - noise_points)
    noise_ratio = float(noise_points / max(cluster_labels.shape[0], 1))

    valid_cluster_mask = cluster_labels != -1
    valid_cluster_labels = cluster_labels[valid_cluster_mask]
    valid_cluster_reduced = reduced[valid_cluster_mask]

    if len(set(valid_cluster_labels.tolist())) >= 2 and valid_cluster_reduced.shape[0] > 2:
        silhouette = float(silhouette_score(valid_cluster_reduced, valid_cluster_labels))
        davies_bouldin = float(
            davies_bouldin_score(valid_cluster_reduced, valid_cluster_labels)
        )
    else:
        silhouette = None
        davies_bouldin = None

    rows = []
    for sample, cluster_label, point in zip(sample_features, cluster_labels, projection):
        row = {
            "sample_index": sample.sample_index,
            "label_name": sample.label_name,
            "score": sample.score,
            "prediction": sample.prediction,
            "cluster": int(cluster_label),
            "umap_x": float(point[0]),
            "umap_y": float(point[1]),
        }
        for key, value in sample.region_props.items():
            row[key] = float(value) if isinstance(value, (np.floating, float)) else int(value)
        rows.append(row)

    result_df = pd.DataFrame(rows)
    result_df.to_csv(os.path.join(args.output_dir, "cluster_assignments.csv"), index=False)

    plot_umap_projection(
        result_df,
        os.path.join(args.output_dir, "umap_cluster_projection.png"),
    )
    plot_cluster_composition(
        result_df,
        os.path.join(args.output_dir, "cluster_composition_by_type.png"),
    )
    save_cluster_montages(
        result_df=result_df,
        test_images=test_images,
        anomaly_maps=anomaly_maps,
        output_dir=args.output_dir,
        num_images=args.num_montage_images,
    )

    summary = {
        "feature_mode": args.feature_mode,
        "total_abnormal_samples": int(len(sample_features)),
        "feature_dim": int(feature_matrix.shape[1]),
        "pca_dim": int(pca_dim),
        "explained_variance_ratio_sum": float(np.sum(reducer.explained_variance_ratio_)),
        "num_clusters": int(num_clusters),
        "noise_points": int(noise_points),
        "clustered_points": int(clustered_points),
        "noise_ratio": noise_ratio,
        "silhouette": silhouette,
        "davies_bouldin": davies_bouldin,
        "clusters": [],
    }

    for cluster_id in sorted(set(cluster_labels.tolist())):
        mask = cluster_labels == cluster_id
        label_counts = result_df.loc[mask, "label_name"].value_counts().sort_index().to_dict()
        summary["clusters"].append(
            {
                "cluster": int(cluster_id),
                "count": int(np.sum(mask)),
                "label_counts": {str(k): int(v) for k, v in label_counts.items()},
                "mean_score": float(result_df.loc[mask, "score"].mean()),
                "std_score": float(result_df.loc[mask, "score"].std(ddof=0)),
            }
        )

    with open(os.path.join(args.output_dir, "cluster_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.output_dir, "clustering_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log(
        "Clustering complete: "
        f"clusters={num_clusters}, noise_ratio={noise_ratio:.4f}, "
        f"silhouette={silhouette}, davies_bouldin={davies_bouldin}"
    )


if __name__ == "__main__":
    main()
