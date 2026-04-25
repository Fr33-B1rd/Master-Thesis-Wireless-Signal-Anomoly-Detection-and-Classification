"""Feature extraction from PatchCore exports for SAS experiments."""

import os

import numpy as np
from dataio.sas import load_sas_test_split
from sas_open_world.common import log


def build_random_projection(input_dim: int, output_dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    projection = rng.standard_normal((input_dim, output_dim), dtype=np.float32)
    projection /= np.sqrt(float(input_dim))
    return projection.astype(np.float32, copy=False)


def weighted_mean_std(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    mean = float(np.average(values, weights=weights))
    variance = float(np.average((values - mean) ** 2, weights=weights))
    return mean, float(np.sqrt(max(variance, 0.0)))


def build_semantic_feature(
    patch_entry: dict,
    feature_mode: str,
    top_k_patches: int,
    gram_projection: np.ndarray | None,
) -> np.ndarray:
    """Top-K weighted aggregation of PatchCore patch embeddings into a single
    sample-level feature vector.

    Modes:
    - ``mean``: L2-normalized score-weighted mean of selected patch embeddings
    - ``gram``: upper-triangle of the score-weighted Gram matrix of the
      embeddings projected through ``gram_projection``
    - ``mean_plus_gram``: concatenation of the two

    Used by the fewshot classifier and by the open-world clustering feature
    cache (``build_feature_cache`` below).
    """
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


def build_handcrafted_patch_stats(
    raw_image: np.ndarray,
    patch_score_map: np.ndarray,
    top_k_patches: int,
    include_temporal_diff: bool,
    include_dynamic_range: bool,
    include_entropy: bool,
    include_frequency_profile: bool,
) -> np.ndarray:
    flat_scores = patch_score_map.reshape(-1).astype(np.float32, copy=False)
    top_k = min(int(top_k_patches), int(flat_scores.shape[0]))
    selected_idx = np.argpartition(flat_scores, -top_k)[-top_k:]
    selected_scores = flat_scores[selected_idx]
    selected_scores = selected_scores - float(selected_scores.min())
    if float(selected_scores.max()) > 0:
        selected_scores = selected_scores / float(selected_scores.max())
    weights = selected_scores + 1e-6

    grid_h, grid_w = patch_score_map.shape
    image_h, image_w = raw_image.shape

    temporal_rows = []
    dynamic_rows = []
    entropy_values = []
    profile_rows = []
    for flat_idx in selected_idx.tolist():
        row = int(flat_idx // grid_w)
        col = int(flat_idx % grid_w)
        y0 = int(round(row * image_h / grid_h))
        y1 = int(round((row + 1) * image_h / grid_h))
        x0 = int(round(col * image_w / grid_w))
        x1 = int(round((col + 1) * image_w / grid_w))
        patch = raw_image[y0:y1, x0:x1].astype(np.float32, copy=False)
        if patch.size == 0:
            patch = raw_image[max(y0 - 1, 0) : max(y0, 1), max(x0 - 1, 0) : max(x0, 1)]

        if include_temporal_diff:
            if patch.shape[1] > 1:
                diff = np.diff(patch, axis=1)
                abs_diff = np.abs(diff)
            else:
                abs_diff = np.zeros((patch.shape[0], 1), dtype=np.float32)
            temporal_rows.append(
                [
                    float(abs_diff.mean()),
                    float(abs_diff.std()),
                    float(abs_diff.max()),
                    float(np.percentile(abs_diff, 90)),
                    float(abs_diff.sum() / max(abs_diff.size, 1)),
                ]
            )

        if include_dynamic_range:
            patch_flat = patch.reshape(-1)
            dynamic_rows.append(
                [
                    float(patch_flat.max() - patch_flat.min()),
                    float(patch_flat.std()),
                    float(np.percentile(patch_flat, 95) - np.percentile(patch_flat, 5)),
                    float(np.mean(np.abs(patch_flat - patch_flat.mean()))),
                ]
            )

        if include_entropy:
            patch_flat = patch.reshape(-1)
            patch_min = float(patch_flat.min())
            patch_max = float(patch_flat.max())
            if patch_max <= patch_min:
                entropy_values.append(0.0)
            else:
                hist, _ = np.histogram(patch_flat, bins=16, range=(patch_min, patch_max))
                prob = hist.astype(np.float64)
                prob = prob / max(float(prob.sum()), 1.0)
                prob = prob[prob > 0]
                entropy = float(-(prob * np.log2(prob)).sum() / np.log2(16.0))
                entropy_values.append(entropy)

        if include_frequency_profile:
            profile = patch.mean(axis=1).astype(np.float32, copy=False)
            reduced_profile = np.asarray(
                [float(chunk.mean()) for chunk in np.array_split(profile, 8)],
                dtype=np.float32,
            )
            profile_rows.append(reduced_profile)

    feature_parts: list[float] = []
    if include_temporal_diff:
        temporal_array = np.asarray(temporal_rows, dtype=np.float32)
        for col_idx in range(temporal_array.shape[1]):
            mean_value, std_value = weighted_mean_std(temporal_array[:, col_idx], weights)
            feature_parts.extend([mean_value, std_value])
    if include_dynamic_range:
        dynamic_array = np.asarray(dynamic_rows, dtype=np.float32)
        for col_idx in range(dynamic_array.shape[1]):
            mean_value, std_value = weighted_mean_std(dynamic_array[:, col_idx], weights)
            feature_parts.extend([mean_value, std_value])
    if include_entropy:
        entropy_array = np.asarray(entropy_values, dtype=np.float32)
        mean_value, std_value = weighted_mean_std(entropy_array, weights)
        feature_parts.extend(
            [
                mean_value,
                std_value,
                float(entropy_array.max()),
                float(np.percentile(entropy_array, 90)),
            ]
        )
    if include_frequency_profile:
        profile_array = np.asarray(profile_rows, dtype=np.float32)
        for col_idx in range(profile_array.shape[1]):
            mean_value, std_value = weighted_mean_std(profile_array[:, col_idx], weights)
            feature_parts.extend([mean_value, std_value])

    return np.asarray(feature_parts, dtype=np.float32)


def build_feature_cache(
    dataset_dir: str,
    patchcore_output_dir: str,
    feature_mode: str,
    top_k_patches: int,
    gram_dim: int,
    seed: int,
    cache_path: str,
    augment_temporal_diff: bool,
    augment_dynamic_range: bool,
    augment_entropy: bool,
    augment_frequency_profile: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if os.path.exists(cache_path):
        cache = np.load(cache_path)
        return cache["X"].astype(np.float32), cache["label_names"].astype(str)

    needs_raw = (
        augment_temporal_diff
        or augment_dynamic_range
        or augment_entropy
        or augment_frequency_profile
    )
    split = load_sas_test_split(dataset_dir, load_images=needs_raw)
    test_label_names = split.label_names
    raw_images = split.images  # None when needs_raw is False
    patch_npz = np.load(os.path.join(patchcore_output_dir, "test_patch_artifacts.npz"))
    sample_indices = patch_npz["sample_indices"].astype(np.int64)
    patch_scores = patch_npz["patch_scores"]
    patch_embeddings = patch_npz["patch_embeddings"]
    embedding_dim = int(patch_npz["embedding_dim"][0])

    if sample_indices.shape[0] != test_label_names.shape[0]:
        raise ValueError("Expected export_selection=all patch artifacts to cover the full test set")

    gram_projection = None
    if feature_mode in {"gram", "mean_plus_gram"}:
        gram_projection = build_random_projection(
            input_dim=embedding_dim,
            output_dim=gram_dim,
            seed=seed,
        )

    log(f"Building {feature_mode} features for {sample_indices.shape[0]} test samples")
    features = []
    for i in range(sample_indices.shape[0]):
        base_feature = build_semantic_feature(
            patch_entry={
                "patch_scores": patch_scores[i],
                "patch_embeddings": patch_embeddings[i],
            },
            feature_mode=feature_mode,
            top_k_patches=top_k_patches,
            gram_projection=gram_projection,
        )
        if augment_temporal_diff or augment_dynamic_range or augment_frequency_profile:
            handcrafted = build_handcrafted_patch_stats(
                raw_image=np.asarray(raw_images[int(sample_indices[i])]).astype(np.float32, copy=False),
                patch_score_map=patch_scores[i].astype(np.float32, copy=False),
                top_k_patches=top_k_patches,
                include_temporal_diff=augment_temporal_diff,
                include_dynamic_range=augment_dynamic_range,
                include_entropy=augment_entropy,
                include_frequency_profile=augment_frequency_profile,
            )
            base_feature = np.concatenate([base_feature, handcrafted], axis=0)
        elif augment_entropy:
            handcrafted = build_handcrafted_patch_stats(
                raw_image=np.asarray(raw_images[int(sample_indices[i])]).astype(np.float32, copy=False),
                patch_score_map=patch_scores[i].astype(np.float32, copy=False),
                top_k_patches=top_k_patches,
                include_temporal_diff=False,
                include_dynamic_range=False,
                include_entropy=True,
                include_frequency_profile=False,
            )
            base_feature = np.concatenate([base_feature, handcrafted], axis=0)
        features.append(base_feature)
    x_all = np.stack(features).astype(np.float32)
    np.savez_compressed(cache_path, X=x_all, label_names=test_label_names.astype("U32"))
    return x_all, test_label_names
