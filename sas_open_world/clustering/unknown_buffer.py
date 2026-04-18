"""HDBSCAN utilities for unknown-buffer clustering with anchors."""

from collections import defaultdict

import hdbscan
import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler
import umap


def cluster_unknown_buffer(
    x_buffer: np.ndarray,
    true_labels: np.ndarray,
    buffer_sample_indices: np.ndarray,
    x_anchor: np.ndarray,
    anchor_labels: np.ndarray,
    anchor_sample_indices: np.ndarray,
    current_known_classes: list[str],
    emerging_classes: list[str],
    pca_dim: int,
    min_cluster_size: int,
    min_samples: int,
    cluster_accept_purity: float,
    cluster_accept_min_size: int,
    cluster_max_anchor_fraction: float,
    seed: int,
    dim_reduction: str = "pca",
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.0,
    cluster_selection_method: str = "eom",
    cluster_accept_min_prob: float = -1.0,
    cluster_anchor_null_alpha: float = -1.0,
    anchor_core_prob_threshold: float = 0.5,
    cluster_anchor_fraction_scale: float = -1.0,
    cluster_anchor_quantile_alpha: float = -1.0,
) -> tuple[pd.DataFrame, dict[str, list[int]], dict]:
    source_unknown = np.full(x_buffer.shape[0], "unknown", dtype="U16")
    source_anchor = np.full(x_anchor.shape[0], "anchor", dtype="U16")
    x_combined = np.concatenate([x_buffer, x_anchor], axis=0)
    labels_combined = np.concatenate([true_labels, anchor_labels], axis=0)
    source_combined = np.concatenate([source_unknown, source_anchor], axis=0)
    sample_indices_combined = np.concatenate([buffer_sample_indices, anchor_sample_indices], axis=0)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_combined)

    if dim_reduction == "umap":
        n_components = min(pca_dim, x_scaled.shape[1])
        if n_components < 2:
            projected = x_scaled
            explained_variance = -1.0
        else:
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=umap_n_neighbors,
                min_dist=umap_min_dist,
                metric="euclidean",
                random_state=seed,
            )
            projected = reducer.fit_transform(x_scaled)
            explained_variance = -1.0
    else:
        n_components = min(pca_dim, x_scaled.shape[1], x_scaled.shape[0] - 1)
        if n_components < 2:
            projected = x_scaled
            explained_variance = 1.0
        else:
            pca = PCA(n_components=n_components, random_state=seed)
            projected = pca.fit_transform(x_scaled)
            explained_variance = float(np.sum(pca.explained_variance_ratio_))

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method=cluster_selection_method,
    )
    cluster_labels = clusterer.fit_predict(projected)
    membership_probs = clusterer.probabilities_

    use_unsupervised = cluster_accept_min_prob >= 0.0
    use_hypergeom_null = cluster_anchor_null_alpha >= 0.0

    anchor_class_names = [str(n) for n in np.unique(anchor_labels).tolist()]
    anchor_class_totals = {
        name: int(np.sum(anchor_labels == name)) for name in anchor_class_names
    }
    n_total_combined = int(x_combined.shape[0])
    n_known_classes_for_bonf = max(len(anchor_class_names), 1)
    alpha_bonf = (
        cluster_anchor_null_alpha / n_known_classes_for_bonf
        if use_hypergeom_null
        else None
    )
    # Scale-free anchor_fraction threshold: expressed as a multiple of the
    # null expectation n_anchor_total / n_total. A leakage cluster has
    # anchor density ~10x the null; a legitimate emerging cluster is at most
    # ~2x (residual UMAP drift). Default scale=-1 -> use absolute threshold.
    use_fraction_scale = cluster_anchor_fraction_scale >= 0.0
    use_hypergeom_quantile = cluster_anchor_quantile_alpha >= 0.0
    null_anchor_fraction = (
        float(x_anchor.shape[0]) / float(max(n_total_combined, 1))
    )
    adaptive_anchor_fraction_threshold = (
        cluster_anchor_fraction_scale * null_anchor_fraction
        if use_fraction_scale
        else None
    )
    n_anchor_total = int(x_anchor.shape[0])

    rows = []
    accepted_indices_by_label: dict[str, list[int]] = defaultdict(list)
    cluster_summaries = []
    for cluster_id in sorted(np.unique(cluster_labels).tolist()):
        cluster_mask = cluster_labels == cluster_id
        cluster_labels_all = labels_combined[cluster_mask]
        indices = np.where(cluster_mask)[0].tolist()
        unknown_mask = cluster_mask & (source_combined == "unknown")
        anchor_mask = cluster_mask & (source_combined == "anchor")
        unknown_indices = np.where(unknown_mask)[0].tolist()
        anchor_count = int(anchor_mask.sum())
        unknown_count = len(unknown_indices)
        anchor_fraction = float(anchor_count / max(len(indices), 1))

        mean_unknown_prob = float(np.mean(membership_probs[unknown_mask])) if unknown_count > 0 else 0.0
        mean_anchor_prob = float(np.mean(membership_probs[anchor_mask])) if anchor_count > 0 else 0.0
        # If anchors are core members (high prob ~ unknowns), gap ~ 0 -> real leakage.
        # If anchors were UMAP-dragged to cluster periphery, gap > 0 -> likely false alarm.
        anchor_prob_gap = mean_unknown_prob - mean_anchor_prob if anchor_count > 0 else 0.0

        # Per-class hypergeometric test: for each known class k, under null
        # "anchors are randomly distributed across clusters", the count of
        # class-k anchors in this cluster ~ Hypergeom(N_total, K_k, n_cluster).
        # p_k = P(X >= observed) = sf(observed - 1, ...). Small p_k means the
        # cluster is significantly enriched in class-k anchors (=> leakage).
        cluster_size_int = int(cluster_mask.sum())
        # Exact per-cluster anchor_fraction threshold via hypergeometric quantile.
        # Under null (anchors randomly distributed), anchor count in a cluster of
        # size n ~ Hypergeom(N_total, K_a_total, n). Threshold = ppf(1-alpha, ...) / n.
        # Larger clusters get tighter thresholds (less variance around the null mean).
        if use_hypergeom_quantile and cluster_size_int > 0:
            quantile_threshold = float(
                hypergeom.ppf(
                    1.0 - cluster_anchor_quantile_alpha,
                    n_total_combined,
                    n_anchor_total,
                    cluster_size_int,
                )
            ) / cluster_size_int
        else:
            quantile_threshold = None

        per_class_anchor_counts: dict[str, int] = {}
        per_class_pvalues: dict[str, float] = {}
        min_p_class: str | None = None
        min_p_value: float = 1.0
        if cluster_id != -1 and cluster_size_int > 0:
            anchor_in_cluster_mask = anchor_mask  # alias
            for class_name in anchor_class_names:
                class_anchor_mask_full = (
                    anchor_in_cluster_mask & (labels_combined == class_name)
                )
                x_k = int(class_anchor_mask_full.sum())
                per_class_anchor_counts[class_name] = x_k
                K_k = anchor_class_totals[class_name]
                if K_k == 0 or cluster_size_int == 0:
                    p_k = 1.0
                else:
                    # sf(k-1) = P(X >= k). sf(-1) = 1.0 when x_k == 0.
                    p_k = float(
                        hypergeom.sf(
                            x_k - 1, n_total_combined, K_k, cluster_size_int
                        )
                    )
                per_class_pvalues[class_name] = p_k
                if p_k < min_p_value:
                    min_p_value = p_k
                    min_p_class = class_name

        if cluster_id == -1:
            majority_label = "noise"
            purity = 0.0
            unknown_purity = 0.0
        else:
            counts = pd.Series(cluster_labels_all).value_counts()
            majority_label = str(counts.index[0])
            purity = float(counts.iloc[0] / max(int(cluster_mask.sum()), 1))
            if unknown_count > 0:
                unknown_counts = pd.Series(labels_combined[unknown_mask]).value_counts()
                unknown_majority_label = str(unknown_counts.index[0])
                unknown_purity = float(unknown_counts.iloc[0] / max(unknown_count, 1))
            else:
                unknown_majority_label = "anchor_only"
                unknown_purity = 0.0

            if use_unsupervised:
                # Unsupervised acceptance: no oracle labels used in the decision.
                # Label assignment still uses ground-truth majority for evaluation.
                if use_hypergeom_null:
                    # Scale-free known-class-leakage test. Reject only if BOTH:
                    #   (i) anchor count for some known class is statistically
                    #       over-represented (hypergeom sf < alpha/M), AND
                    #   (ii) those anchors are core members of the cluster
                    #       (mean_anchor_prob >= core_threshold). If anchors
                    #       have low membership probability, they were pulled
                    #       to the cluster periphery by UMAP and the cluster
                    #       is still a legitimate emerging one.
                    anchors_are_core = (
                        anchor_count == 0
                        or mean_anchor_prob >= anchor_core_prob_threshold
                    )
                    is_known_class_leakage = (
                        min_p_value < alpha_bonf and anchors_are_core
                    )
                    anchor_check_passed = not is_known_class_leakage
                elif use_hypergeom_quantile:
                    # Exact per-cluster threshold: (1-alpha) quantile of
                    # Hypergeom(N, K_a_total, n) / n. Statistically equivalent
                    # to a one-sided permutation test on total anchor_fraction.
                    anchor_check_passed = (
                        anchor_fraction <= quantile_threshold
                    )
                elif use_fraction_scale:
                    # Scale-free approximation: scale x null_anchor_fraction.
                    anchor_check_passed = (
                        anchor_fraction <= adaptive_anchor_fraction_threshold
                    )
                else:
                    # Absolute-threshold fallback: anchor_fraction < threshold.
                    anchor_check_passed = (
                        anchor_fraction <= cluster_max_anchor_fraction
                    )
                accepted = (
                    unknown_count >= cluster_accept_min_size
                    and mean_unknown_prob >= cluster_accept_min_prob
                    and anchor_check_passed
                )
            else:
                # Legacy oracle acceptance (for reference / ablation).
                accepted = (
                    unknown_majority_label in emerging_classes
                    and unknown_majority_label not in current_known_classes
                    and unknown_count >= cluster_accept_min_size
                    and unknown_purity >= cluster_accept_purity
                    and anchor_fraction <= cluster_max_anchor_fraction
                )

            if accepted:
                collect_indices = np.where(unknown_mask)[0].tolist()
                accepted_indices_by_label[unknown_majority_label].extend(
                    [int(idx) for idx in collect_indices]
                )

        cluster_summaries.append(
            {
                "cluster": int(cluster_id),
                "size": int(cluster_mask.sum()),
                "majority_label": majority_label,
                "purity": purity,
                "unknown_count": unknown_count,
                "anchor_count": anchor_count,
                "anchor_fraction": anchor_fraction,
                "unknown_purity": unknown_purity,
                "mean_unknown_prob": mean_unknown_prob,
                "mean_anchor_prob": mean_anchor_prob,
                "anchor_prob_gap": anchor_prob_gap,
                "quantile_threshold": float(quantile_threshold) if quantile_threshold is not None else None,
                "per_class_anchor_counts": per_class_anchor_counts,
                "per_class_pvalues": per_class_pvalues,
                "min_p_value": float(min_p_value),
                "min_p_class": min_p_class,
                "accepted_for_retraining": bool(
                    cluster_id != -1
                    and any(
                        local_idx in accepted_indices_by_label.get(label_name, [])
                        for label_name in accepted_indices_by_label
                        for local_idx in unknown_indices
                    )
                ),
            }
        )

        for local_idx in indices:
            rows.append(
                {
                    "local_combined_index": int(local_idx),
                    "cluster": int(cluster_id),
                    "source": str(source_combined[local_idx]),
                    "sample_index": int(sample_indices_combined[local_idx]),
                    "true_label": str(labels_combined[local_idx]),
                }
            )

    assignment_df = pd.DataFrame(rows).sort_values(["cluster", "source", "sample_index"]).reset_index(drop=True)

    non_noise_mask = cluster_labels != -1
    n_valid_clusters = len(np.unique(cluster_labels[non_noise_mask]))
    if n_valid_clusters >= 2 and non_noise_mask.sum() >= 2:
        sil_score = float(silhouette_score(projected[non_noise_mask], cluster_labels[non_noise_mask]))
        dbi_score = float(davies_bouldin_score(projected[non_noise_mask], cluster_labels[non_noise_mask]))
    else:
        sil_score = None
        dbi_score = None

    summary = {
        "n_unknown_samples": int(x_buffer.shape[0]),
        "n_anchor_samples": int(x_anchor.shape[0]),
        "n_samples": int(x_combined.shape[0]),
        "n_clusters": int(len([cid for cid in np.unique(cluster_labels).tolist() if cid != -1])),
        "noise_count": int(np.sum(cluster_labels == -1)),
        "noise_ratio": float(np.mean(cluster_labels == -1)),
        "dim_reduction": dim_reduction,
        "dim_reduction_components": int(n_components),
        "pca_explained_variance": explained_variance,
        "hypergeom_null_enabled": bool(use_hypergeom_null),
        "hypergeom_null_alpha": float(cluster_anchor_null_alpha),
        "hypergeom_null_alpha_bonf": float(alpha_bonf) if alpha_bonf is not None else None,
        "anchor_core_prob_threshold": float(anchor_core_prob_threshold),
        "anchor_fraction_scale_enabled": bool(use_fraction_scale),
        "anchor_fraction_scale": float(cluster_anchor_fraction_scale),
        "null_anchor_fraction": float(null_anchor_fraction),
        "adaptive_anchor_fraction_threshold": (
            float(adaptive_anchor_fraction_threshold)
            if adaptive_anchor_fraction_threshold is not None else None
        ),
        "anchor_class_totals": anchor_class_totals,
        "silhouette_score": sil_score,
        "davies_bouldin_score": dbi_score,
        "cluster_summaries": cluster_summaries,
    }
    return assignment_df, accepted_indices_by_label, summary
