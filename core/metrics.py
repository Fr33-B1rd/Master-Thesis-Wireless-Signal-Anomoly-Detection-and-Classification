# Pixel-AUROC and AUPRO for SAS anomaly localization.
#
# Inputs:
#   anomaly_maps: (N, H, W) continuous scores (float16/32/64)
#   gt_masks:     (N, H, W) binary GT (uint8/bool, 0/1)
#
# Outputs (dict):
#   - pixel_auroc:      global pixel-level ROC AUC across the full test set
#   - aupro:            MVTec-style per-region overlap AUC, integrated over
#                       FPR in [0, integration_limit] and normalized by that limit
#   - per_class: dict[label -> {pixel_auroc, aupro, num_regions}]
#   - integration_limit, num_regions, num_threshold_points
#
# Implementation notes:
#   - AUROC uses histogram-based streaming computation (no 400M-element sort)
#     so the full 6000*256*256 pixel space fits comfortably in <2 GB.
#   - AUPRO extracts per-region pixels once (4-connectivity by default),
#     then at each threshold does bincount(region_id[above_threshold]).
#     FPR(t) is computed via searchsorted on a sorted negative-pixel array.
#   - Threshold grid is quantile-spaced over the full score distribution,
#     which densely samples the high-score operating region.

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.ndimage import label as cc_label


_STRUCTURE = np.ones((3, 3), dtype=np.int32)  # 8-connectivity


def _pixel_auroc_histogram(scores: np.ndarray, labels: np.ndarray, n_bins: int = 65536) -> float:
    """Histogram-based pixel-AUROC. O(N) memory, O(N + n_bins) time."""
    if labels.sum() == 0 or (~labels).sum() == 0:
        return float("nan")
    smin = float(scores.min())
    smax = float(scores.max())
    if smax <= smin:
        return 0.5
    edges = np.linspace(smin, smax, n_bins + 1, dtype=np.float64)
    idx = np.clip(
        np.searchsorted(edges, scores, side="right") - 1, 0, n_bins - 1
    ).astype(np.int64)
    pos_hist = np.bincount(idx[labels], minlength=n_bins).astype(np.int64)
    neg_hist = np.bincount(idx[~labels], minlength=n_bins).astype(np.int64)
    total_pos = pos_hist.sum()
    total_neg = neg_hist.sum()
    # tpr/fpr above bin-start (score > edge[k])
    pos_above = np.concatenate([[0], np.cumsum(pos_hist[::-1])])[::-1]
    neg_above = np.concatenate([[0], np.cumsum(neg_hist[::-1])])[::-1]
    tpr = pos_above / max(total_pos, 1)
    fpr = neg_above / max(total_neg, 1)
    # sort by fpr ascending for trapz
    order = np.argsort(fpr, kind="stable")
    return float(np.trapz(tpr[order], fpr[order]))


def _extract_regions(
    gt_masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Return (region_id_of_pixel, image_of_pixel, flat_pixel_index, region_sizes, num_regions).

    region_id_of_pixel[k]   = global region id of the k-th positive pixel
    image_of_pixel[k]       = image index of the k-th positive pixel
    flat_pixel_index[k]     = row-major (H*W) index within that image
    region_sizes[r]         = #pixels in region r
    num_regions             = total distinct regions across the test set

    Pixels are grouped by (image, region); the companion arrays let us
    directly gather from anomaly_maps via `flat_scores[image_of_pixel, flat_pixel_index]`.
    """
    n = gt_masks.shape[0]
    region_sizes: list[int] = []
    image_of_pixel: list[np.ndarray] = []
    region_id_of_pixel: list[np.ndarray] = []
    flat_pixel_index: list[np.ndarray] = []
    rid = 0
    for i in range(n):
        mi = gt_masks[i].astype(bool)
        if not mi.any():
            continue
        labeled, ncc = cc_label(mi, structure=_STRUCTURE)
        for r in range(1, ncc + 1):
            pix_mask = labeled == r
            size = int(pix_mask.sum())
            if size == 0:
                continue
            region_sizes.append(size)
            flat_pixel_index.append(np.flatnonzero(pix_mask.ravel()))
            image_of_pixel.append(np.full(size, i, dtype=np.int64))
            region_id_of_pixel.append(np.full(size, rid, dtype=np.int32))
            rid += 1
    if rid == 0:
        return (
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            0,
        )
    return (
        np.concatenate(region_id_of_pixel),
        np.concatenate(image_of_pixel),
        np.concatenate(flat_pixel_index),
        np.asarray(region_sizes, dtype=np.int64),
        rid,
    )


def _compute_aupro_and_auroc(
    anomaly_maps: np.ndarray,
    gt_masks: np.ndarray,
    integration_limit: float,
    n_thresholds: int,
) -> dict:
    """Core metric computation. Returns dict with pixel_auroc, aupro, num_regions."""
    if anomaly_maps.shape != gt_masks.shape:
        raise ValueError(
            f"shape mismatch: anomaly_maps {anomaly_maps.shape} vs gt_masks {gt_masks.shape}"
        )

    n, h, w = anomaly_maps.shape
    flat_scores = np.ascontiguousarray(anomaly_maps).reshape(n, h * w).astype(np.float32, copy=False)
    flat_labels = np.ascontiguousarray(gt_masks).reshape(n, h * w).astype(bool, copy=False)

    # Pixel-AUROC
    pixel_auroc = _pixel_auroc_histogram(flat_scores.ravel(), flat_labels.ravel())

    # AUPRO — extract regions
    region_ids, image_idx, flat_pixel_idx, region_sizes, n_regions = _extract_regions(
        gt_masks
    )
    if n_regions == 0:
        return {
            "pixel_auroc": pixel_auroc,
            "aupro": float("nan"),
            "num_regions": 0,
            "integration_limit": integration_limit,
        }

    # Gather scores at region pixels once
    region_pixel_scores = flat_scores[image_idx, flat_pixel_idx]

    # Gather scores at all negative pixels, sort ascending for searchsorted
    neg_mask = ~flat_labels
    # memory: count true pixels first, allocate, fill
    neg_count = int(neg_mask.sum())
    if neg_count == 0:
        return {
            "pixel_auroc": pixel_auroc,
            "aupro": float("nan"),
            "num_regions": int(n_regions),
            "integration_limit": integration_limit,
        }
    neg_scores = flat_scores[neg_mask]  # float32 copy ~ 1.5 GB for 400M pixels
    neg_scores.sort()

    # Threshold grid: quantile-spaced over the full score distribution.
    # We lean on the high-score tail (where the operating region lives).
    qs = np.concatenate([
        np.linspace(0.0, 0.9, max(n_thresholds // 4, 8), endpoint=False),
        np.linspace(0.9, 1.0, n_thresholds - max(n_thresholds // 4, 8)),
    ])
    t_grid = np.quantile(flat_scores.ravel(), qs).astype(np.float32)
    t_grid = np.unique(t_grid)

    fprs = np.empty(len(t_grid), dtype=np.float64)
    pros = np.empty(len(t_grid), dtype=np.float64)
    inv_region_sizes = 1.0 / region_sizes.astype(np.float64)

    for ti, t in enumerate(t_grid):
        # FPR(t) = #negative pixels with score > t / total_neg
        pos_above_neg = neg_count - np.searchsorted(neg_scores, t, side="right")
        fprs[ti] = pos_above_neg / neg_count
        # PRO(t) = mean over regions of (#region pixels with score > t / region_size)
        above = region_pixel_scores > t
        per_region_hits = np.bincount(region_ids[above], minlength=n_regions)
        pros[ti] = float((per_region_hits * inv_region_sizes).mean())

    # Sort by FPR ascending and truncate at integration_limit
    order = np.argsort(fprs, kind="stable")
    fprs_s = fprs[order]
    pros_s = pros[order]

    keep = fprs_s <= integration_limit
    if keep.all():
        fpr_trunc = fprs_s
        pro_trunc = pros_s
    else:
        last = np.flatnonzero(keep)[-1] if keep.any() else -1
        if last + 1 < len(fprs_s) and last >= 0:
            f0, f1 = fprs_s[last], fprs_s[last + 1]
            p0, p1 = pros_s[last], pros_s[last + 1]
            p_at_limit = p1 if f1 == f0 else p0 + (p1 - p0) * (integration_limit - f0) / (f1 - f0)
            fpr_trunc = np.append(fprs_s[: last + 1], integration_limit)
            pro_trunc = np.append(pros_s[: last + 1], p_at_limit)
        elif last < 0:
            fpr_trunc = np.array([0.0, integration_limit])
            pro_trunc = np.array([pros_s[0], pros_s[0]])
        else:
            fpr_trunc = fprs_s
            pro_trunc = pros_s

    aupro = float(np.trapz(pro_trunc, fpr_trunc) / integration_limit)

    return {
        "pixel_auroc": pixel_auroc,
        "aupro": aupro,
        "num_regions": int(n_regions),
        "integration_limit": integration_limit,
        "num_threshold_points": int(len(t_grid)),
    }


def compute_localization_metrics(
    anomaly_maps: np.ndarray,
    gt_masks: np.ndarray,
    test_label_names: np.ndarray | None = None,
    integration_limit: float = 0.3,
    n_thresholds: int = 200,
    per_class: bool = True,
) -> dict:
    """Full localization metric bundle.

    per_class=True computes AUROC/AUPRO restricted to {normal pixels (fixed)
    ∪ pixels of that class}. This gives each class an independent read that
    is not dominated by the larger classes — required for honest reporting
    given 40-50x coverage imbalance between narrow-band (tone/chirp) and
    wide-band (ofdm/fsk/pulse) classes.
    """
    result = _compute_aupro_and_auroc(
        anomaly_maps, gt_masks, integration_limit, n_thresholds
    )

    if per_class and test_label_names is not None:
        names = np.asarray(test_label_names).astype(str)
        unique = sorted(np.unique(names).tolist())
        normal_idx = np.flatnonzero(names == "normal")
        per_class_out: dict[str, dict] = {}
        for lbl in unique:
            if lbl == "normal":
                continue
            cls_idx = np.flatnonzero(names == lbl)
            combined = np.concatenate([normal_idx, cls_idx])
            sub_maps = anomaly_maps[combined]
            sub_masks = gt_masks[combined]
            sub_result = _compute_aupro_and_auroc(
                sub_maps, sub_masks, integration_limit, n_thresholds
            )
            sub_result["num_samples"] = int(len(combined))
            sub_result["num_anomalous"] = int(len(cls_idx))
            per_class_out[lbl] = sub_result
        result["per_class"] = per_class_out

    return result
