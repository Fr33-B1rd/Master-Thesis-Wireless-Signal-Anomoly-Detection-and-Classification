"""Inference-only PatchCore evaluation on the multi-anomaly test set.

Loads the pre-trained snr14_20 memory bank, scores all 600 multi-anomaly
spectrograms, and writes:
  - scores.csv   : per-sample score + anomaly metadata
  - metrics.json : detection rate, score stats, per-combo breakdown
  - heatmaps/    : montage PNGs for sampled examples
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

# ── project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from train_eval_patchcore_wsad import WSADPatchCore, log


# ── dataset ────────────────────────────────────────────────────────────────────
class SpectrogramDataset(Dataset):
    def __init__(self, specs: np.ndarray, mean: float, std: float, clip_z: float = 5.0):
        x = (specs.astype(np.float32) - mean) / std
        x = np.clip(x, -clip_z, clip_z)
        # replicate to 3-channel, add channel dim → [N, 3, H, W]
        x = np.stack([x, x, x], axis=1)
        self.data = torch.from_numpy(x)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], torch.tensor(1, dtype=torch.long), torch.tensor(idx, dtype=torch.long)


# ── helpers ────────────────────────────────────────────────────────────────────
def load_anomaly_combos(meta_dir: Path, n: int):
    """Returns list of (type_set_str, count, overlap_mode) per sample index."""
    combos = []
    for i in range(n):
        p = meta_dir / f"multi_anomaly_{i:05d}.json"
        if not p.exists():
            combos.append(("?", 0, "?"))
            continue
        d = json.loads(p.read_text())
        types = sorted(set(a["type"] for a in d.get("anomalies", [])))
        pat = d.get("multi_anomaly_pattern", {})
        combos.append(("+".join(types), len(d.get("anomalies", [])), pat.get("overlap_mode", "?")))
    return combos


def save_heatmap_grid(specs, maps, scores, indices, combo_labels, output_path, threshold, cols=5):
    rows = (len(indices) + cols - 1) // cols
    fig, axes = plt.subplots(rows * 2, cols, figsize=(cols * 3, rows * 4))
    axes = np.array(axes).reshape(rows * 2, cols)
    for j, idx in enumerate(indices):
        r, c = divmod(j, cols)
        # spectrogram
        ax_s = axes[r * 2, c]
        ax_s.imshow(specs[idx], cmap="viridis", aspect="auto", origin="lower")
        ax_s.set_title(f"#{idx}\n{combo_labels[idx]}", fontsize=7)
        ax_s.axis("off")
        # heatmap
        ax_h = axes[r * 2 + 1, c]
        im = ax_h.imshow(maps[idx], cmap="jet", aspect="auto", origin="lower")
        flag = "ANOM" if scores[idx] >= threshold else "ok"
        ax_h.set_title(f"score={scores[idx]:.1f} [{flag}]", fontsize=7)
        ax_h.axis("off")
    # blank unused cells
    for j in range(len(indices), rows * cols):
        r, c = divmod(j, cols)
        axes[r * 2, c].axis("off")
        axes[r * 2 + 1, c].axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved heatmap grid: {output_path}")


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    PATCHCORE_DIR = Path(
        "C:/Users/Zhuoer/code/thesis/Anomaly_Localization/outputs/"
        "sas_blob_impulsive_snr14_20_patchcore_wide_resnet50_2_256_zscore_allpatches"
    )
    DATASET_DIR = Path(
        "C:/Users/Zhuoer/code/thesis/Datasets/SAS/rf_dataset_packed_multi_anomaly_test_600"
    )
    FULL_DATASET_DIR = Path(
        "C:/Users/Zhuoer/code/thesis/Datasets/SAS/rf_dataset_multi_anomaly_test_600"
    )
    OUTPUT_DIR = Path(
        "C:/Users/Zhuoer/code/thesis/Anomaly_Localization/outputs/"
        "sas_multi_anomaly_600_patchcore_eval"
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    # ── load trained PatchCore ──────────────────────────────────────────────────
    ckpt = torch.load(PATCHCORE_DIR / "patchcore_memory_bank.pt", map_location=device)
    pc_meta = json.loads((PATCHCORE_DIR / "metrics.json").read_text())

    train_mean = pc_meta["train_mean"]
    train_std  = pc_meta["train_std"]
    threshold  = pc_meta["threshold"]

    model = WSADPatchCore(
        backbone=pc_meta["backbone"],
        layers=ckpt["layers"],
        device=device,
        sample_patches_per_image=32,
        candidate_pool_size=20000,
        coreset_ratio=0.01,
        max_memory_bank_size=2048,
        projection_dim=64,
        query_chunk_size=2048,
        seed=42,
        pretrained=True,
    )
    model.memory_bank = ckpt["memory_bank"].to(device).float()
    model.memory_bank_norms = (model.memory_bank ** 2).sum(dim=1)
    model.embedding_dim = int(ckpt["embedding_dim"])
    model.patch_grid_size = tuple(ckpt["patch_grid_size"])
    log(f"Loaded memory bank: {model.memory_bank.shape[0]} vectors, "
        f"threshold={threshold:.2f}, mean={train_mean:.3f}, std={train_std:.3f}")

    # ── load spectrograms ──────────────────────────────────────────────────────
    specs_raw = np.load(DATASET_DIR / "test_spectrogram.npy", mmap_mode="r")
    log(f"Loaded spectrograms: {specs_raw.shape}")

    dataset = SpectrogramDataset(np.asarray(specs_raw), train_mean, train_std)
    loader  = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    # ── run inference ──────────────────────────────────────────────────────────
    scores, _, indices, anomaly_maps = model.predict(loader)
    sort_order = np.argsort(indices)
    scores      = scores[sort_order]
    anomaly_maps = anomaly_maps[sort_order]
    log(f"Inference done. score range: [{scores.min():.2f}, {scores.max():.2f}]")

    # ── load anomaly metadata ──────────────────────────────────────────────────
    meta_dir = FULL_DATASET_DIR / "test" / "multi_anomaly"
    n = len(specs_raw)
    combos = load_anomaly_combos(meta_dir, n)

    # ── per-sample results ─────────────────────────────────────────────────────
    import csv
    with open(OUTPUT_DIR / "scores.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "score", "detected", "anomaly_types", "anomaly_count", "overlap_mode"])
        for i in range(n):
            types_str, cnt, overlap = combos[i]
            w.writerow([i, f"{scores[i]:.4f}", int(scores[i] >= threshold), types_str, cnt, overlap])

    # ── aggregate stats ────────────────────────────────────────────────────────
    detected = scores >= threshold
    detection_rate = float(detected.mean())
    log(f"Overall detection rate: {detection_rate:.4f} ({detected.sum()}/{n})")

    # by anomaly count
    by_count = {}
    for anom_n in [2, 3]:
        mask = np.array([c[1] == anom_n for c in combos])
        if mask.any():
            by_count[str(anom_n)] = {
                "n": int(mask.sum()),
                "detection_rate": float(detected[mask].mean()),
                "score_mean": float(scores[mask].mean()),
                "score_std":  float(scores[mask].std()),
            }

    # by overlap mode
    by_overlap = {}
    for mode in ["non_overlap", "partial_overlap"]:
        mask = np.array([c[2] == mode for c in combos])
        if mask.any():
            by_overlap[mode] = {
                "n": int(mask.sum()),
                "detection_rate": float(detected[mask].mean()),
                "score_mean": float(scores[mask].mean()),
                "score_std":  float(scores[mask].std()),
            }

    # by type_group (same / different)
    by_type_group: dict[str, dict] = {}
    try:
        for p in meta_dir.glob("*.json"):
            d = json.loads(p.read_text())
            tg = d.get("multi_anomaly_pattern", {}).get("type_group", "?")
            if tg not in by_type_group:
                by_type_group[tg] = {"indices": []}
            idx = int(p.stem.split("_")[-1])
            by_type_group[tg]["indices"].append(idx)
        for tg, info in by_type_group.items():
            idxs = np.array(info["indices"])
            by_type_group[tg] = {
                "n": len(idxs),
                "detection_rate": float(detected[idxs].mean()),
                "score_mean": float(scores[idxs].mean()),
                "score_std":  float(scores[idxs].std()),
            }
    except Exception as e:
        log(f"type_group breakdown skipped: {e}")

    # by individual anomaly type presence
    by_anom_type: dict[str, dict] = {}
    for atype in ["CHIRP", "COMB", "FSK", "OFDM", "PULSE", "TONE"]:
        mask = np.array([atype in c[0] for c in combos])
        if mask.any():
            by_anom_type[atype] = {
                "n": int(mask.sum()),
                "detection_rate": float(detected[mask].mean()),
                "score_mean": float(scores[mask].mean()),
                "score_std":  float(scores[mask].std()),
            }

    metrics = {
        "patchcore_dir": str(PATCHCORE_DIR),
        "threshold": threshold,
        "n_samples": n,
        "overall_detection_rate": detection_rate,
        "score_mean": float(scores.mean()),
        "score_std":  float(scores.std()),
        "score_min":  float(scores.min()),
        "score_max":  float(scores.max()),
        "by_anomaly_count": by_count,
        "by_overlap_mode":  by_overlap,
        "by_type_group":    by_type_group,
        "by_anomaly_type_presence": by_anom_type,
    }
    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    log("Saved metrics.json")

    # ── heatmap grids ──────────────────────────────────────────────────────────
    specs_np = np.asarray(specs_raw)
    combo_labels = [c[0] for c in combos]
    heatmap_dir = OUTPUT_DIR / "heatmaps"
    heatmap_dir.mkdir(exist_ok=True)

    rng = np.random.default_rng(42)

    # 1. random sample of 20 (any)
    sel = rng.choice(n, size=min(20, n), replace=False)
    save_heatmap_grid(specs_np, anomaly_maps, scores, sel, combo_labels,
                      heatmap_dir / "random_20.png", threshold)

    # 2. partial_overlap only
    partial_idx = np.where(np.array([c[2] == "partial_overlap" for c in combos]))[0]
    if len(partial_idx) >= 10:
        sel = rng.choice(partial_idx, size=10, replace=False)
        save_heatmap_grid(specs_np, anomaly_maps, scores, sel, combo_labels,
                          heatmap_dir / "partial_overlap_10.png", threshold)

    # 3. same-type (two/three identical anomalies)
    same_idx = np.where(np.array([c[0].count("+") == 0 and c[1] >= 2 for c in combos]))[0]
    if len(same_idx) >= 10:
        sel = rng.choice(same_idx, size=10, replace=False)
        save_heatmap_grid(specs_np, anomaly_maps, scores, sel, combo_labels,
                          heatmap_dir / "same_type_10.png", threshold)

    # 4. missed detections (score < threshold)
    missed = np.where(~detected)[0]
    log(f"Missed detections: {len(missed)}")
    if len(missed) > 0:
        sel = missed[:min(10, len(missed))]
        save_heatmap_grid(specs_np, anomaly_maps, scores, sel, combo_labels,
                          heatmap_dir / "missed_detections.png", threshold)

    log("Done.")


if __name__ == "__main__":
    main()
