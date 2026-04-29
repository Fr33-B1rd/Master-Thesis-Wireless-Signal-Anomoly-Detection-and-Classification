"""Smoke test for resize_mask + CHIRP polyline fix in build_rf_dataset.py.

Regenerates masks from existing per-sample JSON metadata in rf_dataset_full_*
and compares against the previously packed test_mask.npy. No IQ regeneration.
"""
import json, os, sys, numpy as np, pandas as pd
from scipy.ndimage import label as cc_label

sys.path.insert(0, r"C:/Users/Zhuoer/code/thesis/Datasets/SAS")
from build_rf_dataset import build_anomaly_mask

FULL = r"C:/Users/Zhuoer/code/thesis/Datasets/SAS/rf_dataset_full_impulsive_only_snr14_20_sirm4_0"
PACKED = r"C:/Users/Zhuoer/code/thesis/Datasets/SAS/rf_dataset_packed_impulsive_only_snr14_20_sirm4_0"
old_masks = np.load(os.path.join(PACKED, "test_mask.npy"), mmap_mode="r")
mani = pd.read_csv(os.path.join(PACKED, "test_manifest.csv"))


def regen_one(i):
    row = mani.iloc[i]
    rel = row["metadata_source"].replace("\\", "/")
    with open(os.path.join(FULL, rel), "r") as f:
        meta = json.load(f)
    sr = float(meta["sample_rate_hz"])
    nfft = int(meta["stft_nfft"])
    sz = int(meta["spectrogram_size"])
    rot = meta.get("spectrogram_rotation", "none") == "ccw_90"
    return build_anomaly_mask(meta, sr, nfft, sz, rot), meta


def eval_class(class_name, n=50):
    idx = mani.index[mani["label"] == class_name].tolist()[:n]
    new_pos, old_pos, new_cc, old_cc = [], [], [], []
    new_zero = old_zero = 0
    for i in idx:
        new, _ = regen_one(i)
        old = np.asarray(old_masks[i])
        np_pos, op = int(new.sum()), int(old.sum())
        new_pos.append(np_pos)
        old_pos.append(op)
        if np_pos == 0:
            new_zero += 1
        if op == 0:
            old_zero += 1
        if np_pos > 0:
            _, nc = cc_label(new, structure=np.ones((3, 3)))
            new_cc.append(nc)
        if op > 0:
            _, oc = cc_label(old, structure=np.ones((3, 3)))
            old_cc.append(oc)
    print(f"{class_name:<8} n={len(idx)}")
    print(
        f"  OLD: zero={old_zero:>3}/{n} pos_med={int(np.median(old_pos)):>6} cc_med={int(np.median(old_cc)) if old_cc else 0:>3}"
    )
    print(
        f"  NEW: zero={new_zero:>3}/{n} pos_med={int(np.median(new_pos)):>6} cc_med={int(np.median(new_cc)) if new_cc else 0:>3}"
    )


for c in ["tone", "chirp", "comb", "fsk", "ofdm", "pulse", "normal"]:
    eval_class(c, n=50)

print()
print("--- Targeted: previously zero-mask tone indices 5205-5209 ---")
for i in [5205, 5206, 5207, 5208, 5209]:
    new, meta = regen_one(i)
    old = np.asarray(old_masks[i])
    anom = meta.get("anomalies", [{}])[0]
    lbl = mani.iloc[i]["label"]
    print(
        f"  i={i} label={lbl:<6} type={anom.get('type')!s:<6} center_hz={anom.get('center_offset_hz', 0):>12.0f} "
        f"bw_hz={anom.get('occupied_bandwidth_hz', 0):>12.0f} old_sum={int(old.sum()):>4} new_sum={int(new.sum()):>4}"
    )
