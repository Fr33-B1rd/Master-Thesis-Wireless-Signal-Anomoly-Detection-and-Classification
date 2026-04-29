#!/usr/bin/env python3
"""Regenerate GT anomaly masks in-place from existing per-sample JSON metadata.

Applies the fixed build_anomaly_mask (max-pool downsampling + CHIRP polyline)
to every sample in rf_dataset_full_<tag>/{train,test}/<label>/, overwrites the
corresponding *_mask.npy files, and then re-packs only the split_mask.npy
arrays in rf_dataset_packed_<tag>/ (IQ/spec/labels are left untouched so
downstream caches like PatchCore outputs stay aligned).

Usage:
    python regenerate_masks.py \
        --full-dir   rf_dataset_full_impulsive_only_snr14_20_sirm4_0 \
        --packed-dir rf_dataset_packed_impulsive_only_snr14_20_sirm4_0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

sys.path.insert(0, str(Path(__file__).parent))
from build_rf_dataset import build_anomaly_mask  # noqa: E402


def collect_samples(split_dir: Path) -> list[tuple[str, str, Path, Path]]:
    labels = sorted(p.name for p in split_dir.iterdir() if p.is_dir())
    samples: list[tuple[str, str, Path, Path]] = []
    for label in labels:
        label_dir = split_dir / label
        iq_files = sorted(
            p for p in label_dir.glob("*.npy") if not p.name.endswith("_mask.npy")
        )
        for iq_path in iq_files:
            stem = iq_path.stem
            meta_path = label_dir / f"{stem}.json"
            mask_path = label_dir / f"{stem}_mask.npy"
            if not meta_path.exists():
                raise FileNotFoundError(meta_path)
            samples.append((label, stem, meta_path, mask_path))
    return samples


def regen_split(full_dir: Path, packed_dir: Path, split: str) -> None:
    split_dir = full_dir / split
    if not split_dir.exists():
        print(f"[{split}] skip: {split_dir} not found")
        return

    samples = collect_samples(split_dir)
    if not samples:
        print(f"[{split}] skip: no samples")
        return

    print(f"[{split}] regenerating {len(samples)} per-sample masks in full/ tree...")
    for i, (_label, _stem, meta_path, mask_path) in enumerate(samples):
        with meta_path.open("r") as f:
            meta = json.load(f)
        mask = build_anomaly_mask(
            metadata=meta,
            sample_rate=float(meta["sample_rate_hz"]),
            nfft=int(meta["stft_nfft"]),
            image_size=int(meta["spectrogram_size"]),
            rotate_ccw_90=(meta.get("spectrogram_rotation", "none") == "ccw_90"),
        )
        np.save(mask_path, mask.astype(np.uint8))
        if (i + 1) % 500 == 0 or (i + 1) == len(samples):
            print(f"  [{split}] {i + 1}/{len(samples)}")

    packed_path = packed_dir / f"{split}_mask.npy"
    print(f"[{split}] repacking -> {packed_path}")
    first = np.load(samples[0][3], mmap_mode="r")
    arr = open_memmap(
        packed_path, mode="w+", dtype=np.uint8,
        shape=(len(samples),) + tuple(first.shape),
    )
    for i, (_l, _s, _m, mask_path) in enumerate(samples):
        arr[i] = np.load(mask_path)
    arr.flush()
    print(f"[{split}] done.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-dir", type=Path, required=True)
    ap.add_argument("--packed-dir", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    args = ap.parse_args()

    if not args.full_dir.exists():
        raise FileNotFoundError(args.full_dir)
    args.packed_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        regen_split(args.full_dir, args.packed_dir, split)

    print("all splits done.")


if __name__ == "__main__":
    main()
