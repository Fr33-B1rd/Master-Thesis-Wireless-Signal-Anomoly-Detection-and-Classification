#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from numpy.lib.format import open_memmap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pack per-sample RF dataset files into split-level NumPy arrays."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("rf_dataset_full"),
        help="Directory containing the existing train/test dataset tree.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("rf_dataset_packed"),
        help="Directory to store packed train/test arrays.",
    )
    return parser.parse_args()


def split_labels(split_dir: Path) -> list[str]:
    return sorted([path.name for path in split_dir.iterdir() if path.is_dir()])


def gather_samples(split_dir: Path, labels: list[str]) -> list[tuple[str, Path, Path, Path, Path]]:
    samples: list[tuple[str, Path, Path, Path, Path]] = []
    for label in labels:
        label_dir = split_dir / label
        iq_files = sorted(path for path in label_dir.glob("*.npy") if not path.name.endswith("_mask.npy"))
        for iq_path in iq_files:
            stem = iq_path.stem
            spec_path = label_dir / f"{stem}.png"
            mask_path = label_dir / f"{stem}_mask.npy"
            meta_path = label_dir / f"{stem}.json"
            if not spec_path.exists() or not meta_path.exists() or not mask_path.exists():
                raise FileNotFoundError(f"Missing paired files for {iq_path}")
            samples.append((label, iq_path, spec_path, mask_path, meta_path))
    return samples


def pack_split(input_dir: Path, output_dir: Path, split: str) -> None:
    split_dir = input_dir / split
    if not split_dir.exists():
        return

    labels = split_labels(split_dir)
    samples = gather_samples(split_dir, labels)
    if not samples:
        return

    first_iq = np.load(samples[0][1], mmap_mode="r")
    first_spec = np.array(Image.open(samples[0][2]).convert("L"), dtype=np.uint8)
    first_mask = np.load(samples[0][3], mmap_mode="r")

    iq_array = open_memmap(
        output_dir / f"{split}_iq.npy",
        mode="w+",
        dtype=np.complex64,
        shape=(len(samples),) + first_iq.shape,
    )
    spec_array = open_memmap(
        output_dir / f"{split}_spectrogram.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(len(samples),) + first_spec.shape,
    )
    mask_array = open_memmap(
        output_dir / f"{split}_mask.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(len(samples),) + first_mask.shape,
    )
    label_array = open_memmap(
        output_dir / f"{split}_labels.npy",
        mode="w+",
        dtype="<U16",
        shape=(len(samples),),
    )

    rows: list[dict[str, str | int]] = []
    for index, (label, iq_path, spec_path, mask_path, meta_path) in enumerate(samples):
        iq_array[index] = np.load(iq_path)
        spec_array[index] = np.array(Image.open(spec_path).convert("L"), dtype=np.uint8)
        mask_array[index] = np.load(mask_path)
        label_array[index] = label
        rows.append(
            {
                "index": index,
                "label": label,
                "iq_source": str(iq_path.relative_to(input_dir)),
                "spectrogram_source": str(spec_path.relative_to(input_dir)),
                "mask_source": str(mask_path.relative_to(input_dir)),
                "metadata_source": str(meta_path.relative_to(input_dir)),
            }
        )

    iq_array.flush()
    spec_array.flush()
    mask_array.flush()
    label_array.flush()

    manifest_path = output_dir / f"{split}_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["index", "label", "iq_source", "spectrogram_source", "mask_source", "metadata_source"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pack_split(args.input_dir, args.output_dir, "train")
    pack_split(args.input_dir, args.output_dir, "test")
    print(f"Packed dataset written to '{args.output_dir}'.")


if __name__ == "__main__":
    main()
