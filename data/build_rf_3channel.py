#!/usr/bin/env python3
"""Compute 3-channel spectrograms (log-magnitude, instantaneous frequency,
group delay) from an existing packed RF dataset's IQ arrays.

Writes into the same packed directory as a sibling of ``*_spectrogram.npy``:
  - ``train_spectrogram_3c.npy`` / ``test_spectrogram_3c.npy`` (float16,
    shape (N, 3, H, W))
  - ``channel_stats.json`` (per-channel mean/std fit on the train split)

Does not touch the existing single-channel magnitude arrays. Does not
regenerate masks or IQ.

Channel semantics (after per-sample mask, before dataset-wide
standardization applied at loader time):
  - Ch0  log_mag     = clip(20*log10(|X|), db_min, db_max) rescaled to [0, 1]
                      (identical to existing single-channel baseline)
  - Ch1  inst_freq   = (dphi/dt / pi) * mask(|X|)
  - Ch2  group_delay = (-dphi/df / pi) * mask(|X|)

Two gate modes are supported (``--gate-mode``):
  - ``hard`` (default): ``mask = 1`` where ``20*log10(|X|) > db_min +
    mask_margin_db``, else 0. Uses the SAME dB threshold as Ch0's clip,
    so Ch1/Ch2 are zero exactly where Ch0 saturates at the clip floor.
    No per-sample ``epsilon`` hyperparameter; only ``mask_margin_db``,
    which shares units with the existing ``db_min``/``db_max`` clip.
  - ``soft`` (legacy): ``mask = |X|/(|X| + eps*max(|X|))``. Kept for
    backwards comparison with the first pack. Smooth but leaks
    background phase noise for narrowband signals where peak|X| is
    concentrated in a small support.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from numpy.lib.format import open_memmap
from scipy import signal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute 3-channel (mag, IF, GD) spectrograms from packed IQ."
    )
    parser.add_argument(
        "--packed-dir",
        type=Path,
        required=True,
        help="Packed dataset directory containing train_iq.npy / test_iq.npy.",
    )
    parser.add_argument("--sample-rate", type=float, default=20_000_000.0)
    parser.add_argument("--nfft", type=int, default=1024)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument(
        "--gate-mode",
        choices=["hard", "soft"],
        default="hard",
        help="How to gate phase channels. 'hard' applies a fixed dB"
        " threshold (mask=1 where 20log10|X| > gate_db_threshold); 'soft'"
        " uses the legacy |X|/(|X|+eps*max|X|) formula.",
    )
    parser.add_argument(
        "--gate-db-threshold",
        type=float,
        default=-45.0,
        help="(hard gate only) Absolute dB threshold in units of 20log10|X|."
        " Default -45 is calibrated from the normal-class p99 in the SAS"
        " impulsive-only datasets (noise ceiling). Pixels below this are"
        " zeroed in Ch1/Ch2; pixels above carry phase info.",
    )
    parser.add_argument(
        "--mask-epsilon-ratio",
        type=float,
        default=0.05,
        help="(soft gate only) eps as a fraction of per-sample max|X|. Only"
        " used when --gate-mode=soft.",
    )
    parser.add_argument(
        "--db-min",
        type=float,
        default=-85.0,
        help="Lower dB clip for log_mag channel (matches build_rf_dataset.py).",
    )
    parser.add_argument(
        "--db-max",
        type=float,
        default=-10.0,
        help="Upper dB clip for log_mag channel.",
    )
    parser.add_argument(
        "--splits",
        default="train,test",
        help="Comma-separated splits to process.",
    )
    parser.add_argument(
        "--rotate-ccw-90",
        action="store_true",
        help="Match build_rf_dataset.py --rotate-ccw-90 orientation. If the"
        " existing test_spectrogram.npy was built with this flag, pass it"
        " here too so the 3-channel arrays align.",
    )
    return parser.parse_args()


def compute_three_channels(
    iq: np.ndarray,
    sample_rate: float,
    nfft: int,
    image_size: int,
    mask_eps_ratio: float,
    rotate_ccw_90: bool,
    db_min: float = -85.0,
    db_max: float = -10.0,
    gate_mode: str = "hard",
    gate_db_threshold: float = -45.0,
) -> np.ndarray:
    """Return (3, image_size, image_size) float32 array (pre-standardization)."""
    noverlap = nfft // 2
    _freqs, _times, stft = signal.stft(
        iq,
        fs=sample_rate,
        window="hann",
        nperseg=nfft,
        noverlap=noverlap,
        nfft=nfft,
        return_onesided=False,
        boundary=None,
        padded=False,
    )
    # fftshift along frequency to put DC at center (matches build_rf_dataset).
    x = np.fft.fftshift(stft, axes=0).astype(np.complex64)
    if x.size == 0:
        blank = np.zeros((3, image_size, image_size), dtype=np.float32)
        return blank

    amag = np.abs(x).astype(np.float32)
    phase = np.angle(x).astype(np.float32)

    # Ch0: dB-scale log magnitude clipped to [db_min, db_max] then rescaled
    # to [0, 1]. Identical (up to uint8 rounding) to what the baseline
    # single-channel pipeline stores, so Ch0 carries the same information
    # as the R=G=B baseline.
    spec_db = 20.0 * np.log10(np.maximum(amag, 1e-12))
    spec_db = np.clip(spec_db, db_min, db_max)
    db_span = max(db_max - db_min, 1e-12)
    log_mag = ((spec_db - db_min) / db_span).astype(np.float32)

    # Ch1: instantaneous frequency = d phi / d t, unwrapped along time.
    # Normalize by pi so values land in roughly [-1, 1].
    uwt = np.unwrap(phase, axis=1)
    if_raw = np.diff(uwt, axis=1) / np.pi
    # Pad to full T with edge replication so IF shape matches magnitude.
    if_raw = np.concatenate([if_raw[:, :1], if_raw], axis=1).astype(np.float32)

    # Ch2: group delay = -d phi / d omega, unwrapped along freq after fftshift.
    uwf = np.unwrap(phase, axis=0)
    gd_raw = -np.diff(uwf, axis=0) / np.pi
    gd_raw = np.concatenate([gd_raw[:1, :], gd_raw], axis=0).astype(np.float32)

    # Gate phase channels so IF/GD carry signal only where there is
    # measurable energy. 'hard' uses an absolute dB threshold (by default
    # calibrated from the normal-class noise ceiling for these datasets).
    # Note: spec_db has already been clipped to [db_min, db_max] above, so
    # the comparison is against the clipped values; this is fine as long
    # as gate_db_threshold > db_min, which any sensible value satisfies.
    if gate_mode == "hard":
        hard_mask = (spec_db > float(gate_db_threshold)).astype(np.float32)
        inst_freq = if_raw * hard_mask
        group_delay = gd_raw * hard_mask
    else:  # soft (legacy)
        peak = float(np.max(amag))
        eps = max(peak * float(mask_eps_ratio), 1e-8)
        soft_mask = amag / (amag + eps)
        inst_freq = if_raw * soft_mask
        group_delay = gd_raw * soft_mask

    def _resize(arr2d: np.ndarray) -> np.ndarray:
        src_h, src_w = arr2d.shape
        if src_h == image_size and src_w == image_size:
            return arr2d.astype(np.float32, copy=False)
        img = Image.fromarray(arr2d.astype(np.float32), mode="F")
        img = img.resize((image_size, image_size), Image.Resampling.BILINEAR)
        return np.asarray(img, dtype=np.float32)

    channels = np.stack(
        [_resize(log_mag), _resize(inst_freq), _resize(group_delay)], axis=0
    )
    if rotate_ccw_90:
        channels = np.rot90(channels, k=1, axes=(1, 2)).copy()
    return channels.astype(np.float32)


def process_split(
    packed_dir: Path,
    split: str,
    sample_rate: float,
    nfft: int,
    image_size: int,
    mask_eps_ratio: float,
    rotate_ccw_90: bool,
    db_min: float,
    db_max: float,
    gate_mode: str,
    gate_db_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Write {split}_spectrogram_3c.npy. Returns (per-channel sum,
    per-channel sum-of-squares, count) for later train-only stats."""
    iq_path = packed_dir / f"{split}_iq.npy"
    if not iq_path.exists():
        raise FileNotFoundError(f"Missing {iq_path}")
    iq_array = np.load(iq_path, mmap_mode="r")
    n_samples = int(iq_array.shape[0])

    out_path = packed_dir / f"{split}_spectrogram_3c.npy"
    out = open_memmap(
        out_path,
        mode="w+",
        dtype=np.float16,
        shape=(n_samples, 3, image_size, image_size),
    )

    ch_sum = np.zeros(3, dtype=np.float64)
    ch_sq = np.zeros(3, dtype=np.float64)
    ch_count = np.zeros(3, dtype=np.int64)

    start = time.time()
    for i in range(n_samples):
        iq = np.asarray(iq_array[i])
        ch = compute_three_channels(
            iq,
            sample_rate=sample_rate,
            nfft=nfft,
            image_size=image_size,
            mask_eps_ratio=mask_eps_ratio,
            rotate_ccw_90=rotate_ccw_90,
            db_min=db_min,
            db_max=db_max,
            gate_mode=gate_mode,
            gate_db_threshold=gate_db_threshold,
        )
        out[i] = ch.astype(np.float16)
        # Accumulate statistics in float64 for numerical stability.
        ch_sum += ch.reshape(3, -1).sum(axis=1).astype(np.float64)
        ch_sq += (ch.reshape(3, -1) ** 2).sum(axis=1).astype(np.float64)
        ch_count += ch.reshape(3, -1).shape[1]
        if (i + 1) % 500 == 0 or (i + 1) == n_samples:
            elapsed = time.time() - start
            print(
                f"  [{split}] {i + 1}/{n_samples}  ({elapsed:.1f}s elapsed,"
                f" {(i + 1) / max(elapsed, 1e-6):.1f} samp/s)"
            )
    out.flush()
    del out

    return ch_sum, ch_sq, ch_count


def main() -> None:
    args = parse_args()
    packed_dir = args.packed_dir
    if not packed_dir.exists():
        raise FileNotFoundError(f"--packed-dir {packed_dir} does not exist")

    splits = [part.strip() for part in args.splits.split(",") if part.strip()]

    # Check for alignment with existing single-channel array if present.
    ref_spec = packed_dir / "test_spectrogram.npy"
    if ref_spec.exists():
        ref = np.load(ref_spec, mmap_mode="r")
        if ref.shape[1] != args.image_size or ref.shape[2] != args.image_size:
            print(
                f"WARN: existing test_spectrogram.npy is {ref.shape[1]}x{ref.shape[2]}"
                f" but --image-size={args.image_size}. ROI boxes may misalign."
            )

    train_sum = None
    train_sq = None
    train_count = None
    for split in splits:
        print(f"[{split}] computing 3-channel spectrograms...")
        ch_sum, ch_sq, ch_count = process_split(
            packed_dir=packed_dir,
            split=split,
            sample_rate=args.sample_rate,
            nfft=args.nfft,
            image_size=args.image_size,
            mask_eps_ratio=args.mask_epsilon_ratio,
            rotate_ccw_90=args.rotate_ccw_90,
            db_min=args.db_min,
            db_max=args.db_max,
            gate_mode=args.gate_mode,
            gate_db_threshold=args.gate_db_threshold,
        )
        if split == "train":
            train_sum, train_sq, train_count = ch_sum, ch_sq, ch_count

    if train_sum is None:
        raise RuntimeError(
            "channel_stats.json requires the train split; include 'train' in --splits."
        )
    mean = (train_sum / np.maximum(train_count, 1)).astype(np.float64)
    var = (train_sq / np.maximum(train_count, 1)).astype(np.float64) - mean**2
    var = np.maximum(var, 1e-12)
    std = np.sqrt(var)

    stats = {
        "channels": ["log_mag", "inst_freq", "group_delay"],
        "image_size": int(args.image_size),
        "stft_nfft": int(args.nfft),
        "stft_hop": int(args.nfft // 2),
        "sample_rate": float(args.sample_rate),
        "gate_mode": str(args.gate_mode),
        "gate_db_threshold": float(args.gate_db_threshold),
        "mask_epsilon_ratio": float(args.mask_epsilon_ratio),
        "rotate_ccw_90": bool(args.rotate_ccw_90),
        "db_min": float(args.db_min),
        "db_max": float(args.db_max),
        "mean": [float(x) for x in mean.tolist()],
        "std": [float(x) for x in std.tolist()],
        "train_pixel_count_per_channel": int(train_count[0]),
    }
    stats_path = packed_dir / "channel_stats.json"
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    print(f"Wrote {stats_path}:")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
