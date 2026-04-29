#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import signal

from generate_wifi_iq import (
    ANOMALY_TYPES,
    BACKGROUND_NOISE_MODELS,
    DEFAULT_BANDWIDTH,
    DEFAULT_CENTER_FREQUENCY,
    DEFAULT_SAMPLE_RATE,
    generate_capture,
    validate_background_noise_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build RF IQ + 256x256 spectrogram datasets from the current simulator."
    )
    parser.add_argument(
        "--mode",
        choices=("demo", "full"),
        default="demo",
        help="Generate a small anomaly demo set or the full dataset structure.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("rf_dataset_demo"),
        help="Output directory for IQ, metadata, and spectrograms.",
    )
    parser.add_argument(
        "--anomaly-types",
        type=str,
        default="tone,pulse,chirp",
        help="Comma-separated anomaly labels to include in anomaly subsets.",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=DEFAULT_SAMPLE_RATE,
        help="IQ sample rate in samples per second.",
    )
    parser.add_argument(
        "--center-frequency",
        type=float,
        default=DEFAULT_CENTER_FREQUENCY,
        help="RF center frequency in Hz.",
    )
    parser.add_argument(
        "--bandwidth",
        type=float,
        default=DEFAULT_BANDWIDTH,
        help="Capture bandwidth in Hz.",
    )
    parser.add_argument(
        "--duration-ms",
        type=float,
        default=20.0,
        help="Capture duration in milliseconds.",
    )
    parser.add_argument(
        "--anomaly-duration-min-ms",
        type=float,
        default=5.0,
        help="Minimum anomaly duration in milliseconds.",
    )
    parser.add_argument(
        "--anomaly-duration-max-ms",
        type=float,
        default=20.0,
        help="Maximum anomaly duration in milliseconds.",
    )
    parser.add_argument(
        "--anomaly-bandwidth-min-mhz",
        type=float,
        default=1.0,
        help="Minimum anomaly bandwidth in MHz for Pulse/Chirp.",
    )
    parser.add_argument(
        "--anomaly-bandwidth-max-mhz",
        type=float,
        default=5.0,
        help="Maximum anomaly bandwidth in MHz for Pulse/Chirp.",
    )
    parser.add_argument(
        "--anomaly-rms-min-db",
        type=float,
        default=-9.0,
        help="Minimum anomaly RMS level in dB relative to full scale.",
    )
    parser.add_argument(
        "--anomaly-rms-max-db",
        type=float,
        default=-3.0,
        help="Maximum anomaly RMS level in dB relative to full scale.",
    )
    parser.add_argument(
        "--wifi-snr-min-db",
        type=float,
        default=22.0,
        help="Minimum WiFi packet SNR in dB when additive background noise is enabled.",
    )
    parser.add_argument(
        "--wifi-snr-max-db",
        type=float,
        default=34.5,
        help="Maximum WiFi packet SNR in dB when additive background noise is enabled.",
    )
    parser.add_argument(
        "--anomaly-to-wifi-ratio-min-db",
        type=float,
        default=-7.5,
        help="Minimum anomaly-to-WiFi power ratio in dB when WiFi background is present.",
    )
    parser.add_argument(
        "--anomaly-to-wifi-ratio-max-db",
        type=float,
        default=3.0,
        help="Maximum anomaly-to-WiFi power ratio in dB when WiFi background is present.",
    )
    parser.add_argument(
        "--nfft",
        type=int,
        default=1024,
        help="STFT FFT size before resizing.",
    )
    parser.add_argument(
        "--spectrogram-normalization",
        choices=("fixed_db", "per_sample_peak"),
        default="fixed_db",
        help="Map spectrogram magnitudes using a fixed dB range or the legacy per-sample peak range.",
    )
    parser.add_argument(
        "--spectrogram-db-min",
        type=float,
        default=-85.0,
        help="Lower dB bound for fixed-dB spectrogram normalization.",
    )
    parser.add_argument(
        "--spectrogram-db-max",
        type=float,
        default=-10.0,
        help="Upper dB bound for fixed-dB spectrogram normalization.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Final square spectrogram size in pixels.",
    )
    parser.add_argument(
        "--rotate-ccw-90",
        action="store_true",
        help="Rotate the spectrogram 90 degrees counterclockwise so x=freq and y=time.",
    )
    parser.add_argument(
        "--no-wifi-background",
        action="store_true",
        help="Disable the WiFi background and keep only noise or the injected anomaly.",
    )
    parser.add_argument(
        "--no-background-noise",
        action="store_true",
        help="Disable additive background noise while keeping other enabled signal components.",
    )
    parser.add_argument(
        "--background-noise-models",
        type=str,
        default="white",
        help="Comma-separated background noise models to sample from when additive noise is enabled.",
    )
    parser.add_argument(
        "--enable-blob-noise",
        action="store_true",
        help="Allow the partial_band_blob background noise model. Disabled by default.",
    )
    parser.add_argument("--seed", type=int, default=2026, help="Random seed.")
    return parser.parse_args()


def parse_anomaly_types(text: str) -> list[str]:
    labels = [item.strip().lower() for item in text.split(",") if item.strip()]
    if not labels:
        raise ValueError("At least one anomaly type must be provided.")
    invalid = [label for label in labels if label not in ANOMALY_TYPES]
    if invalid:
        supported = ", ".join(ANOMALY_TYPES)
        raise ValueError(f"Unsupported anomaly type(s): {', '.join(invalid)}. Supported: {supported}")
    return labels


def parse_background_noise_models(text: str, enable_blob_noise: bool) -> list[str]:
    labels = [item.strip() for item in text.split(",") if item.strip()]
    if not labels:
        raise ValueError("At least one background noise model must be provided.")
    for label in labels:
        validate_background_noise_model(label, enable_blob_noise)
    return labels


def dataset_plan(mode: str, anomaly_types: list[str]) -> list[tuple[str, str, int]]:
    if mode == "demo":
        return [("demo", label, 4) for label in anomaly_types]
    plan = [
        ("train", "normal", 4000),
        ("test", "normal", 1200),
    ]
    plan.extend(("test", label, 800) for label in anomaly_types)
    return plan


def compute_spectrogram(
    iq: np.ndarray,
    sample_rate: float,
    nfft: int,
    normalization: str,
    db_min: float,
    db_max: float,
) -> np.ndarray:
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
    spec = np.fft.fftshift(stft, axes=0)
    spec_db = 20.0 * np.log10(np.maximum(np.abs(spec), 1e-12))
    if normalization == "per_sample_peak":
        max_db = float(np.max(spec_db))
        db_hi = max_db
        db_lo = max_db - 80.0
    else:
        db_lo = db_min
        db_hi = db_max
    spec_db = np.clip(spec_db, db_lo, db_hi)
    spec_img = ((spec_db - db_lo) / max(db_hi - db_lo, 1e-12) * 255.0).astype(np.uint8)
    return spec_img


def stft_frame_centers(num_samples: int, nfft: int) -> np.ndarray:
    hop = nfft // 2
    if num_samples < nfft:
        return np.array([], dtype=np.float64)
    frame_count = 1 + (num_samples - nfft) // hop
    return (nfft / 2.0) + np.arange(frame_count, dtype=np.float64) * hop


def stft_frequency_axis(sample_rate: float, nfft: int) -> np.ndarray:
    return np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / sample_rate))


def overlapping_frame_indices(
    frame_centers_samples: np.ndarray,
    nfft: int,
    start_sample: int,
    end_sample: int,
) -> np.ndarray:
    half_window = nfft / 2.0
    frame_starts = frame_centers_samples - half_window
    frame_ends = frame_centers_samples + half_window
    return np.where((frame_ends > start_sample) & (frame_starts < end_sample))[0]


def band_row_indices(freq_axis_hz: np.ndarray, low_hz: float, high_hz: float) -> np.ndarray:
    if freq_axis_hz.size == 0:
        return np.array([], dtype=np.int64)
    if high_hz < low_hz:
        low_hz, high_hz = high_hz, low_hz
    if freq_axis_hz.size == 1:
        return np.array([0], dtype=np.int64)
    bin_hz = abs(float(freq_axis_hz[1] - freq_axis_hz[0]))
    rows = np.where((freq_axis_hz >= low_hz - 0.5 * bin_hz) & (freq_axis_hz <= high_hz + 0.5 * bin_hz))[0]
    if rows.size > 0:
        return rows.astype(np.int64)
    center_hz = 0.5 * (low_hz + high_hz)
    nearest = int(np.argmin(np.abs(freq_axis_hz - center_hz)))
    return np.array([nearest], dtype=np.int64)


def nearest_row_index(freq_axis_hz: np.ndarray, center_hz: float) -> int:
    return int(np.argmin(np.abs(freq_axis_hz - center_hz)))


def dilated_rows(center_row: int, row_radius: int, row_count: int) -> np.ndarray:
    lo = max(0, center_row - row_radius)
    hi = min(row_count, center_row + row_radius + 1)
    return np.arange(lo, hi, dtype=np.int64)


def bresenham_points(r0: int, c0: int, r1: int, c1: int) -> np.ndarray:
    """Integer pixels on the line segment (r0,c0)->(r1,c1) inclusive.

    Used to connect per-frame (row, col) CHIRP points into a continuous polyline
    before rasterization. Without this the stamped columns form a diagonal of
    isolated "+" shapes (median ~26 connected components per chirp), which
    breaks per-region AUPRO evaluation.
    """
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dc - dr
    r, c = int(r0), int(c0)
    points = []
    while True:
        points.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dr:
            err -= dr
            c += sc
        if e2 < dc:
            err += dc
            r += sr
    return np.asarray(points, dtype=np.int64)


def build_anomaly_mask(
    metadata: dict,
    sample_rate: float,
    nfft: int,
    image_size: int,
    rotate_ccw_90: bool,
) -> np.ndarray:
    num_samples = int(metadata["num_samples"])
    raw_mask = np.zeros((nfft, stft_frame_centers(num_samples, nfft).size), dtype=np.uint8)
    if raw_mask.size == 0 or int(metadata.get("anomaly_count", 0)) == 0:
        resized = resize_mask(raw_mask, image_size=image_size)
        return maybe_rotate_spectrogram(resized, rotate_ccw_90=rotate_ccw_90)

    frame_centers = stft_frame_centers(num_samples, nfft)
    freq_axis = stft_frequency_axis(sample_rate, nfft)

    for anomaly in metadata.get("anomalies", []):
        active_cols = overlapping_frame_indices(
            frame_centers_samples=frame_centers,
            nfft=nfft,
            start_sample=int(anomaly["start_sample"]),
            end_sample=int(anomaly["end_sample"]),
        )
        if active_cols.size == 0:
            continue

        anomaly_type = str(anomaly["type"]).upper()
        if anomaly_type == "TONE":
            row = nearest_row_index(freq_axis, float(anomaly["center_offset_hz"]))
            rows = dilated_rows(row, row_radius=1, row_count=nfft)
            raw_mask[np.ix_(rows, active_cols)] = 1
        elif anomaly_type == "CHIRP":
            start_freq = float(anomaly["chirp_start_offset_hz"])
            end_freq = float(anomaly["chirp_end_offset_hz"])
            start_sample = int(anomaly["start_sample"])
            duration_samples = max(int(anomaly["end_sample"]) - start_sample, 1)
            local_times = np.clip((frame_centers[active_cols] - start_sample) / sample_rate, 0.0, duration_samples / sample_rate)
            duration_s = max(duration_samples / sample_rate, 1e-12)
            freqs_hz = start_freq + (end_freq - start_freq) * (local_times / duration_s)
            # Stamp a continuous polyline along the chirp trajectory. Per-column
            # stamps alone leave row-gaps whenever the frequency change between
            # adjacent STFT frames exceeds 2*row_radius bins; Bresenham fills
            # those gaps so the resulting GT is a single connected region.
            prev_row: int | None = None
            prev_col: int | None = None
            n_cols = raw_mask.shape[1]
            for col, freq_hz in zip(active_cols, freqs_hz):
                row = nearest_row_index(freq_axis, float(freq_hz))
                if prev_row is not None:
                    for lr, lc in bresenham_points(prev_row, prev_col, int(row), int(col)):
                        if 0 <= lr < nfft and 0 <= lc < n_cols:
                            rr = dilated_rows(int(lr), row_radius=1, row_count=nfft)
                            raw_mask[rr, lc] = 1
                else:
                    rr = dilated_rows(int(row), row_radius=1, row_count=nfft)
                    raw_mask[rr, int(col)] = 1
                prev_row = int(row)
                prev_col = int(col)
        elif anomaly_type == "COMB":
            tone_count = int(anomaly.get("tone_count", 1))
            center_hz = float(anomaly["center_offset_hz"])
            spacing_hz = float(anomaly.get("tone_spacing_hz", 0.0))
            start_hz = center_hz - 0.5 * spacing_hz * max(tone_count - 1, 0)
            for tone_idx in range(tone_count):
                freq_hz = start_hz + tone_idx * spacing_hz
                row = nearest_row_index(freq_axis, freq_hz)
                rows = dilated_rows(row, row_radius=1, row_count=nfft)
                raw_mask[np.ix_(rows, active_cols)] = 1
        else:
            center_hz = float(anomaly.get("center_offset_hz", 0.0))
            occupied_hz = float(anomaly.get("occupied_bandwidth_hz", 0.0))
            rows = band_row_indices(
                freq_axis,
                center_hz - 0.5 * occupied_hz,
                center_hz + 0.5 * occupied_hz,
            )
            raw_mask[np.ix_(rows, active_cols)] = 1

    resized = resize_mask(raw_mask, image_size=image_size)
    return maybe_rotate_spectrogram(resized, rotate_ccw_90=rotate_ccw_90)


def resize_spectrogram(spec: np.ndarray, image_size: int) -> np.ndarray:
    image = Image.fromarray(spec, mode="L")
    image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    return np.array(image, dtype=np.uint8)


def resize_mask(mask: np.ndarray, image_size: int) -> np.ndarray:
    """Resize a binary anomaly mask to (image_size, image_size).

    Downsampling uses MAX-pool semantics over block boundaries: any positive
    source pixel inside a destination cell keeps the destination positive.
    This is required because PIL's NEAREST resize samples a single source pixel
    per destination cell, which causes narrow-band masks (e.g. 3-row TONE
    bands at nfft=1024 downsampled to image_size=256, block=4x) to vanish
    ~25% of the time depending on sub-cell alignment.

    Upsampling falls back to NEAREST (no information loss possible).
    """
    src_h, src_w = mask.shape
    mbin = (mask > 0)
    if src_h == image_size and src_w == image_size:
        return mbin.astype(np.uint8)
    if src_h == 0 or src_w == 0:
        return np.zeros((image_size, image_size), dtype=np.uint8)

    if src_h >= image_size:
        row_edges = np.linspace(0, src_h, image_size + 1).astype(np.int64)
        row_reduced = np.zeros((image_size, src_w), dtype=bool)
        for i in range(image_size):
            r0 = row_edges[i]
            r1 = max(row_edges[i + 1], r0 + 1)
            row_reduced[i] = mbin[r0:r1].any(axis=0)
    else:
        image = Image.fromarray(mbin.astype(np.uint8) * 255, mode="L")
        image = image.resize((src_w, image_size), Image.Resampling.NEAREST)
        row_reduced = np.array(image, dtype=np.uint8) > 0

    if src_w >= image_size:
        col_edges = np.linspace(0, src_w, image_size + 1).astype(np.int64)
        out = np.zeros((image_size, image_size), dtype=bool)
        for j in range(image_size):
            c0 = col_edges[j]
            c1 = max(col_edges[j + 1], c0 + 1)
            out[:, j] = row_reduced[:, c0:c1].any(axis=1)
    else:
        image = Image.fromarray(row_reduced.astype(np.uint8) * 255, mode="L")
        image = image.resize((image_size, image_size), Image.Resampling.NEAREST)
        out = np.array(image, dtype=np.uint8) > 0
    return out.astype(np.uint8)


def maybe_rotate_spectrogram(spec: np.ndarray, rotate_ccw_90: bool) -> np.ndarray:
    if not rotate_ccw_90:
        return spec
    return np.rot90(spec, k=1).copy()


def save_iq(iq: np.ndarray, path: Path) -> None:
    np.save(path, iq.astype(np.complex64))


def save_spectrogram_image(spec: np.ndarray, path: Path) -> None:
    Image.fromarray(spec, mode="L").save(path)


def save_mask(mask: np.ndarray, path: Path) -> None:
    np.save(path, mask.astype(np.uint8))


def save_metadata(metadata: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)


def build_capture(
    label: str,
    sample_rate: float,
    center_frequency: float,
    bandwidth: float,
    duration_ms: float,
    anomaly_duration_min_ms: float,
    anomaly_duration_max_ms: float,
    anomaly_bandwidth_min_mhz: float,
    anomaly_bandwidth_max_mhz: float,
    anomaly_rms_db_range: tuple[float, float],
    wifi_snr_db_range: tuple[float, float],
    anomaly_to_wifi_ratio_db_range: tuple[float, float] | None,
    include_wifi_background: bool,
    include_background_noise: bool,
    background_noise_model: str,
    enable_blob_noise: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    is_normal = label == "normal"
    return generate_capture(
        duration_ms=duration_ms,
        sample_rate=sample_rate,
        center_frequency=center_frequency,
        bandwidth=bandwidth,
        anomaly_duration_min_ms=anomaly_duration_min_ms,
        anomaly_duration_max_ms=anomaly_duration_max_ms,
        anomaly_bandwidth_min_mhz=anomaly_bandwidth_min_mhz,
        anomaly_bandwidth_max_mhz=anomaly_bandwidth_max_mhz,
        anomaly_rms_db_range=anomaly_rms_db_range,
        wifi_snr_db_range=wifi_snr_db_range,
        anomaly_to_wifi_ratio_db_range=anomaly_to_wifi_ratio_db_range,
        include_anomalies=not is_normal,
        forced_anomaly_type=None if is_normal else label,
        include_wifi_background=include_wifi_background,
        include_background_noise=include_background_noise,
        background_noise_model=background_noise_model,
        enable_blob_noise=enable_blob_noise,
        forced_traffic_profile=None,
        rng=rng,
    )


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    anomaly_types = parse_anomaly_types(args.anomaly_types)
    background_noise_models = parse_background_noise_models(
        args.background_noise_models,
        enable_blob_noise=args.enable_blob_noise,
    )
    plan = dataset_plan(args.mode, anomaly_types)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, str | int | float]] = []

    for split, label, count in plan:
        subset_dir = args.output_dir / split / label
        subset_dir.mkdir(parents=True, exist_ok=True)

        for index in range(count):
            selected_background_noise_model = str(rng.choice(np.array(background_noise_models)))
            iq, metadata = build_capture(
                label=label,
                sample_rate=args.sample_rate,
                center_frequency=args.center_frequency,
                bandwidth=args.bandwidth,
                duration_ms=args.duration_ms,
                anomaly_duration_min_ms=args.anomaly_duration_min_ms,
                anomaly_duration_max_ms=args.anomaly_duration_max_ms,
                anomaly_bandwidth_min_mhz=args.anomaly_bandwidth_min_mhz,
                anomaly_bandwidth_max_mhz=args.anomaly_bandwidth_max_mhz,
                anomaly_rms_db_range=(args.anomaly_rms_min_db, args.anomaly_rms_max_db),
                wifi_snr_db_range=(args.wifi_snr_min_db, args.wifi_snr_max_db),
                anomaly_to_wifi_ratio_db_range=(
                    args.anomaly_to_wifi_ratio_min_db,
                    args.anomaly_to_wifi_ratio_max_db,
                ),
                include_wifi_background=not args.no_wifi_background,
                include_background_noise=not args.no_background_noise,
                background_noise_model=selected_background_noise_model,
                enable_blob_noise=args.enable_blob_noise,
                rng=rng,
            )
            spec = compute_spectrogram(
                iq,
                sample_rate=args.sample_rate,
                nfft=args.nfft,
                normalization=args.spectrogram_normalization,
                db_min=args.spectrogram_db_min,
                db_max=args.spectrogram_db_max,
            )
            spec = resize_spectrogram(spec, image_size=args.image_size)
            spec = maybe_rotate_spectrogram(spec, rotate_ccw_90=args.rotate_ccw_90)
            mask = build_anomaly_mask(
                metadata=metadata,
                sample_rate=args.sample_rate,
                nfft=args.nfft,
                image_size=args.image_size,
                rotate_ccw_90=args.rotate_ccw_90,
            )

            stem = f"{label}_{index:05d}"
            iq_path = subset_dir / f"{stem}.npy"
            spec_path = subset_dir / f"{stem}.png"
            mask_path = subset_dir / f"{stem}_mask.npy"
            meta_path = subset_dir / f"{stem}.json"

            metadata.update(
                {
                    "dataset_split": split,
                    "label": label,
                    "spectrogram_size": args.image_size,
                    "stft_nfft": args.nfft,
                    "include_wifi_background": not args.no_wifi_background,
                    "include_background_noise": not args.no_background_noise,
                    "blob_noise_enabled": args.enable_blob_noise,
                    "background_noise_model": selected_background_noise_model
                    if not args.no_background_noise
                    else "disabled",
                    "background_noise_models_pool": background_noise_models if not args.no_background_noise else [],
                    "spectrogram_rotation": "ccw_90" if args.rotate_ccw_90 else "none",
                    "mask_format": "npy",
                    "mask_dtype": "uint8",
                    "mask_shape": [args.image_size, args.image_size],
                    "mask_semantics": "approx_time_frequency_anomaly_mask",
                    "anomaly_rms_db_range": [args.anomaly_rms_min_db, args.anomaly_rms_max_db],
                    "wifi_snr_db_range": [args.wifi_snr_min_db, args.wifi_snr_max_db],
                    "anomaly_to_wifi_ratio_db_range": [
                        args.anomaly_to_wifi_ratio_min_db,
                        args.anomaly_to_wifi_ratio_max_db,
                    ],
                    "spectrogram_normalization": args.spectrogram_normalization,
                    "spectrogram_db_range": [args.spectrogram_db_min, args.spectrogram_db_max],
                }
            )

            save_iq(iq, iq_path)
            save_spectrogram_image(spec, spec_path)
            save_mask(mask, mask_path)
            save_metadata(metadata, meta_path)

            manifest_rows.append(
                {
                    "split": split,
                    "label": label,
                    "index": index,
                    "iq_path": str(iq_path.relative_to(args.output_dir)),
                    "spectrogram_path": str(spec_path.relative_to(args.output_dir)),
                    "mask_path": str(mask_path.relative_to(args.output_dir)),
                    "metadata_path": str(meta_path.relative_to(args.output_dir)),
                }
            )

    manifest_path = args.output_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "label", "index", "iq_path", "spectrogram_path", "mask_path", "metadata_path"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Built dataset in '{args.output_dir}' with mode='{args.mode}'.")


if __name__ == "__main__":
    main()
