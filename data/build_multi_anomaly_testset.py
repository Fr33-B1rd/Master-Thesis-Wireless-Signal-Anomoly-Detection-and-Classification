#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from build_rf_dataset import compute_spectrogram, resize_spectrogram, save_iq, save_metadata, save_spectrogram_image
from generate_wifi_iq import (
    ANOMALY_TYPES,
    DEFAULT_BANDWIDTH,
    DEFAULT_CENTER_FREQUENCY,
    DEFAULT_SAMPLE_RATE,
    generate_anomaly_signal,
    generate_capture,
    normalize_rms,
    validate_background_noise_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a multi-anomaly RF test set with 2-3 anomalies per sample."
    )
    parser.add_argument("--count", type=int, default=600, help="Number of test samples to generate.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("rf_dataset_multi_anomaly_test_600"),
        help="Output directory for IQ, metadata, and spectrograms.",
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
    parser.add_argument("--duration-ms", type=float, default=20.0, help="Capture duration in milliseconds.")
    parser.add_argument(
        "--background-noise-models",
        type=str,
        default="impulsive_mix",
        help="Comma-separated background noise models to sample from.",
    )
    parser.add_argument(
        "--enable-blob-noise",
        action="store_true",
        help="Allow the partial_band_blob background noise model. Disabled by default.",
    )
    parser.add_argument("--wifi-snr-min-db", type=float, default=14.0)
    parser.add_argument("--wifi-snr-max-db", type=float, default=20.0)
    parser.add_argument("--anomaly-to-wifi-ratio-min-db", type=float, default=-4.0)
    parser.add_argument("--anomaly-to-wifi-ratio-max-db", type=float, default=0.0)
    parser.add_argument("--anomaly-bandwidth-min-mhz", type=float, default=1.0)
    parser.add_argument("--anomaly-bandwidth-max-mhz", type=float, default=5.0)
    parser.add_argument(
        "--spectrogram-db-min", type=float, default=-85.0, help="Lower dB bound for fixed-dB spectrogram normalization."
    )
    parser.add_argument(
        "--spectrogram-db-max", type=float, default=-10.0, help="Upper dB bound for fixed-dB spectrogram normalization."
    )
    parser.add_argument("--nfft", type=int, default=1024, help="STFT FFT size before resizing.")
    parser.add_argument("--image-size", type=int, default=256, help="Final square spectrogram size in pixels.")
    parser.add_argument("--seed", type=int, default=20260416, help="Random seed.")
    return parser.parse_args()


def parse_background_noise_models(text: str, enable_blob_noise: bool) -> list[str]:
    labels = [item.strip() for item in text.split(",") if item.strip()]
    if not labels:
        raise ValueError("At least one background noise model must be provided.")
    for label in labels:
        validate_background_noise_model(label, enable_blob_noise)
    return labels


def anomaly_count_schedule(index: int, total_count: int) -> int:
    threshold = int(round(total_count * 0.70))
    return 2 if index < threshold else 3


def anomaly_duration_range_ms(anomaly_count: int) -> tuple[float, float]:
    if anomaly_count == 2:
        return 3.0, 8.0
    return 2.0, 6.0


def choose_anomaly_types(anomaly_count: int, rng: np.random.Generator) -> tuple[list[str], str]:
    same_class = bool(rng.random() < 0.40)
    if same_class:
        chosen = str(rng.choice(np.array(ANOMALY_TYPES)))
        return [chosen] * anomaly_count, "same"
    chosen = rng.choice(np.array(ANOMALY_TYPES), size=anomaly_count, replace=False)
    return [str(item) for item in chosen], "different"


def allocate_nonoverlap_starts(total_samples: int, durations: list[int], rng: np.random.Generator) -> list[int]:
    total_duration = sum(durations)
    if total_duration >= total_samples:
        return [0] * len(durations)
    gap_total = total_samples - total_duration
    gap_weights = rng.dirichlet(np.ones(len(durations) + 1))
    gaps = np.floor(gap_weights * gap_total).astype(int)
    gaps[-1] += gap_total - int(np.sum(gaps))
    starts: list[int] = []
    cursor = int(gaps[0])
    for idx, duration in enumerate(durations):
        starts.append(cursor)
        cursor += duration + int(gaps[idx + 1])
    return starts


def allocate_overlap_starts(total_samples: int, durations: list[int], rng: np.random.Generator) -> list[int]:
    starts = [int(rng.integers(0, max(total_samples - durations[0] + 1, 1)))]
    for idx in range(1, len(durations)):
        prev_start = starts[-1]
        prev_duration = durations[idx - 1]
        shift = int(prev_duration * rng.uniform(0.30, 0.70))
        start = prev_start + shift
        max_start = max(total_samples - durations[idx], 0)
        start = max(0, min(start, max_start))
        starts.append(start)
    return starts


def estimate_reference_wifi_rms(metadata: dict) -> float:
    noise_floor_dbfs = metadata.get("noise_floor_dbfs")
    packets = metadata.get("packets", [])
    if noise_floor_dbfs is None or not packets:
        return 10.0 ** (-18.0 / 20.0)
    noise_sigma = 10.0 ** (float(noise_floor_dbfs) / 20.0)
    packet_rms_values = [
        noise_sigma * 10.0 ** (float(packet["approx_snr_db"]) / 20.0)
        for packet in packets
        if packet.get("approx_snr_db") is not None
    ]
    if not packet_rms_values:
        return 10.0 ** (-18.0 / 20.0)
    return float(np.median(packet_rms_values))


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    background_noise_models = parse_background_noise_models(
        args.background_noise_models,
        enable_blob_noise=args.enable_blob_noise,
    )
    test_dir = args.output_dir / "test" / "multi_anomaly"
    test_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str | int | float]] = []
    total_samples = int(round(args.duration_ms * 1e-3 * args.sample_rate))

    for index in range(args.count):
        selected_noise_model = str(rng.choice(np.array(background_noise_models)))
        base_iq, metadata = generate_capture(
            duration_ms=args.duration_ms,
            sample_rate=args.sample_rate,
            center_frequency=args.center_frequency,
            bandwidth=args.bandwidth,
            anomaly_duration_min_ms=1.0,
            anomaly_duration_max_ms=1.0,
            anomaly_bandwidth_min_mhz=args.anomaly_bandwidth_min_mhz,
            anomaly_bandwidth_max_mhz=args.anomaly_bandwidth_max_mhz,
            anomaly_rms_db_range=(-9.0, -3.0),
            wifi_snr_db_range=(args.wifi_snr_min_db, args.wifi_snr_max_db),
            anomaly_to_wifi_ratio_db_range=None,
            include_anomalies=False,
            forced_anomaly_type=None,
            include_wifi_background=True,
            include_background_noise=True,
            background_noise_model=selected_noise_model,
            enable_blob_noise=args.enable_blob_noise,
            forced_traffic_profile=None,
            rng=rng,
        )
        capture = base_iq.astype(np.complex64, copy=True)

        anomaly_count = anomaly_count_schedule(index, args.count)
        duration_min_ms, duration_max_ms = anomaly_duration_range_ms(anomaly_count)
        durations = [
            int(round(rng.uniform(duration_min_ms, duration_max_ms) * 1e-3 * args.sample_rate))
            for _ in range(anomaly_count)
        ]
        durations = [max(1, min(duration, total_samples)) for duration in durations]
        overlap_mode = "partial_overlap" if rng.random() < 0.30 else "non_overlap"
        starts = (
            allocate_overlap_starts(total_samples, durations, rng)
            if overlap_mode == "partial_overlap"
            else allocate_nonoverlap_starts(total_samples, durations, rng)
        )
        anomaly_types, anomaly_type_group = choose_anomaly_types(anomaly_count, rng)
        reference_wifi_rms = estimate_reference_wifi_rms(metadata)

        anomalies: list[dict] = []
        for anomaly_index, (anomaly_type, duration_samples, start_sample) in enumerate(
            zip(anomaly_types, durations, starts)
        ):
            bandwidth_hz = 0.0
            if anomaly_type != "tone":
                bandwidth_hz = float(
                    rng.uniform(args.anomaly_bandwidth_min_mhz * 1e6, args.anomaly_bandwidth_max_mhz * 1e6)
                )
            signal, anomaly_meta = generate_anomaly_signal(
                anomaly_type=anomaly_type,
                duration_samples=duration_samples,
                sample_rate=args.sample_rate,
                bandwidth_hz=bandwidth_hz,
                capture_bandwidth_hz=args.bandwidth,
                anomaly_rms_db_range=(-9.0, -3.0),
                rng=rng,
            )
            target_ratio_db = float(
                rng.uniform(args.anomaly_to_wifi_ratio_min_db, args.anomaly_to_wifi_ratio_max_db)
            )
            target_anomaly_rms = reference_wifi_rms * 10.0 ** (target_ratio_db / 20.0)
            signal = normalize_rms(signal, target_rms=target_anomaly_rms)
            end_sample = min(total_samples, start_sample + duration_samples)
            actual_signal = signal[: end_sample - start_sample]
            capture[start_sample:end_sample] += actual_signal
            anomaly_rms = float(np.sqrt(np.mean(np.abs(actual_signal) ** 2)))
            anomaly_meta.update(
                {
                    "start_sample": int(start_sample),
                    "end_sample": int(end_sample),
                    "start_time_ms": float(start_sample / args.sample_rate * 1e3),
                    "duration_ms": float((end_sample - start_sample) / args.sample_rate * 1e3),
                    "target_anomaly_to_wifi_ratio_db": target_ratio_db,
                    "anomaly_rms": anomaly_rms,
                }
            )
            anomaly_meta["instance_index"] = anomaly_index
            anomalies.append(anomaly_meta)

        metadata["capture_type"] = "multi_anomaly"
        metadata["label"] = "multi_anomaly"
        metadata["dataset_split"] = "test"
        metadata["anomaly_count"] = len(anomalies)
        metadata["anomalies"] = anomalies
        metadata["multi_anomaly_pattern"] = {
            "anomaly_count": len(anomalies),
            "type_group": anomaly_type_group,
            "overlap_mode": overlap_mode,
        }
        metadata["background_noise_models_pool"] = background_noise_models
        metadata["blob_noise_enabled"] = args.enable_blob_noise
        metadata["wifi_snr_db_range"] = [args.wifi_snr_min_db, args.wifi_snr_max_db]
        metadata["anomaly_to_wifi_ratio_db_range"] = [
            args.anomaly_to_wifi_ratio_min_db,
            args.anomaly_to_wifi_ratio_max_db,
        ]
        metadata["spectrogram_size"] = args.image_size
        metadata["stft_nfft"] = args.nfft
        metadata["spectrogram_normalization"] = "fixed_db"
        metadata["spectrogram_db_range"] = [args.spectrogram_db_min, args.spectrogram_db_max]
        metadata["spectrogram_rotation"] = "none"

        spec = compute_spectrogram(
            capture,
            sample_rate=args.sample_rate,
            nfft=args.nfft,
            normalization="fixed_db",
            db_min=args.spectrogram_db_min,
            db_max=args.spectrogram_db_max,
        )
        spec = resize_spectrogram(spec, image_size=args.image_size)

        stem = f"multi_anomaly_{index:05d}"
        iq_path = test_dir / f"{stem}.npy"
        spec_path = test_dir / f"{stem}.png"
        meta_path = test_dir / f"{stem}.json"
        save_iq(capture, iq_path)
        save_spectrogram_image(spec, spec_path)
        save_metadata(metadata, meta_path)

        manifest_rows.append(
            {
                "split": "test",
                "label": "multi_anomaly",
                "index": index,
                "iq_path": str(iq_path.relative_to(args.output_dir)),
                "spectrogram_path": str(spec_path.relative_to(args.output_dir)),
                "metadata_path": str(meta_path.relative_to(args.output_dir)),
                "anomaly_count": len(anomalies),
                "overlap_mode": overlap_mode,
                "type_group": anomaly_type_group,
                "background_noise_model": selected_noise_model,
            }
        )

    manifest_path = args.output_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "label",
                "index",
                "iq_path",
                "spectrogram_path",
                "metadata_path",
                "anomaly_count",
                "overlap_mode",
                "type_group",
                "background_noise_model",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Built multi-anomaly test set in '{args.output_dir}' with count={args.count}.")


if __name__ == "__main__":
    main()
