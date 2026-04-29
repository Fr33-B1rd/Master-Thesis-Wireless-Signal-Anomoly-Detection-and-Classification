#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

DEFAULT_SAMPLE_RATE = 20_000_000
DEFAULT_CENTER_FREQUENCY = 2_432_000_000
ANOMALY_COLORS = {
    "TONE": "#d62728",
    "PULSE": "#1f77b4",
    "CHIRP": "#ff7f0e",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot time domain, PSD, and spectrogram for a complex IQ capture."
    )
    parser.add_argument("input", type=Path, help="Input IQ file (.npy or interleaved float32 .bin).")
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=None,
        help="Sample rate in samples per second. Defaults to metadata or 20 MHz.",
    )
    parser.add_argument(
        "--center-frequency",
        type=float,
        default=None,
        help="Center frequency in Hz. Defaults to metadata or 2.432 GHz.",
    )
    parser.add_argument(
        "--max-time-ms",
        type=float,
        default=20.0,
        help="Displayed time span for the time-domain view in milliseconds.",
    )
    parser.add_argument(
        "--nfft",
        type=int,
        default=1024,
        help="FFT size for PSD and spectrogram.",
    )
    parser.add_argument(
        "--spec-duration-ms",
        type=float,
        default=None,
        help="Spectrogram duration in milliseconds. Defaults to the full capture.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Output figure path. Defaults to <input_stem>_overview.png.",
    )
    return parser.parse_args()


def load_metadata(iq_path: Path) -> dict:
    metadata_path = iq_path.with_suffix(".json")
    if not metadata_path.exists():
        return {}
    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_iq(iq_path: Path) -> np.ndarray:
    suffix = iq_path.suffix.lower()
    if suffix == ".npy":
        iq = np.load(iq_path)
        if not np.iscomplexobj(iq):
            raise ValueError(f"{iq_path} is not a complex-valued .npy file.")
        return iq.astype(np.complex64)

    if suffix == ".bin":
        raw = np.fromfile(iq_path, dtype=np.float32)
        if raw.size % 2 != 0:
            raise ValueError(f"{iq_path} does not contain an even number of float32 values.")
        return (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)

    raise ValueError(f"Unsupported input format: {iq_path.suffix}")


def stft_spectrogram(iq: np.ndarray, nfft: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hop = nfft // 4
    if iq.size < nfft:
        padded = np.zeros(nfft, dtype=np.complex64)
        padded[: iq.size] = iq
        iq = padded

    window = np.hanning(nfft).astype(np.float32)
    starts = np.arange(0, max(iq.size - nfft + 1, 1), hop, dtype=np.int64)
    frames = np.stack([iq[start : start + nfft] * window for start in starts], axis=0)
    spectrum = np.fft.fftshift(np.fft.fft(frames, axis=1), axes=1)
    power_db = 20.0 * np.log10(np.maximum(np.abs(spectrum), 1e-12))
    return power_db.T, starts, window


def annotate_anomalies(
    time_axis,
    spec_axis,
    anomalies: list[dict],
    center_frequency: float,
    preview_limit_ms: float,
    spec_limit_ms: float,
) -> None:
    if not anomalies:
        return

    time_ymin, time_ymax = time_axis.get_ylim()
    time_text_level = time_ymax - 0.08 * (time_ymax - time_ymin)

    spec_ymin, spec_ymax = spec_axis.get_ylim()
    spec_text_margin = 0.03 * (spec_ymax - spec_ymin)

    for anomaly in anomalies:
        label = anomaly.get("type", "ANOM")
        color = ANOMALY_COLORS.get(label, "#9467bd")
        start_ms = float(anomaly.get("start_time_ms", 0.0))
        end_ms = start_ms + float(anomaly.get("duration_ms", 0.0))

        if start_ms < preview_limit_ms:
            visible_end_ms = min(end_ms, preview_limit_ms)
            time_axis.axvspan(start_ms, visible_end_ms, color=color, alpha=0.18)
            text_x = min(start_ms + 0.2, max(preview_limit_ms - 0.5, 0.2))
            time_axis.text(
                text_x,
                time_text_level,
                label,
                color=color,
                fontsize=9,
                weight="bold",
                ha="left",
                va="top",
                bbox={"facecolor": "white", "edgecolor": color, "alpha": 0.7, "pad": 1.5},
            )

        if start_ms >= spec_limit_ms:
            continue

        visible_end_ms = min(end_ms, spec_limit_ms)
        if label == "TONE" and "center_offset_hz" in anomaly:
            tone_freq_mhz = (center_frequency + float(anomaly["center_offset_hz"])) / 1e6
            spec_axis.plot(
                [start_ms, visible_end_ms],
                [tone_freq_mhz, tone_freq_mhz],
                color=color,
                linewidth=2.0,
            )
            text_y = min(tone_freq_mhz + spec_text_margin, spec_ymax - spec_text_margin)
        elif label == "CHIRP" and "chirp_start_offset_hz" in anomaly and "chirp_end_offset_hz" in anomaly:
            start_freq_mhz = (center_frequency + float(anomaly["chirp_start_offset_hz"])) / 1e6
            end_freq_mhz = (center_frequency + float(anomaly["chirp_end_offset_hz"])) / 1e6
            visible_ratio = max(min((visible_end_ms - start_ms) / max(end_ms - start_ms, 1e-9), 1.0), 0.0)
            visible_end_freq_mhz = start_freq_mhz + visible_ratio * (end_freq_mhz - start_freq_mhz)
            spec_axis.plot(
                [start_ms, visible_end_ms],
                [start_freq_mhz, visible_end_freq_mhz],
                color=color,
                linewidth=2.0,
            )
            text_y = min(
                max(start_freq_mhz, visible_end_freq_mhz) + spec_text_margin,
                spec_ymax - spec_text_margin,
            )
        else:
            center_offset_hz = float(anomaly.get("center_offset_hz", 0.0))
            occupied_bandwidth_hz = float(anomaly.get("occupied_bandwidth_hz", 0.0))
            center_mhz = (center_frequency + center_offset_hz) / 1e6
            bottom_mhz = center_mhz - occupied_bandwidth_hz / 2.0 / 1e6
            height_mhz = occupied_bandwidth_hz / 1e6

            rect = Rectangle(
                (start_ms, bottom_mhz),
                max(visible_end_ms - start_ms, 1e-3),
                max(height_mhz, 1e-6),
                linewidth=1.5,
                edgecolor=color,
                facecolor="none",
            )
            spec_axis.add_patch(rect)
            text_y = min(bottom_mhz + height_mhz + spec_text_margin, spec_ymax - spec_text_margin)

        spec_axis.text(
            start_ms + 0.15,
            text_y,
            label,
            color=color,
            fontsize=9,
            weight="bold",
            ha="left",
            va="bottom",
            bbox={"facecolor": "white", "edgecolor": color, "alpha": 0.75, "pad": 1.5},
        )


def plot_capture(
    iq: np.ndarray,
    sample_rate: float,
    center_frequency: float,
    max_time_ms: float,
    spec_duration_ms: float | None,
    nfft: int,
    save_path: Path,
    anomalies: list[dict] | None = None,
    capture_type: str | None = None,
) -> None:
    time_axis_ms = np.arange(iq.size) / sample_rate * 1e3
    preview_samples = min(iq.size, max(1, int(max_time_ms * 1e-3 * sample_rate)))

    freq_axis_hz = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / sample_rate))
    psd = np.fft.fftshift(np.fft.fft(iq[: max(nfft, min(iq.size, 131072))], n=nfft))
    psd_db = 20.0 * np.log10(np.maximum(np.abs(psd), 1e-12))

    spec_samples = iq.size
    if spec_duration_ms is not None:
        spec_samples = min(iq.size, max(1, int(spec_duration_ms * 1e-3 * sample_rate)))
    spec_db, starts, _window = stft_spectrogram(iq[:spec_samples], nfft)
    spec_time_ms = starts / sample_rate * 1e3
    spec_freq_mhz = (center_frequency + freq_axis_hz) / 1e6

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)

    axes[0].plot(time_axis_ms[:preview_samples], iq.real[:preview_samples], label="I", linewidth=0.8)
    axes[0].plot(time_axis_ms[:preview_samples], iq.imag[:preview_samples], label="Q", linewidth=0.8, alpha=0.8)
    axes[0].set_title("IQ Time Domain")
    axes[0].set_xlabel("Time (ms)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot((center_frequency + freq_axis_hz) / 1e6, psd_db, linewidth=0.9)
    axes[1].set_title("Power Spectrum")
    axes[1].set_xlabel("Frequency (MHz)")
    axes[1].set_ylabel("Magnitude (dB)")
    axes[1].grid(True, alpha=0.3)

    mesh = axes[2].imshow(
        spec_db,
        origin="lower",
        aspect="auto",
        extent=[spec_time_ms[0], spec_time_ms[-1] if spec_time_ms.size > 1 else spec_time_ms[0] + 1e-3, spec_freq_mhz[0], spec_freq_mhz[-1]],
        cmap="viridis",
    )
    axes[2].set_title("Spectrogram")
    axes[2].set_xlabel("Time (ms)")
    axes[2].set_ylabel("Frequency (MHz)")
    fig.colorbar(mesh, ax=axes[2], label="Magnitude (dB)")

    preview_limit_ms = preview_samples / sample_rate * 1e3
    spec_limit_ms = spec_samples / sample_rate * 1e3
    annotate_anomalies(
        time_axis=axes[0],
        spec_axis=axes[2],
        anomalies=anomalies or [],
        center_frequency=center_frequency,
        preview_limit_ms=preview_limit_ms,
        spec_limit_ms=spec_limit_ms,
    )

    label = f" | type={capture_type}" if capture_type else ""
    fig.suptitle(
        f"IQ Overview{label} | fc={center_frequency / 1e9:.6f} GHz | fs={sample_rate / 1e6:.2f} MS/s | N={iq.size}"
    )
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    metadata = load_metadata(args.input)
    sample_rate = args.sample_rate or metadata.get("sample_rate_hz", DEFAULT_SAMPLE_RATE)
    center_frequency = args.center_frequency or metadata.get(
        "center_frequency_hz", DEFAULT_CENTER_FREQUENCY
    )

    iq = load_iq(args.input)
    save_path = args.save or args.input.with_name(f"{args.input.stem}_overview.png")
    plot_capture(
        iq=iq,
        sample_rate=float(sample_rate),
        center_frequency=float(center_frequency),
        max_time_ms=args.max_time_ms,
        spec_duration_ms=args.spec_duration_ms,
        nfft=args.nfft,
        save_path=save_path,
        anomalies=metadata.get("anomalies", []),
        capture_type=metadata.get("capture_type"),
    )
    print(f"Saved plot to '{save_path}'.")


if __name__ == "__main__":
    main()
