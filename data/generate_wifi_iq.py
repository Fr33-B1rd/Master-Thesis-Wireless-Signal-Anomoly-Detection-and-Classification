#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FFT_LEN = 64
CP_LEN = 16
LTF_CP_LEN = 32
DEFAULT_SAMPLE_RATE = 20_000_000
DEFAULT_CENTER_FREQUENCY = 2_432_000_000
DEFAULT_BANDWIDTH = 20_000_000
USED_SUBCARRIERS = np.concatenate((np.arange(-26, 0), np.arange(1, 27)))
PILOT_SUBCARRIERS = np.array([-21, -7, 7, 21])
DATA_SUBCARRIERS = np.array(
    [carrier for carrier in USED_SUBCARRIERS if carrier not in set(PILOT_SUBCARRIERS)]
)
PILOT_VALUES = np.array([1.0, 1.0, 1.0, -1.0], dtype=np.complex128)
SUPPORTED_WIFI_MODULATIONS = ("bpsk", "qpsk", "16qam")
ANOMALY_TYPES = ("tone", "pulse", "chirp", "fsk", "ofdm", "comb")
BACKGROUND_NOISE_MODELS = ("white", "bandlimited", "colored_spurs", "impulsive_mix", "partial_band_blob")
WIFI_SUBCARRIER_WEIGHTS = {
    int(carrier): 0.72 + 0.28 * np.cos((abs(carrier) / 26.0) * (np.pi / 2.0))
    for carrier in USED_SUBCARRIERS
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate normal and anomalous WiFi-like IQ captures."
    )
    parser.add_argument("--count", type=int, default=1, help="Number of captures to generate per type.")
    parser.add_argument(
        "--duration-ms",
        type=float,
        default=20.0,
        help="Capture duration in milliseconds.",
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
        "--output-dir",
        type=Path,
        default=Path("generated_wifi_iq"),
        help="Directory for generated IQ files and metadata.",
    )
    parser.add_argument(
        "--format",
        choices=("npy", "bin"),
        default="npy",
        help="Output IQ format: .npy complex64 or interleaved float32 .bin.",
    )
    parser.add_argument(
        "--capture-mode",
        choices=("normal", "anomaly", "both"),
        default="both",
        help="Generate normal captures, anomaly captures, or both.",
    )
    parser.add_argument(
        "--no-wifi-background",
        action="store_true",
        help="Disable the WiFi background so captures contain only noise or the injected anomaly.",
    )
    parser.add_argument(
        "--no-background-noise",
        action="store_true",
        help="Disable additive background noise while keeping other enabled signal components.",
    )
    parser.add_argument(
        "--background-noise-model",
        choices=BACKGROUND_NOISE_MODELS,
        default="white",
        help="Background noise model used when additive noise is enabled.",
    )
    parser.add_argument(
        "--enable-blob-noise",
        action="store_true",
        help="Allow the partial_band_blob background noise model. Disabled by default.",
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
        help="Minimum anomaly bandwidth in MHz.",
    )
    parser.add_argument(
        "--anomaly-bandwidth-max-mhz",
        type=float,
        default=5.0,
        help="Maximum anomaly bandwidth in MHz.",
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
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    return parser.parse_args()


def normalize_rms(signal: np.ndarray, target_rms: float = 1.0) -> np.ndarray:
    rms = np.sqrt(np.mean(np.abs(signal) ** 2))
    if rms <= 1e-12:
        return signal.astype(np.complex64)
    return (signal * (target_rms / rms)).astype(np.complex64)


def approx_snr_db(signal_rms: float, noise_sigma: float) -> float | None:
    if noise_sigma <= 0.0:
        return None
    return float(20.0 * np.log10(signal_rms / max(noise_sigma, 1e-12)))


def validate_background_noise_model(model: str, enable_blob_noise: bool) -> None:
    if model not in BACKGROUND_NOISE_MODELS:
        supported = ", ".join(BACKGROUND_NOISE_MODELS)
        raise ValueError(f"Unsupported background noise model: {model}. Supported: {supported}")
    if model == "partial_band_blob" and not enable_blob_noise:
        raise ValueError(
            "Background noise model 'partial_band_blob' is disabled by default. "
            "Pass --enable-blob-noise to allow it."
        )


def random_constellation(modulation: str, count: int, rng: np.random.Generator) -> np.ndarray:
    if modulation == "bpsk":
        return rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=count).astype(np.complex128)
    if modulation == "qpsk":
        i = rng.choice(np.array([-1.0, 1.0]), size=count)
        q = rng.choice(np.array([-1.0, 1.0]), size=count)
        return (i + 1j * q) / np.sqrt(2.0)
    if modulation == "16qam":
        levels = np.array([-3.0, -1.0, 1.0, 3.0], dtype=np.float64)
        i = rng.choice(levels, size=count)
        q = rng.choice(levels, size=count)
        return (i + 1j * q) / np.sqrt(10.0)
    raise ValueError(f"Unsupported modulation: {modulation}")


def design_lowpass(cutoff_hz: float, sample_rate: float, taps: int = 129) -> np.ndarray:
    cutoff_hz = min(cutoff_hz, 0.49 * sample_rate)
    n = np.arange(taps, dtype=np.float64) - (taps - 1) / 2.0
    h = 2.0 * cutoff_hz / sample_rate * np.sinc(2.0 * cutoff_hz * n / sample_rate)
    h *= np.hamming(taps)
    h /= np.sum(h)
    return h.astype(np.float64)


def bandlimit_signal(signal: np.ndarray, cutoff_hz: float, sample_rate: float) -> np.ndarray:
    taps = design_lowpass(cutoff_hz=cutoff_hz, sample_rate=sample_rate)
    return np.convolve(signal, taps, mode="same")


def apply_taper(signal: np.ndarray, edge_samples: int) -> np.ndarray:
    edge_samples = max(1, min(edge_samples, signal.size // 2))
    window = np.ones(signal.size, dtype=np.float64)
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, edge_samples, endpoint=True))
    window[:edge_samples] = ramp
    window[-edge_samples:] = ramp[::-1]
    return signal * window


def choose_center_offset_hz(
    occupied_bandwidth_hz: float,
    capture_bandwidth_hz: float,
    rng: np.random.Generator,
) -> float:
    guard_hz = 250_000.0
    max_offset = max((capture_bandwidth_hz - occupied_bandwidth_hz) / 2.0 - guard_hz, 0.0)
    return float(rng.uniform(-max_offset, max_offset))


def complex_white_noise(
    sample_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    return ((rng.normal(size=sample_count) + 1j * rng.normal(size=sample_count)) / np.sqrt(2.0)).astype(
        np.complex64
    )


def generate_background_noise(
    sample_count: int,
    sample_rate: float,
    noise_sigma: float,
    model: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    noise = complex_white_noise(sample_count, rng).astype(np.complex128)
    metadata: dict[str, float | int | list[float]] = {"background_noise_model": model}

    if model == "white":
        pass
    elif model == "bandlimited":
        cutoff_hz = float(rng.uniform(0.28 * sample_rate, 0.42 * sample_rate))
        noise = bandlimit_signal(noise, cutoff_hz=cutoff_hz / 2.0, sample_rate=sample_rate)
        metadata["noise_cutoff_hz"] = cutoff_hz
    elif model == "colored_spurs":
        freq_response = np.fft.fft(noise)
        freqs = np.fft.fftfreq(sample_count, d=1.0 / sample_rate)
        tilt_db = float(rng.uniform(-10.0, 10.0))
        ripple_strength_db = float(rng.uniform(1.0, 3.0))
        normalized_freq = np.abs(freqs) / max(sample_rate / 2.0, 1e-12)
        shaping_db = tilt_db * (normalized_freq - 0.5) + ripple_strength_db * np.cos(2.0 * np.pi * 3.0 * normalized_freq)
        shaping = 10.0 ** (shaping_db / 20.0)
        noise = np.fft.ifft(freq_response * shaping)

        spur_count = int(rng.integers(2, 5))
        spur_offsets_hz: list[float] = []
        t = np.arange(sample_count, dtype=np.float64) / sample_rate
        for _ in range(spur_count):
            offset_hz = choose_center_offset_hz(0.0, sample_rate, rng)
            spur_level_db = float(rng.uniform(-18.0, -10.0))
            spur_amplitude = noise_sigma * 10.0 ** (spur_level_db / 20.0)
            phase0 = float(rng.uniform(0.0, 2.0 * np.pi))
            noise += spur_amplitude * np.exp(1j * (2.0 * np.pi * offset_hz * t + phase0))
            spur_offsets_hz.append(float(offset_hz))
        metadata["noise_tilt_db"] = tilt_db
        metadata["noise_ripple_strength_db"] = ripple_strength_db
        metadata["spur_count"] = spur_count
        metadata["spur_offsets_hz"] = spur_offsets_hz
    elif model == "impulsive_mix":
        impulse_count = int(rng.integers(6, 16))
        envelope = np.ones(sample_count, dtype=np.float64)
        impulse_peaks_db: list[float] = []
        for _ in range(impulse_count):
            width = int(rng.integers(16, 256))
            start = int(rng.integers(0, max(sample_count - width, 1)))
            peak_db = float(rng.uniform(8.0, 18.0))
            pulse = np.hanning(width)
            envelope[start : start + width] += (10.0 ** (peak_db / 20.0) - 1.0) * pulse
            impulse_peaks_db.append(peak_db)
        noise *= envelope
        metadata["impulse_count"] = impulse_count
        metadata["impulse_peak_db_range"] = [float(min(impulse_peaks_db)), float(max(impulse_peaks_db))]
    elif model == "partial_band_blob":
        blob_count = int(rng.integers(1, 3))
        blob_bandwidths_hz: list[float] = []
        blob_centers_hz: list[float] = []
        blob_durations_ms: list[float] = []
        blob_levels_db: list[float] = []
        sample_times = np.arange(sample_count, dtype=np.float64) / sample_rate
        blob_mix = np.zeros(sample_count, dtype=np.complex128)

        for _ in range(blob_count):
            blob_bandwidth_hz = float(rng.uniform(1.5e6, 5.0e6))
            blob_center_hz = choose_center_offset_hz(blob_bandwidth_hz, sample_rate, rng)
            blob_duration_samples = int(rng.integers(max(8192, sample_count // 8), max(8193, sample_count // 2)))
            blob_duration_samples = min(blob_duration_samples, sample_count)
            blob_start = int(rng.integers(0, max(sample_count - blob_duration_samples + 1, 1)))
            blob_level_db = float(rng.uniform(4.0, 10.0))

            local = complex_white_noise(sample_count, rng).astype(np.complex128)
            local = bandlimit_signal(local, cutoff_hz=blob_bandwidth_hz / 2.0, sample_rate=sample_rate)
            local *= np.exp(
                1j * (2.0 * np.pi * blob_center_hz * sample_times + rng.uniform(0.0, 2.0 * np.pi))
            )

            envelope = np.zeros(sample_count, dtype=np.float64)
            taper = np.hanning(blob_duration_samples)
            envelope[blob_start : blob_start + blob_duration_samples] = taper
            local *= envelope * (10.0 ** (blob_level_db / 20.0))
            blob_mix += local

            blob_bandwidths_hz.append(blob_bandwidth_hz)
            blob_centers_hz.append(blob_center_hz)
            blob_durations_ms.append(float(blob_duration_samples / sample_rate * 1e3))
            blob_levels_db.append(blob_level_db)

        noise += blob_mix
        metadata["blob_count"] = blob_count
        metadata["blob_bandwidths_hz"] = blob_bandwidths_hz
        metadata["blob_center_offsets_hz"] = blob_centers_hz
        metadata["blob_durations_ms"] = blob_durations_ms
        metadata["blob_levels_db_above_base"] = blob_levels_db
    else:
        raise ValueError(f"Unsupported background noise model: {model}")

    noise = normalize_rms(noise, target_rms=noise_sigma)
    metadata["noise_rms_target"] = float(noise_sigma)
    return noise.astype(np.complex64), metadata


def build_ofdm_symbol(
    data_symbols: np.ndarray,
    pilot_polarity: float = 1.0,
    cp_len: int = CP_LEN,
) -> np.ndarray:
    freq_domain = np.zeros(FFT_LEN, dtype=np.complex128)
    data_weights = np.array([WIFI_SUBCARRIER_WEIGHTS[int(carrier)] for carrier in DATA_SUBCARRIERS])
    pilot_weights = np.array([WIFI_SUBCARRIER_WEIGHTS[int(carrier)] for carrier in PILOT_SUBCARRIERS])
    freq_domain[DATA_SUBCARRIERS % FFT_LEN] = data_symbols * data_weights
    freq_domain[PILOT_SUBCARRIERS % FFT_LEN] = PILOT_VALUES * pilot_polarity * pilot_weights
    time_domain = np.fft.ifft(freq_domain) * np.sqrt(FFT_LEN)
    return np.concatenate((time_domain[-cp_len:], time_domain))


def short_training_field() -> np.ndarray:
    tones = np.zeros(FFT_LEN, dtype=np.complex128)
    short_subcarriers = np.array([-24, -20, -16, -12, -8, -4, 4, 8, 12, 16, 20, 24])
    short_values = np.array(
        [
            1 + 1j,
            -1 - 1j,
            1 + 1j,
            -1 - 1j,
            -1 - 1j,
            1 + 1j,
            1 + 1j,
            -1 - 1j,
            1 + 1j,
            1 + 1j,
            -1 - 1j,
            1 + 1j,
        ],
        dtype=np.complex128,
    )
    short_weights = np.array(
        [WIFI_SUBCARRIER_WEIGHTS.get(int(carrier), 0.9) for carrier in short_subcarriers],
        dtype=np.float64,
    )
    tones[short_subcarriers % FFT_LEN] = short_values * short_weights
    short_symbol = np.fft.ifft(tones) * np.sqrt(FFT_LEN)
    return np.tile(short_symbol[:16], 10)


def long_training_field() -> np.ndarray:
    tones = np.zeros(FFT_LEN, dtype=np.complex128)
    fixed_bpsk = np.array(
        [
            1,
            1,
            -1,
            -1,
            1,
            1,
            -1,
            1,
            -1,
            1,
            1,
            1,
            1,
            1,
            1,
            -1,
            -1,
            1,
            1,
            -1,
            1,
            -1,
            1,
            1,
            1,
            1,
            -1,
            1,
            -1,
            -1,
            1,
            1,
            -1,
            1,
            -1,
            1,
            -1,
            -1,
            -1,
            -1,
            -1,
            1,
            1,
            -1,
            -1,
            1,
            -1,
            1,
            -1,
            1,
            1,
            1,
        ],
        dtype=np.complex128,
    )
    used_weights = np.array([WIFI_SUBCARRIER_WEIGHTS[int(carrier)] for carrier in USED_SUBCARRIERS])
    tones[USED_SUBCARRIERS % FFT_LEN] = fixed_bpsk * used_weights
    long_symbol = np.fft.ifft(tones) * np.sqrt(FFT_LEN)
    return np.concatenate((long_symbol[-LTF_CP_LEN:], long_symbol, long_symbol))


def signal_field(rng: np.random.Generator) -> np.ndarray:
    signal_bits = rng.choice(np.array([-1.0, 1.0]), size=DATA_SUBCARRIERS.size).astype(np.complex128)
    return build_ofdm_symbol(signal_bits)


def generate_wifi_packet(
    sample_rate: float,
    rng: np.random.Generator,
    data_symbol_count_range: tuple[int, int] = (4, 24),
) -> tuple[np.ndarray, dict]:
    modulation = rng.choice(SUPPORTED_WIFI_MODULATIONS, p=(0.2, 0.5, 0.3))
    data_symbol_count = int(rng.integers(data_symbol_count_range[0], data_symbol_count_range[1] + 1))
    parts = [short_training_field(), long_training_field(), signal_field(rng)]

    for symbol_index in range(data_symbol_count):
        data_symbols = random_constellation(modulation, DATA_SUBCARRIERS.size, rng)
        polarity = -1.0 if symbol_index % 2 else 1.0
        parts.append(build_ofdm_symbol(data_symbols, pilot_polarity=polarity))

    packet = np.concatenate(parts)
    packet = bandlimit_signal(packet, cutoff_hz=0.42 * sample_rate, sample_rate=sample_rate)
    packet = normalize_rms(packet, target_rms=1.0)
    metadata = {
        "modulation": modulation,
        "data_symbol_count": data_symbol_count,
        "packet_samples": int(packet.size),
        "packet_duration_us": float(packet.size / sample_rate * 1e6),
    }
    return packet, metadata


def apply_channel(
    packet: np.ndarray,
    sample_rate: float,
    noise_sigma: float,
    wifi_snr_db_range: tuple[float, float],
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    tap_count = int(rng.integers(1, 4))
    taps = (rng.normal(size=tap_count) + 1j * rng.normal(size=tap_count)) * np.exp(
        -0.8 * np.arange(tap_count)
    )
    taps /= max(np.linalg.norm(taps), 1e-12)
    shaped = np.convolve(packet, taps, mode="full")[: packet.size]

    freq_offset_hz = float(rng.uniform(-120_000.0, 120_000.0))
    phase_offset_rad = float(rng.uniform(0.0, 2.0 * np.pi))
    samples = np.arange(packet.size, dtype=np.float64)
    shaped *= np.exp(1j * (2.0 * np.pi * freq_offset_hz * samples / sample_rate + phase_offset_rad))

    target_wifi_snr_db = None
    if noise_sigma > 0.0:
        target_wifi_snr_db = float(rng.uniform(*wifi_snr_db_range))
        target_rms = float(noise_sigma * 10.0 ** (target_wifi_snr_db / 20.0))
    else:
        target_rms = float(10.0 ** (rng.uniform(-6.0, -1.5) / 20.0))
    shaped = normalize_rms(shaped, target_rms=target_rms)
    channel_meta = {
        "tap_count": tap_count,
        "carrier_offset_hz": freq_offset_hz,
        "phase_offset_rad": phase_offset_rad,
        "amplitude_scale": target_rms,
        "target_wifi_snr_db": target_wifi_snr_db,
    }
    return shaped.astype(np.complex64), channel_meta


def generate_tone_anomaly(
    duration_samples: int,
    sample_rate: float,
    bandwidth_hz: float,
    capture_bandwidth_hz: float,
    anomaly_rms_db_range: tuple[float, float],
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    center_offset_hz = choose_center_offset_hz(0.0, capture_bandwidth_hz, rng)
    t = np.arange(duration_samples, dtype=np.float64) / sample_rate
    phase0 = rng.uniform(0.0, 2.0 * np.pi)
    signal = np.exp(1j * (2.0 * np.pi * center_offset_hz * t + phase0))
    signal = apply_taper(signal, edge_samples=max(64, duration_samples // 50))
    signal = normalize_rms(
        signal,
        target_rms=10.0 ** (rng.uniform(*anomaly_rms_db_range) / 20.0),
    )
    metadata = {
        "type": "TONE",
        "center_offset_hz": center_offset_hz,
    }
    return signal, metadata


def generate_pulse_anomaly(
    duration_samples: int,
    sample_rate: float,
    bandwidth_hz: float,
    capture_bandwidth_hz: float,
    anomaly_rms_db_range: tuple[float, float],
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    center_offset_hz = choose_center_offset_hz(bandwidth_hz, capture_bandwidth_hz, rng)
    baseband = (
        rng.normal(size=duration_samples) + 1j * rng.normal(size=duration_samples)
    ) / np.sqrt(2.0)
    baseband = bandlimit_signal(baseband, cutoff_hz=bandwidth_hz / 2.0, sample_rate=sample_rate)

    pulse_count = int(rng.integers(3, 8))
    envelope = np.zeros(duration_samples, dtype=np.float64)
    min_width = max(16, int(0.05 * duration_samples / pulse_count))
    max_width = max(min_width + 1, int(0.35 * duration_samples / max(pulse_count, 1)))
    for _ in range(pulse_count):
        width = int(rng.integers(min_width, max_width + 1))
        start = int(rng.integers(0, max(duration_samples - width, 1)))
        pulse_window = np.hanning(width)
        envelope[start : start + width] += pulse_window
    envelope = np.clip(envelope, 0.0, 1.0)

    t = np.arange(duration_samples, dtype=np.float64) / sample_rate
    carrier = np.exp(1j * (2.0 * np.pi * center_offset_hz * t + rng.uniform(0.0, 2.0 * np.pi)))
    signal = baseband * envelope * carrier
    signal = apply_taper(signal, edge_samples=max(64, duration_samples // 50))
    signal = normalize_rms(
        signal,
        target_rms=10.0 ** (rng.uniform(*anomaly_rms_db_range) / 20.0),
    )
    metadata = {
        "type": "PULSE",
        "occupied_bandwidth_hz": float(bandwidth_hz),
        "center_offset_hz": center_offset_hz,
        "pulse_count": pulse_count,
    }
    return signal, metadata


def generate_chirp_anomaly(
    duration_samples: int,
    sample_rate: float,
    bandwidth_hz: float,
    capture_bandwidth_hz: float,
    anomaly_rms_db_range: tuple[float, float],
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    duration_s = duration_samples / sample_rate
    center_offset_hz = choose_center_offset_hz(bandwidth_hz, capture_bandwidth_hz, rng)
    direction = rng.choice(np.array([-1.0, 1.0]))
    if direction > 0:
        start_offset_hz = center_offset_hz - 0.5 * bandwidth_hz
        end_offset_hz = center_offset_hz + 0.5 * bandwidth_hz
    else:
        start_offset_hz = center_offset_hz + 0.5 * bandwidth_hz
        end_offset_hz = center_offset_hz - 0.5 * bandwidth_hz

    chirp_rate_hz_per_s = (end_offset_hz - start_offset_hz) / max(duration_s, 1e-12)
    t = np.arange(duration_samples, dtype=np.float64) / sample_rate
    phase0 = rng.uniform(0.0, 2.0 * np.pi)
    phase = 2.0 * np.pi * (start_offset_hz * t + 0.5 * chirp_rate_hz_per_s * t**2) + phase0
    signal = np.exp(1j * phase)
    signal = apply_taper(signal, edge_samples=max(64, duration_samples // 50))
    signal = normalize_rms(
        signal,
        target_rms=10.0 ** (rng.uniform(*anomaly_rms_db_range) / 20.0),
    )
    metadata = {
        "type": "CHIRP",
        "occupied_bandwidth_hz": float(bandwidth_hz),
        "center_offset_hz": center_offset_hz,
        "chirp_start_offset_hz": float(start_offset_hz),
        "chirp_end_offset_hz": float(end_offset_hz),
        "chirp_tuning_rate_hz_per_s": float(chirp_rate_hz_per_s),
    }
    return signal, metadata


def generate_fsk_anomaly(
    duration_samples: int,
    sample_rate: float,
    bandwidth_hz: float,
    capture_bandwidth_hz: float,
    anomaly_rms_db_range: tuple[float, float],
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    center_offset_hz = choose_center_offset_hz(bandwidth_hz, capture_bandwidth_hz, rng)
    tone_count = int(rng.choice(np.array([2, 4])))
    if tone_count == 2:
        tone_offsets = np.array([-0.5 * bandwidth_hz, 0.5 * bandwidth_hz], dtype=np.float64)
    else:
        tone_offsets = np.linspace(-0.5 * bandwidth_hz, 0.5 * bandwidth_hz, tone_count, dtype=np.float64)

    nominal_symbol_samples = int(
        rng.integers(max(256, duration_samples // 48), max(512, duration_samples // 14) + 1)
    )
    nominal_symbol_samples = max(128, min(nominal_symbol_samples, duration_samples))
    symbol_count = max(1, int(np.ceil(duration_samples / nominal_symbol_samples)))
    symbols = rng.integers(0, tone_count, size=symbol_count)
    instantaneous_frequency = np.repeat(tone_offsets[symbols], nominal_symbol_samples)[:duration_samples]
    instantaneous_frequency = center_offset_hz + instantaneous_frequency

    phase0 = rng.uniform(0.0, 2.0 * np.pi)
    phase = phase0 + 2.0 * np.pi * np.cumsum(instantaneous_frequency) / sample_rate
    signal = np.exp(1j * phase)
    signal = apply_taper(signal, edge_samples=max(64, duration_samples // 50))
    signal = normalize_rms(
        signal,
        target_rms=10.0 ** (rng.uniform(*anomaly_rms_db_range) / 20.0),
    )
    metadata = {
        "type": "FSK",
        "occupied_bandwidth_hz": float(bandwidth_hz),
        "center_offset_hz": center_offset_hz,
        "tone_count": tone_count,
        "symbol_count": symbol_count,
        "symbol_samples": nominal_symbol_samples,
    }
    return signal, metadata


def generate_ofdm_anomaly(
    duration_samples: int,
    sample_rate: float,
    bandwidth_hz: float,
    capture_bandwidth_hz: float,
    anomaly_rms_db_range: tuple[float, float],
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    center_offset_hz = choose_center_offset_hz(bandwidth_hz, capture_bandwidth_hz, rng)
    fft_len = 256
    cp_len = 32
    subcarrier_spacing_hz = sample_rate / fft_len
    active_pairs = int(np.clip(np.floor((bandwidth_hz / 2.0) / subcarrier_spacing_hz), 4, fft_len // 2 - 3))
    used_subcarriers = np.concatenate(
        (np.arange(-active_pairs, 0, dtype=np.int64), np.arange(1, active_pairs + 1, dtype=np.int64))
    )
    symbol_len = fft_len + cp_len
    symbol_count = max(1, int(np.ceil(duration_samples / symbol_len)))
    parts: list[np.ndarray] = []

    for _ in range(symbol_count):
        freq_domain = np.zeros(fft_len, dtype=np.complex128)
        qpsk_symbols = random_constellation("qpsk", used_subcarriers.size, rng)
        freq_domain[used_subcarriers % fft_len] = qpsk_symbols
        time_domain = np.fft.ifft(freq_domain) * np.sqrt(fft_len)
        parts.append(np.concatenate((time_domain[-cp_len:], time_domain)))

    signal = np.concatenate(parts)[:duration_samples]
    signal = bandlimit_signal(
        signal,
        cutoff_hz=min(0.55 * bandwidth_hz, 0.48 * sample_rate),
        sample_rate=sample_rate,
    )
    t = np.arange(duration_samples, dtype=np.float64) / sample_rate
    signal *= np.exp(1j * (2.0 * np.pi * center_offset_hz * t + rng.uniform(0.0, 2.0 * np.pi)))
    signal = apply_taper(signal, edge_samples=max(64, duration_samples // 50))
    signal = normalize_rms(
        signal,
        target_rms=10.0 ** (rng.uniform(*anomaly_rms_db_range) / 20.0),
    )
    metadata = {
        "type": "OFDM",
        "occupied_bandwidth_hz": float(2.0 * active_pairs * subcarrier_spacing_hz),
        "center_offset_hz": center_offset_hz,
        "fft_length": fft_len,
        "cp_length": cp_len,
        "subcarrier_spacing_hz": float(subcarrier_spacing_hz),
        "active_subcarrier_count": int(used_subcarriers.size),
        "symbol_count": symbol_count,
    }
    return signal, metadata


def generate_comb_anomaly(
    duration_samples: int,
    sample_rate: float,
    bandwidth_hz: float,
    capture_bandwidth_hz: float,
    anomaly_rms_db_range: tuple[float, float],
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    center_offset_hz = choose_center_offset_hz(bandwidth_hz, capture_bandwidth_hz, rng)
    tone_count = int(rng.integers(5, 13))
    tone_offsets = np.linspace(-0.5 * bandwidth_hz, 0.5 * bandwidth_hz, tone_count, dtype=np.float64)
    tone_amplitudes = rng.uniform(0.7, 1.0, size=tone_count)
    tone_phases = rng.uniform(0.0, 2.0 * np.pi, size=tone_count)
    t = np.arange(duration_samples, dtype=np.float64) / sample_rate

    signal = np.zeros(duration_samples, dtype=np.complex128)
    for amplitude, phase0, offset in zip(tone_amplitudes, tone_phases, tone_offsets):
        signal += amplitude * np.exp(1j * (2.0 * np.pi * (center_offset_hz + offset) * t + phase0))

    signal = apply_taper(signal, edge_samples=max(64, duration_samples // 50))
    signal = normalize_rms(
        signal,
        target_rms=10.0 ** (rng.uniform(*anomaly_rms_db_range) / 20.0),
    )
    metadata = {
        "type": "COMB",
        "occupied_bandwidth_hz": float(bandwidth_hz),
        "center_offset_hz": center_offset_hz,
        "tone_count": tone_count,
        "tone_spacing_hz": float(bandwidth_hz / max(tone_count - 1, 1)),
    }
    return signal, metadata


def generate_anomaly_signal(
    anomaly_type: str,
    duration_samples: int,
    sample_rate: float,
    bandwidth_hz: float,
    capture_bandwidth_hz: float,
    anomaly_rms_db_range: tuple[float, float],
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    if anomaly_type == "tone":
        return generate_tone_anomaly(
            duration_samples=duration_samples,
            sample_rate=sample_rate,
            bandwidth_hz=bandwidth_hz,
            capture_bandwidth_hz=capture_bandwidth_hz,
            anomaly_rms_db_range=anomaly_rms_db_range,
            rng=rng,
        )
    if anomaly_type == "pulse":
        return generate_pulse_anomaly(
            duration_samples=duration_samples,
            sample_rate=sample_rate,
            bandwidth_hz=bandwidth_hz,
            capture_bandwidth_hz=capture_bandwidth_hz,
            anomaly_rms_db_range=anomaly_rms_db_range,
            rng=rng,
        )
    if anomaly_type == "chirp":
        return generate_chirp_anomaly(
            duration_samples=duration_samples,
            sample_rate=sample_rate,
            bandwidth_hz=bandwidth_hz,
            capture_bandwidth_hz=capture_bandwidth_hz,
            anomaly_rms_db_range=anomaly_rms_db_range,
            rng=rng,
        )
    if anomaly_type == "fsk":
        return generate_fsk_anomaly(
            duration_samples=duration_samples,
            sample_rate=sample_rate,
            bandwidth_hz=bandwidth_hz,
            capture_bandwidth_hz=capture_bandwidth_hz,
            anomaly_rms_db_range=anomaly_rms_db_range,
            rng=rng,
        )
    if anomaly_type == "ofdm":
        return generate_ofdm_anomaly(
            duration_samples=duration_samples,
            sample_rate=sample_rate,
            bandwidth_hz=bandwidth_hz,
            capture_bandwidth_hz=capture_bandwidth_hz,
            anomaly_rms_db_range=anomaly_rms_db_range,
            rng=rng,
        )
    if anomaly_type == "comb":
        return generate_comb_anomaly(
            duration_samples=duration_samples,
            sample_rate=sample_rate,
            bandwidth_hz=bandwidth_hz,
            capture_bandwidth_hz=capture_bandwidth_hz,
            anomaly_rms_db_range=anomaly_rms_db_range,
            rng=rng,
        )
    raise ValueError(f"Unsupported anomaly type: {anomaly_type}")


def add_wifi_background(
    capture: np.ndarray,
    sample_rate: float,
    noise_sigma: float,
    wifi_snr_db_range: tuple[float, float],
    forced_traffic_profile: str | None,
    rng: np.random.Generator,
) -> tuple[list[dict], list[float]]:
    packets = []
    packet_rms_values: list[float] = []
    total_samples = capture.size
    minimum_packet_samples = 640
    traffic_profile = forced_traffic_profile or str(
        rng.choice(np.array(["low", "medium", "high"]), p=[0.35, 0.45, 0.20])
    )
    if traffic_profile == "low":
        busy_burst_count = int(rng.integers(1, 3))
        idle_time_range_us = (80.0, 500.0)
        symbol_count_range = (3, 12)
    elif traffic_profile == "medium":
        busy_burst_count = int(rng.integers(1, 4))
        idle_time_range_us = (50.0, 250.0)
        symbol_count_range = (4, 20)
    else:
        busy_burst_count = int(rng.integers(2, 5))
        idle_time_range_us = (30.0, 150.0)
        symbol_count_range = (6, 28)

    min_busy_len = max(minimum_packet_samples * 2, int(0.12 * total_samples))
    max_busy_len = max(min_busy_len + 1, int(0.45 * total_samples))
    busy_windows: list[tuple[int, int]] = []
    for _ in range(busy_burst_count):
        window_len = int(rng.integers(min_busy_len, max_busy_len + 1))
        start = int(rng.integers(0, max(total_samples - window_len + 1, 1)))
        end = min(total_samples, start + window_len)
        busy_windows.append((start, end))
    busy_windows.sort()

    merged_windows: list[tuple[int, int]] = []
    for start, end in busy_windows:
        if not merged_windows or start > merged_windows[-1][1]:
            merged_windows.append((start, end))
        else:
            merged_windows[-1] = (merged_windows[-1][0], max(merged_windows[-1][1], end))

    for window_start, window_end in merged_windows:
        cursor = window_start
        while cursor + minimum_packet_samples < window_end:
            idle_time_us = float(rng.uniform(*idle_time_range_us))
            start = cursor + int(idle_time_us * 1e-6 * sample_rate)
            if start + minimum_packet_samples >= window_end:
                break

            packet, packet_meta = generate_wifi_packet(
                sample_rate,
                rng,
                data_symbol_count_range=symbol_count_range,
            )
            packet, channel_meta = apply_channel(packet, sample_rate, noise_sigma, wifi_snr_db_range, rng)
            end = start + packet.size
            if end >= window_end:
                break

            capture[start:end] += packet
            packet_rms = float(np.sqrt(np.mean(np.abs(packet) ** 2)))
            packet_meta.update(channel_meta)
            packet_meta.update(
                {
                    "start_sample": int(start),
                    "end_sample": int(end),
                    "start_time_us": float(start / sample_rate * 1e6),
                    "duration_us": float(packet.size / sample_rate * 1e6),
                    "approx_snr_db": approx_snr_db(packet_rms, noise_sigma),
                }
            )
            packets.append(packet_meta)
            packet_rms_values.append(packet_rms)
            cursor = end
    if packets:
        for packet in packets:
            packet["traffic_profile"] = traffic_profile
            packet["busy_window_count"] = len(merged_windows)
    return packets, packet_rms_values


def generate_capture(
    duration_ms: float,
    sample_rate: float,
    center_frequency: float,
    bandwidth: float,
    anomaly_duration_min_ms: float,
    anomaly_duration_max_ms: float,
    anomaly_bandwidth_min_mhz: float,
    anomaly_bandwidth_max_mhz: float,
    anomaly_rms_db_range: tuple[float, float],
    wifi_snr_db_range: tuple[float, float],
    anomaly_to_wifi_ratio_db_range: tuple[float, float] | None,
    include_anomalies: bool,
    forced_anomaly_type: str | None,
    include_wifi_background: bool,
    include_background_noise: bool,
    background_noise_model: str,
    enable_blob_noise: bool,
    forced_traffic_profile: str | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    total_samples = int(round(duration_ms * 1e-3 * sample_rate))
    noise_metadata: dict[str, float | int | list[float] | str] = {"background_noise_model": "disabled"}
    if include_background_noise:
        validate_background_noise_model(background_noise_model, enable_blob_noise)
        noise_floor_dbfs = float(rng.uniform(-36.0, -28.0))
        noise_sigma = 10.0 ** (noise_floor_dbfs / 20.0)
        capture, noise_metadata = generate_background_noise(
            sample_count=total_samples,
            sample_rate=sample_rate,
            noise_sigma=noise_sigma,
            model=background_noise_model,
            rng=rng,
        )
    else:
        noise_floor_dbfs = None
        noise_sigma = 0.0
        capture = np.zeros(total_samples, dtype=np.complex64)

    packets = []
    packet_rms_values: list[float] = []
    if include_wifi_background:
        packets, packet_rms_values = add_wifi_background(
            capture=capture,
            sample_rate=sample_rate,
            noise_sigma=noise_sigma,
            wifi_snr_db_range=wifi_snr_db_range,
            forced_traffic_profile=forced_traffic_profile,
            rng=rng,
        )

    anomalies = []
    if include_anomalies:
        anomaly_type = forced_anomaly_type or str(rng.choice(np.array(ANOMALY_TYPES)))
        max_duration_ms = min(anomaly_duration_max_ms, duration_ms)
        min_duration_ms = min(anomaly_duration_min_ms, max_duration_ms)
        anomaly_duration_ms = float(rng.uniform(min_duration_ms, max_duration_ms))
        anomaly_duration_samples = max(1, int(round(anomaly_duration_ms * 1e-3 * sample_rate)))
        anomaly_duration_samples = min(anomaly_duration_samples, total_samples)
        bandwidth_hz = 0.0
        if anomaly_type != "tone":
            bandwidth_hz = float(
                rng.uniform(anomaly_bandwidth_min_mhz * 1e6, anomaly_bandwidth_max_mhz * 1e6)
            )
        start = int(rng.integers(0, max(total_samples - anomaly_duration_samples + 1, 1)))
        signal, anomaly_meta = generate_anomaly_signal(
            anomaly_type=anomaly_type,
            duration_samples=anomaly_duration_samples,
            sample_rate=sample_rate,
            bandwidth_hz=bandwidth_hz,
            capture_bandwidth_hz=bandwidth,
            anomaly_rms_db_range=anomaly_rms_db_range,
            rng=rng,
        )
        target_anomaly_to_wifi_ratio_db = None
        if anomaly_to_wifi_ratio_db_range is not None and packet_rms_values:
            reference_wifi_rms = float(np.median(packet_rms_values))
            target_anomaly_to_wifi_ratio_db = float(rng.uniform(*anomaly_to_wifi_ratio_db_range))
            target_anomaly_rms = reference_wifi_rms * 10.0 ** (target_anomaly_to_wifi_ratio_db / 20.0)
            signal = normalize_rms(signal, target_rms=target_anomaly_rms)
        end = start + anomaly_duration_samples
        capture[start:end] += signal
        anomaly_rms = float(np.sqrt(np.mean(np.abs(signal) ** 2)))
        anomaly_meta.update(
            {
                "start_sample": int(start),
                "end_sample": int(end),
                "start_time_ms": float(start / sample_rate * 1e3),
                "duration_ms": float(anomaly_duration_samples / sample_rate * 1e3),
                "approx_snr_db": approx_snr_db(anomaly_rms, noise_sigma),
                "target_anomaly_to_wifi_ratio_db": target_anomaly_to_wifi_ratio_db,
                "anomaly_rms": anomaly_rms,
            }
        )
        anomalies.append(anomaly_meta)

    capture = capture.astype(np.complex64)

    capture_rms = float(np.sqrt(np.mean(np.abs(capture) ** 2)))
    max_abs_i = float(np.max(np.abs(capture.real)))
    max_abs_q = float(np.max(np.abs(capture.imag)))
    component_peak = max(max_abs_i, max_abs_q)
    capture_rms_db_rel_unit = float(20.0 * np.log10(max(capture_rms, 1e-12)))
    pre_clip_component_peak_dbfs = float(20.0 * np.log10(max(component_peak, 1e-12)))
    would_clip = bool(component_peak > 1.0)

    metadata = {
        "center_frequency_hz": float(center_frequency),
        "bandwidth_hz": float(bandwidth),
        "sample_rate_hz": float(sample_rate),
        "duration_ms": float(duration_ms),
        "num_samples": int(total_samples),
        "noise_floor_dbfs": noise_floor_dbfs,
        "background_noise_model": noise_metadata.get("background_noise_model"),
        "background_noise_metadata": noise_metadata,
        "packet_count": len(packets),
        "packets": packets,
        "include_wifi_background": include_wifi_background,
        "include_background_noise": include_background_noise,
        "blob_noise_enabled": bool(enable_blob_noise),
        "wifi_snr_db_range": [float(wifi_snr_db_range[0]), float(wifi_snr_db_range[1])],
        "anomaly_to_wifi_ratio_db_range": None
        if anomaly_to_wifi_ratio_db_range is None
        else [float(anomaly_to_wifi_ratio_db_range[0]), float(anomaly_to_wifi_ratio_db_range[1])],
        "capture_type": "anomaly" if include_anomalies else "normal",
        "anomaly_count": len(anomalies),
        "anomalies": anomalies,
        "capture_rms_db_rel_unit": capture_rms_db_rel_unit,
        "pre_clip_component_peak_dbfs": pre_clip_component_peak_dbfs,
        "max_abs_i": max_abs_i,
        "max_abs_q": max_abs_q,
        "would_clip": would_clip,
    }
    return capture, metadata


def save_capture(
    iq: np.ndarray,
    metadata: dict,
    output_dir: Path,
    index: int,
    output_format: str,
    capture_type: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{capture_type}_capture_{index:04d}"
    if output_format == "npy":
        np.save(output_dir / f"{stem}.npy", iq.astype(np.complex64))
    else:
        interleaved = np.empty(iq.size * 2, dtype=np.float32)
        interleaved[0::2] = iq.real.astype(np.float32)
        interleaved[1::2] = iq.imag.astype(np.float32)
        interleaved.tofile(output_dir / f"{stem}.bin")

    with (output_dir / f"{stem}.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    selected_types = ["normal", "anomaly"] if args.capture_mode == "both" else [args.capture_mode]

    for capture_type in selected_types:
        for index in range(args.count):
            iq, metadata = generate_capture(
                duration_ms=args.duration_ms,
                sample_rate=args.sample_rate,
                center_frequency=args.center_frequency,
                bandwidth=args.bandwidth,
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
                include_anomalies=(capture_type == "anomaly"),
                forced_anomaly_type=None,
                include_wifi_background=not args.no_wifi_background,
                include_background_noise=not args.no_background_noise,
                background_noise_model=args.background_noise_model,
                enable_blob_noise=args.enable_blob_noise,
                forced_traffic_profile=None,
                rng=rng,
            )
            metadata["capture_index"] = index
            metadata["output_format"] = args.format
            save_capture(iq, metadata, args.output_dir, index, args.format, capture_type)

    print(
        f"Generated {args.count} {args.capture_mode} WiFi IQ capture set(s) in '{args.output_dir}' "
        f"at {args.center_frequency / 1e9:.6f} GHz."
    )


if __name__ == "__main__":
    main()
