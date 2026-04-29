# IAD Dataset

[16QAM-Train-Test]

Download Link: https://drive.google.com/file/d/1ovDYV13R_EWmYEVk_92hATxrHXeOeBgp/view?usp=sharing


[QPSK-Train-Test]

Download Link: https://drive.google.com/file/d/1YW8pwe7p9YvqxR3qiod0rQUuJtaTyR_7/view?usp=sharing


[CHIRP-Train-Test]

Download Link: https://drive.google.com/file/d/1sFAPuycJLGzYibOSEh-xwy3JZnK2MUgQ/view?usp=sharing


[GMSK-Train-Test]

Download Link: https://drive.google.com/file/d/1ZNpTWqhMY1_XVCz_mVzFkGAUfWBRbA_V/view?usp=sharing

Note: this is the dataset originally created by the authors of this repository: https://github.com/QXSLAB/vae_ism_ano.git

# Synthetic Anomaly Signal (SAS) Dataset Generation

This directory holds the simulator + tooling that produces the **packed RF spectrogram datasets** consumed by the `Anomaly_Localization` repository. It is independent of the model code: nothing here imports from the model repo, and the model repo never modifies anything here.

The reference dataset is `rf_dataset_packed_impulsive_only_snr8_14_sirm8_4`. To reproduce it, run:

```bash
bash build_dataset.sh
```


## Contents

```
data/
├── dataset.md                      ← this file
├── build_dataset.sh                ← shell script for generating SAS dataset
│
├── generate_wifi_iq.py             ← IQ simulator (library; imported by Step 1)
├── build_rf_dataset.py             ← Step 1: per-sample IQ + spec + mask + json
├── pack_rf_dataset.py              ← Step 2: pack per-sample → split-level npy
├── build_rf_3channel.py            ← Step 3: 3-channel spec + channel_stats
│
├── build_multi_anomaly_testset.py  ← (utility) alt test set with multi-anomaly captures
├── regenerate_masks.py             ← (utility) re-derive masks from existing JSON metadata
├── _smoke_verify_mask.py           ← (utility) mask validation smoke test
└── plot_iq_capture.py              ← (utility) IQ visualization
```

## Pipeline overview

```
generate_wifi_iq.py    (library — IQ simulator, anomaly + WiFi + noise)
        ↑
        │ from generate_wifi_iq import generate_capture, ANOMALY_TYPES, ...
        │
build_rf_dataset.py    Step 1: --output-dir rf_dataset_full_<TAG>/
        │              writes per-sample .npy/.json/.png + masks
        │              under {train,test}/<label>/
        ↓
pack_rf_dataset.py     Step 2: --input-dir rf_dataset_full_<TAG>/
        │                      --output-dir rf_dataset_packed_<TAG>/
        │              concatenates per-sample files into split-level
        │              numpy bundles (train_iq.npy, test_iq.npy, …)
        ↓
build_rf_3channel.py   Step 3: --packed-dir rf_dataset_packed_<TAG>/
        │              writes *_spectrogram_3c.npy + channel_stats.json
        ↓
   rf_dataset_packed_<TAG>/      ← consumed by --dataset-dir in model scripts
```

## What each step produces

### Step 1 — `build_rf_dataset.py` (per-sample files)

Writes one directory tree, organized as `<output-dir>/{train,test}/<label>/<sample>.{npy,json,png}` plus `<sample>_mask.npy`. Generates ~10000 train + 6000 test captures across 7 labels (`normal`, `chirp`, `pulse`, `tone`, `fsk`, `ofdm`, `comb`).

Each capture is 20 ms of complex IQ at 20 MHz (= 400 000 samples), composed of:
- WiFi 802.11 OFDM background traffic (BPSK / QPSK / 16QAM packets at random SNR within the configured range)
- One injected anomaly drawn from `ANOMALY_TYPES` (in non-`normal` classes only)
- Background noise drawn from one of `BACKGROUND_NOISE_MODELS` (the canonical config uses `impulsive_mix` only)

The `--seed` flag (default 2026) controls **all** randomness in this step: anomaly draws, packet sequencing, noise realizations, SNR samples.

### Step 2 — `pack_rf_dataset.py` (packed numpy)

Pure I/O step, **no random component**. Reads everything Step 1 wrote and produces:
- `train_iq.npy`, `test_iq.npy` — complex64, shape `(N, 400_000)`
- `train_spectrogram.npy`, `test_spectrogram.npy` — uint8 `[0, 255]`, shape `(N, 256, 256)`
- `train_labels.npy`, `test_labels.npy` — U16 strings
- `train_mask.npy`, `test_mask.npy` — GT anomaly masks
- `train_manifest.csv`, `test_manifest.csv` — per-sample metadata path index

This is the format the model repo's `dataio.sas.load_sas_bundle` reads.

### Step 3 — `build_rf_3channel.py` (3-channel + stats)

Recomputes STFT on the packed IQ to produce 3 channels per sample:
- **Ch0** `log_mag` — dB-normalized magnitude, range `[0, 1]` (same content as Step 2's uint8 `*_spectrogram.npy`, but float16)
- **Ch1** `IF` — instantaneous frequency, `∂φ/∂t / π`
- **Ch2** `GD` — group delay, `−∂φ/∂f / π`

Phase channels (Ch1, Ch2) are masked by a soft energy gate so low-amplitude bins don't inject phase noise:

```
mask = |X| / (|X| + ε · peak|X|)        with ε = 0.05 (--mask-epsilon-ratio)
```

The `ε = 0.05` value is **support-adaptive** — its effective threshold ranges from ~4σ to ~10σ across signal classes, not a fixed CFAR. (Detailed numerical justification lives in the `_archive/` history of the model repo.)

Outputs:
- `train_spectrogram_3c.npy`, `test_spectrogram_3c.npy` — float16, shape `(N, 3, H, W)`
- `channel_stats.json` — per-channel mean/std fit on the train split, used by the model's `ThreeChannelTransform`

## Customization

The dataset name encodes its key parameters:

| Token | Meaning | CLI flag(s) in Step 1 |
|---|---|---|
| `impulsive_only` | only `impulsive_mix` background noise; blob noise off | `--background-noise-models impulsive_mix` (omit `--enable-blob-noise`) |
| `snr8_14` | wifi SNR drawn from `[8.0, 14.0]` dB | `--wifi-snr-min-db 8.0 --wifi-snr-max-db 14.0` |
| `sirm8_4` | anomaly-to-wifi-ratio drawn from `[-8.0, -4.0]` dB | `--anomaly-to-wifi-ratio-min-db -8.0 --anomaly-to-wifi-ratio-max-db -4.0` |

To make a new variant (e.g., harder SNR `snr2_8`):

1. **Copy** `build_dataset.sh` to a new name (e.g., `build_dataset_snr2_8.sh`).
2. **Edit** the `CONFIG` block at the top:
   ```bash
   TAG="impulsive_only_snr2_8_sirm12_8"
   WIFI_SNR_MIN=2.0
   WIFI_SNR_MAX=8.0
   SIR_MIN=-12.0
   SIR_MAX=-8.0
   ```
3. **Run** the new script. Steps 1, 2, 3 will populate fresh `rf_dataset_{full,packed}_<new_TAG>/` directories without touching existing variants.

`build_dataset.sh` is **idempotent**: each step skips itself if its outputs already exist, so re-running after a partial generation just resumes from where it stopped. To force regeneration of a step, delete the relevant output directory or files first.

## Known utility scripts (not on main path)

These four files live in this directory but are not part of the canonical generation pipeline:

| Script | Purpose | When to use |
|---|---|---|
| `build_multi_anomaly_testset.py` | Generates an alternate test set with **multiple anomalies per capture** | Some `_archive/` experiments in the model repo expect this; the main pipeline does not |
| `regenerate_masks.py` | Reads the per-sample `.json` metadata in an existing `rf_dataset_full_<TAG>/`, re-derives `*_mask.npy` in place, and re-packs only the `*_mask.npy` arrays into `rf_dataset_packed_<TAG>/` (IQ / spectrograms / labels untouched) | When the mask format/definition changes and you want to update existing datasets without re-running the simulator (saves ~30 min) |
| `_smoke_verify_mask.py` | Loads a single sample's mask + spectrogram and overlays them | Manual debugging |
| `plot_iq_capture.py` | Visualizes a single IQ capture (time, freq, constellation) | Paper figures / inspection |

## Reproduction caveats

- **Determinism**: Step 1 is bit-exact reproducible *only* given the same `--seed` **and** the same numpy/scipy versions. The model repo's `requirements-frozen.txt` pins these (numpy 1.26.4, scipy 1.13.1). Different scipy versions can produce different `signal.stft` output at the bit level even with the same inputs.
- **Output size**: A full default dataset (`rf_dataset_full_*/`) is ~25 GB on disk; the packed version is ~16 GB; the 3-channel float16 add-ons are another ~6 GB. Plan storage accordingly.
- **Default seed (2026)** matches the dataset that produced the reference numbers. Changing `SEED` will produce a different dataset that **will not** match those reference numbers.

## Quick reference: recovering generation parameters from a sample

If you have an existing dataset and want to know what flags produced it:

```python
import json
with open("rf_dataset_full_<TAG>/test/chirp/chirp_00000.json") as f:
    m = json.load(f)
print(m["wifi_snr_db_range"])               # [8.0, 14.0]
print(m["anomaly_to_wifi_ratio_db_range"])  # [-8.0, -4.0]
print(m["background_noise_models_pool"])    # ["impulsive_mix"]
print(m["blob_noise_enabled"])              # False
print(m["spectrogram_db_range"])            # [-85.0, -10.0]
print(m["sample_rate_hz"], m["duration_ms"], m["stft_nfft"], m["spectrogram_size"])
```

Every per-sample `.json` carries the full generation context, so a dataset is always self-describing.
