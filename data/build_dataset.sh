#!/bin/bash
# One-click SAS RF dataset generation.
#
# Default config produces the canonical reference dataset:
#   rf_dataset_packed_impulsive_only_snr8_14_sirm8_4
# which is the dataset consumed by the main Anomaly_Localization pipeline
# (Stage 1 PatchCore + Stage 2 ConvNeXtV2 open-world).
#
# To generate a different variant: copy this script, edit the CONFIG block
# below, run.
#
# Idempotency: each step skips itself if its output already exists.
# To force regeneration of a step, delete its output directory/files first.
set -e
cd "$(dirname "$0")"

# =============================================================================
# CONFIG — edit here for a different dataset variant
# =============================================================================

# Tag used in the output directory names. Convention:
#   <bg_noise_only>_snr<wifi_min>_<wifi_max>_sirm<|sir_min|>_<|sir_max|>
TAG="impulsive_only_snr8_14_sirm8_4"
SEED=2026

PYEXE="/c/Users/Zhuoer/anaconda3/envs/IIS/python.exe"

# --- Signal / noise model ----------------------------------------------------
BG_NOISE_MODEL="impulsive_mix"   # comma-separated; "impulsive_mix" means impulsive-only
WIFI_SNR_MIN=8.0
WIFI_SNR_MAX=14.0
SIR_MIN=-8.0                     # anomaly-to-wifi-ratio min (dB, negative = anomaly weaker than wifi)
SIR_MAX=-4.0
# Blob noise is OFF by default (we omit --enable-blob-noise below).

# --- RF capture --------------------------------------------------------------
SAMPLE_RATE=20000000             # 20 MHz
DURATION_MS=20.0                 # 20 ms per capture → 400 000 IQ samples
NFFT=1024
IMAGE_SIZE=256                   # spectrogram resized to 256×256

# --- Spectrogram normalization ----------------------------------------------
SPEC_NORM=fixed_db
DB_MIN=-85.0
DB_MAX=-10.0

# --- 3-channel phase-channel gating (Step 3) --------------------------------
GATE_MODE=soft                   # "soft" = legacy |X|/(|X|+eps*max|X|); "hard" = fixed dB threshold
MASK_EPSILON_RATIO=0.05

# --- Output paths (relative to this directory) ------------------------------
FULL_DIR="rf_dataset_full_${TAG}"
PACKED_DIR="rf_dataset_packed_${TAG}"

# =============================================================================
# Pipeline
# =============================================================================

echo "================================================================"
echo "  SAS dataset generation"
echo "  TAG    = ${TAG}"
echo "  SEED   = ${SEED}"
echo "  OUTPUT = ${PACKED_DIR}/  (Step 3 = main consumer)"
echo "================================================================"
echo

# --- Step 1: per-sample IQ + spectrograms + masks + JSON metadata -----------
if [ -d "${FULL_DIR}/train" ] && [ -d "${FULL_DIR}/test" ]; then
    echo "[1/3] Skipping Step 1: ${FULL_DIR}/{train,test}/ already exist."
    echo "       Delete ${FULL_DIR}/ to force regeneration."
else
    echo "[1/3] Step 1 — build_rf_dataset.py (per-sample IQ + spec + mask + json) ..."
    "$PYEXE" build_rf_dataset.py \
        --mode full \
        --output-dir "${FULL_DIR}" \
        --background-noise-models "${BG_NOISE_MODEL}" \
        --wifi-snr-min-db "${WIFI_SNR_MIN}" \
        --wifi-snr-max-db "${WIFI_SNR_MAX}" \
        --anomaly-to-wifi-ratio-min-db "${SIR_MIN}" \
        --anomaly-to-wifi-ratio-max-db "${SIR_MAX}" \
        --sample-rate "${SAMPLE_RATE}" \
        --duration-ms "${DURATION_MS}" \
        --nfft "${NFFT}" \
        --image-size "${IMAGE_SIZE}" \
        --spectrogram-normalization "${SPEC_NORM}" \
        --spectrogram-db-min "${DB_MIN}" \
        --spectrogram-db-max "${DB_MAX}" \
        --seed "${SEED}"
fi
echo

# --- Step 2: pack per-sample files into split-level numpy bundles ----------
if [ -f "${PACKED_DIR}/train_iq.npy" ] && [ -f "${PACKED_DIR}/test_iq.npy" ]; then
    echo "[2/3] Skipping Step 2: ${PACKED_DIR}/{train,test}_iq.npy already exist."
    echo "       Delete those .npy files to force regeneration."
else
    echo "[2/3] Step 2 — pack_rf_dataset.py (split-level npy + manifest.csv) ..."
    "$PYEXE" pack_rf_dataset.py \
        --input-dir "${FULL_DIR}" \
        --output-dir "${PACKED_DIR}"
fi
echo

# --- Step 3: 3-channel spectrograms (mag, IF, GD) + channel_stats.json -----
if [ -f "${PACKED_DIR}/test_spectrogram_3c.npy" ] && [ -f "${PACKED_DIR}/channel_stats.json" ]; then
    echo "[3/3] Skipping Step 3: ${PACKED_DIR}/test_spectrogram_3c.npy + channel_stats.json already exist."
    echo "       Delete those files to force regeneration."
else
    echo "[3/3] Step 3 — build_rf_3channel.py (mag / IF / GD + channel_stats) ..."
    "$PYEXE" build_rf_3channel.py \
        --packed-dir "${PACKED_DIR}" \
        --gate-mode "${GATE_MODE}" \
        --mask-epsilon-ratio "${MASK_EPSILON_RATIO}" \
        --sample-rate "${SAMPLE_RATE}" \
        --nfft "${NFFT}" \
        --image-size "${IMAGE_SIZE}" \
        --db-min "${DB_MIN}" \
        --db-max "${DB_MAX}"
fi
echo

echo "================================================================"
echo "  Done."
echo "  Use this in the model-side pipeline:"
echo "    --dataset-dir $(pwd)/${PACKED_DIR}"
echo "================================================================"
