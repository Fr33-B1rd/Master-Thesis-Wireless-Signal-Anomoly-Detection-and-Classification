#!/bin/bash
# Held-out cross-SNR validation of v5 scale-free anchor_fraction rule.
# scale=2.0 frozen from snr14_20 seed 42 sweep. Evaluated once on snr8_14
# and snr2_8 datasets, 10 seeds each. No re-tuning per dataset.
set -e
cd "$(dirname "$0")/.."

PYEXE="/c/Users/Zhuoer/anaconda3/envs/IIS/python.exe"
SCALE=2.0

run_dataset () {
  local tag=$1
  local dataset=$2
  local patchcore=$3
  for seed in 42 43 44 45 46 47 48 49 50 51; do
    OUT="./outputs/sas_blob_impulsive_${tag}_v5_scale2.0_seed${seed}"
    echo "=== ${tag} seed=${seed} -> ${OUT} ==="
    "$PYEXE" simulate_sas_convnextv2_open_world.py \
      --dataset-dir "$dataset" \
      --output-dir "$OUT" \
      --patchcore-output-dir "$patchcore" \
      --input-mode roi \
      --cluster-dim-reduction umap \
      --cluster-min-cluster-size 50 \
      --cluster-selection-method eom \
      --cluster-accept-min-size 20 \
      --cluster-accept-min-prob 0.5 \
      --cluster-anchor-fraction-scale "$SCALE" \
      --seed "$seed"
  done
}

run_dataset snr8_14 \
  "C:/Users/Zhuoer/code/thesis/Datasets/SAS/rf_dataset_packed_blob_impulsive_snr8_14_sirm8_4" \
  "./outputs/sas_blob_impulsive_snr8_14_patchcore_wide_resnet50_2_256_zscore_allpatches"

run_dataset snr2_8 \
  "C:/Users/Zhuoer/code/thesis/Datasets/SAS/rf_dataset_packed_blob_impulsive_snr2_8_sirm12_8" \
  "./outputs/sas_blob_impulsive_snr2_8_patchcore_wide_resnet50_2_256_zscore_allpatches"

echo "=== cross-SNR held-out done ==="
