#!/bin/bash
# SpecAugment smoke test on v6 seed 42.
set -e
cd "$(dirname "$0")/.."

DATASET="C:/Users/Zhuoer/code/thesis/Datasets/SAS/rf_dataset_packed_blob_impulsive_snr14_20_sirm4_0"
PATCHCORE="./outputs/sas_blob_impulsive_snr14_20_patchcore_wide_resnet50_2_256_zscore_allpatches"
PYEXE="/c/Users/Zhuoer/anaconda3/envs/IIS/python.exe"

OUT="./outputs/sas_blob_impulsive_v6_specaug_seed42"
echo "=== v6 specaug seed=42 -> ${OUT} ==="
"$PYEXE" simulate_sas_convnextv2_open_world.py \
  --dataset-dir "$DATASET" \
  --output-dir "$OUT" \
  --patchcore-output-dir "$PATCHCORE" \
  --input-mode roi \
  --cluster-dim-reduction umap \
  --cluster-min-cluster-size 50 \
  --cluster-selection-method eom \
  --cluster-accept-min-size 20 \
  --cluster-accept-min-prob 0.5 \
  --cluster-anchor-quantile-alpha 0.05 \
  --seed 42
echo "=== done ==="
