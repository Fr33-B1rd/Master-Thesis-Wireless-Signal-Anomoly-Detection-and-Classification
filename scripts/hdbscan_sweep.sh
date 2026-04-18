#!/bin/bash
# Sweep min_cluster_size x cluster_selection_method for HDBSCAN fragmentation diagnosis.
# Runs with --cluster-only (skips retrain) on snr14_20 seed 42.
set -e
cd "$(dirname "$0")/.."

DATASET="C:/Users/Zhuoer/code/thesis/Datasets/SAS/rf_dataset_packed_blob_impulsive_snr14_20_sirm4_0"
PATCHCORE="./outputs/sas_blob_impulsive_snr14_20_patchcore_wide_resnet50_2_256_zscore_allpatches"
PYEXE="/c/Users/Zhuoer/anaconda3/envs/IIS/python.exe"

for mcs in 30 50 80; do
  for method in eom leaf; do
    OUT="./outputs/hdbscan_sweep/mcs${mcs}_${method}"
    echo "=== running mcs=${mcs} method=${method} -> ${OUT} ==="
    "$PYEXE" simulate_sas_convnextv2_open_world.py \
      --dataset-dir "$DATASET" \
      --output-dir "$OUT" \
      --patchcore-output-dir "$PATCHCORE" \
      --input-mode roi \
      --cluster-dim-reduction umap \
      --cluster-min-cluster-size "$mcs" \
      --cluster-selection-method "$method" \
      --cluster-accept-min-size 20 \
      --cluster-only \
      --seed 42
  done
done
echo "=== sweep done ==="
