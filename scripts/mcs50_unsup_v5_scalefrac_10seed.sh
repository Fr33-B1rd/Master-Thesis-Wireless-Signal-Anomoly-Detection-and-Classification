#!/bin/bash
# v5: scale-free anchor_fraction threshold = 2.5 x (n_anchor_total / n_total).
# Physical meaning: reject cluster with anchor density more than 2.5x the
# null expectation. Auto-scales with anchor count / buffer size. No oracle,
# no dataset-specific magic number (scale=2.5 is just 'a few sigma above null').
# Disables hypergeometric path.
set -e
cd "$(dirname "$0")/.."

DATASET="C:/Users/Zhuoer/code/thesis/Datasets/SAS/rf_dataset_packed_blob_impulsive_snr14_20_sirm4_0"
PATCHCORE="./outputs/sas_blob_impulsive_snr14_20_patchcore_wide_resnet50_2_256_zscore_allpatches"
PYEXE="/c/Users/Zhuoer/anaconda3/envs/IIS/python.exe"

for seed in 42 43 44 45 46 47 48 49 50 51; do
  OUT="./outputs/sas_blob_impulsive_mcs50_unsup_v5_scalefrac_seed${seed}"
  echo "=== v5 scalefrac seed=${seed} -> ${OUT} ==="
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
    --cluster-anchor-fraction-scale 2.5 \
    --seed "$seed"
done
echo "=== v5 10-seed done ==="
