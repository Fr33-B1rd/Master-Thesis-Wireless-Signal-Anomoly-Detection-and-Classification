#!/bin/bash
# v4 prototype: replace --cluster-max-anchor-fraction with per-class
# hypergeometric null test (--cluster-anchor-null-alpha 0.01, Bonferroni
# over known classes). Seed 42 only for quick look.
set -e
cd "$(dirname "$0")/.."

DATASET="C:/Users/Zhuoer/code/thesis/Datasets/SAS/rf_dataset_packed_blob_impulsive_snr14_20_sirm4_0"
PATCHCORE="./outputs/sas_blob_impulsive_snr14_20_patchcore_wide_resnet50_2_256_zscore_allpatches"
PYEXE="/c/Users/Zhuoer/anaconda3/envs/IIS/python.exe"

seed=42
OUT="./outputs/sas_blob_impulsive_mcs50_unsup_v4_hypergeom_seed${seed}"
echo "=== v4 hypergeom prototype seed=${seed} -> ${OUT} ==="
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
  --cluster-anchor-null-alpha 0.01 \
  --seed "$seed"
echo "=== v4 prototype seed=${seed} done ==="
