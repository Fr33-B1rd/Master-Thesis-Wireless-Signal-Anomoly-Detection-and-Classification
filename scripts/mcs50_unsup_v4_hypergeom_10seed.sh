#!/bin/bash
# v4: replace --cluster-max-anchor-fraction with scale-free rule:
#   (hypergeometric per-class anchor concentration test, alpha=0.01 Bonferroni)
#   AND (anchor mean HDBSCAN prob >= 0.5 = core members) -> reject as leakage.
# No oracle labels, no absolute anchor_fraction threshold.
set -e
cd "$(dirname "$0")/.."

DATASET="C:/Users/Zhuoer/code/thesis/Datasets/SAS/rf_dataset_packed_blob_impulsive_snr14_20_sirm4_0"
PATCHCORE="./outputs/sas_blob_impulsive_snr14_20_patchcore_wide_resnet50_2_256_zscore_allpatches"
PYEXE="/c/Users/Zhuoer/anaconda3/envs/IIS/python.exe"

for seed in 42 43 44 45 46 47 48 49 50 51; do
  OUT="./outputs/sas_blob_impulsive_mcs50_unsup_v4_hypergeom_seed${seed}"
  echo "=== v4 hypergeom seed=${seed} -> ${OUT} ==="
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
    --anchor-core-prob-threshold 0.5 \
    --seed "$seed"
done
echo "=== v4 hypergeom 10-seed done ==="
