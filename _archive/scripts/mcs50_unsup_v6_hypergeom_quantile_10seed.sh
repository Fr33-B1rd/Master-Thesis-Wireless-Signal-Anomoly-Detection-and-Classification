#!/bin/bash
# v6: exact hypergeometric quantile threshold for anchor_fraction.
# Per-cluster threshold = hypergeom.ppf(0.95, N, K_a_total, n) / n.
# Statistically rigorous, parameter-free (alpha=0.05 is a standard choice),
# cluster-size-dependent. No empirical scale factor.
set -e
cd "$(dirname "$0")/.."

DATASET="C:/Users/Zhuoer/code/thesis/Datasets/SAS/rf_dataset_packed_blob_impulsive_snr14_20_sirm4_0"
PATCHCORE="./outputs/sas_blob_impulsive_snr14_20_patchcore_wide_resnet50_2_256_zscore_allpatches"
PYEXE="/c/Users/Zhuoer/anaconda3/envs/IIS/python.exe"

for seed in 42 43 44 45 46 47 48 49 50 51; do
  OUT="./outputs/sas_blob_impulsive_mcs50_unsup_v6_hq_seed${seed}"
  echo "=== v6 hypergeom-quantile seed=${seed} -> ${OUT} ==="
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
    --seed "$seed"
done
echo "=== v6 10-seed done ==="
