#!/bin/bash
# v5 scale sweep on snr14_20 seed 42. 5 values. Picks best scale for
# anchor_fraction threshold = scale x (n_anchor/n_total) on the dev dataset.
# Held-out evaluation on snr8_14 / snr2_8 uses the winning scale only.
set -e
cd "$(dirname "$0")/.."

DATASET="C:/Users/Zhuoer/code/thesis/Datasets/SAS/rf_dataset_packed_blob_impulsive_snr14_20_sirm4_0"
PATCHCORE="./outputs/sas_blob_impulsive_snr14_20_patchcore_wide_resnet50_2_256_zscore_allpatches"
PYEXE="/c/Users/Zhuoer/anaconda3/envs/IIS/python.exe"
seed=42

for scale in 1.5 2.0 2.5 3.0 5.0; do
  OUT="./outputs/sas_blob_impulsive_v5_scalesweep_scale${scale}_seed${seed}"
  echo "=== scale=${scale} -> ${OUT} ==="
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
    --cluster-anchor-fraction-scale "$scale" \
    --seed "$seed"
done
echo "=== scale sweep done ==="
