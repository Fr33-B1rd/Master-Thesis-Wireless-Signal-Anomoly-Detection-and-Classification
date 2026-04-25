#!/bin/bash
# Refactor validation: full pipeline (Stage 1 PatchCore + Stage 2 mag_1ch × 3 seeds)
# on snr8_14. Outputs go to *new* dirs so existing reference runs are preserved
# for diff-against. set -e: Stage 1 failure aborts before any Stage 2 fires.
#
# Purpose: verify that the recent code restructure (Scope B Stage 1 split,
# Scope C Stage 2 split, dataio rename, support collapse) did not regress
# the end-to-end pipeline. Same hyperparameters as the original mag_1ch
# 3-seed ablation; only the output paths differ.
set -e
cd "$(dirname "$0")/.."

PYEXE="/c/Users/Zhuoer/anaconda3/envs/IIS/python.exe"
DATASET="C:/Users/Zhuoer/code/thesis/Datasets/SAS/rf_dataset_packed_impulsive_only_snr8_14_sirm8_4"
PATCHCORE_OUT="./outputs/sas_impulsive_only_snr8_14_sirm8_4_patchcore_refactor_check"

echo "=== STAGE 1: PatchCore -> ${PATCHCORE_OUT} ==="
"$PYEXE" train_eval_patchcore_sas.py \
  --dataset-dir "$DATASET" \
  --output-dir "$PATCHCORE_OUT" \
  --backbone wide_resnet50_2 \
  --layers layer2 layer3 \
  --input-size 256 \
  --batch-size 8 \
  --num-workers 0 \
  --seed 42 \
  --clip-z 5.0 \
  --sample-patches-per-image 32 \
  --candidate-pool-size 20000 \
  --coreset-ratio 0.01 \
  --max-memory-bank-size 2048 \
  --projection-dim 64 \
  --query-chunk-size 2048 \
  --num-vis 10 \
  --device cuda
echo "=== STAGE 1 DONE ==="

for seed in 42 43 44; do
  out="./outputs/sas_impulsive_only_snr8_14_mag_1ch_refactor_seed${seed}"
  echo "=== START STAGE 2 mag_1ch seed=${seed} -> ${out} ==="
  "$PYEXE" simulate_sas_convnextv2_open_world.py \
    --dataset-dir "$DATASET" \
    --output-dir "$out" \
    --patchcore-output-dir "$PATCHCORE_OUT" \
    --input-mode roi \
    --cluster-dim-reduction umap \
    --cluster-min-cluster-size 50 \
    --cluster-selection-method eom \
    --cluster-accept-min-size 20 \
    --cluster-accept-min-prob 0.5 \
    --cluster-anchor-quantile-alpha 0.05 \
    --input-channels mag_1ch \
    --seed "$seed"
  echo "=== DONE STAGE 2 mag_1ch seed=${seed} ==="
done
echo "=== ALL DONE: 1 Stage 1 + 3 Stage 2 runs ==="
