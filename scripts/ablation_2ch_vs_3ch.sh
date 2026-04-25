#!/bin/bash
# 2ch vs 3ch ablation on snr8_14 (soft gate).
# Configs: mag_if (Ch0+Ch1, GD zeroed), mag_gd (Ch0+Ch2, IF zeroed).
# 3 seeds each, 6 runs total. Sequential (GPU contention).
# Baselines mag / mag_if_gd already exist for seeds 42/43/44.
set -e
cd "$(dirname "$0")/.."

PYEXE="/c/Users/Zhuoer/anaconda3/envs/IIS/python.exe"
DATASET="C:/Users/Zhuoer/code/thesis/Datasets/SAS/rf_dataset_packed_impulsive_only_snr8_14_sirm8_4"
PATCHCORE="./outputs/sas_impulsive_only_snr8_14_sirm8_4_patchcore_wide_resnet50_2_256_zscore_allpatches"

run_one() {
  local mode="$1"   # mag_if or mag_gd
  local seed="$2"
  local out="./outputs/sas_impulsive_only_snr8_14_${mode}_seed${seed}"
  echo "=== START mode=${mode} seed=${seed} -> ${out} ==="
  "$PYEXE" simulate_sas_convnextv2_open_world.py \
    --dataset-dir "$DATASET" \
    --output-dir "$out" \
    --patchcore-output-dir "$PATCHCORE" \
    --input-mode roi \
    --cluster-dim-reduction umap \
    --cluster-min-cluster-size 50 \
    --cluster-selection-method eom \
    --cluster-accept-min-size 20 \
    --cluster-accept-min-prob 0.5 \
    --cluster-anchor-quantile-alpha 0.05 \
    --input-channels "$mode" \
    --seed "$seed"
  echo "=== DONE mode=${mode} seed=${seed} ==="
}

for seed in 42 43 44; do
  run_one mag_if $seed
  run_one mag_gd $seed
done

echo "=== ALL 6 RUNS DONE ==="
