#!/bin/bash
# True 1-channel baseline (mag_1ch): in_channels=1 Conv2d with weights
# initialized from the SUM of pretrained RGB weights. Isolates "stem
# surgery + proper 1-channel fine-tuning" from "phase information in IF/GD".
# 3 seeds on snr8_14.
set -e
cd "$(dirname "$0")/.."

PYEXE="/c/Users/Zhuoer/anaconda3/envs/IIS/python.exe"
DATASET="C:/Users/Zhuoer/code/thesis/Datasets/SAS/rf_dataset_packed_impulsive_only_snr8_14_sirm8_4"
PATCHCORE="./outputs/sas_impulsive_only_snr8_14_sirm8_4_patchcore_wide_resnet50_2_256_zscore_allpatches"

for seed in 42 43 44; do
  out="./outputs/sas_impulsive_only_snr8_14_mag_1ch_seed${seed}"
  echo "=== START mag_1ch seed=${seed} -> ${out} ==="
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
    --input-channels mag_1ch \
    --seed "$seed"
  echo "=== DONE mag_1ch seed=${seed} ==="
done
echo "=== ALL 3 RUNS DONE ==="
