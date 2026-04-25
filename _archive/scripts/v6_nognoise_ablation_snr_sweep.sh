#!/bin/bash
# Ablation: v6 hypergeom-quantile WITHOUT Gaussian noise augmentation during classifier
# training (--train-noise-std 0). 3 SNR regimes × 3 seeds = 9 runs.
# Purpose: test whether the fsk/chirp conditional-recognition reversal at high SNR
# is caused by the train-time gnoise. Output dirs use _nognoise_ tag for disambiguation.
set -e
cd "$(dirname "$0")/.."

PYEXE="/c/Users/Zhuoer/anaconda3/envs/IIS/python.exe"

run_one_snr() {
  local tag="$1"
  local dataset="$2"
  local patchcore="$3"
  for seed in 42 43 44; do
    local out="./outputs/sas_impulsive_only_${tag}_v6_nognoise_seed${seed}"
    echo "=== impulsive-only ${tag} v6 NO-gnoise seed=${seed} -> ${out} ==="
    "$PYEXE" simulate_sas_convnextv2_open_world.py \
      --dataset-dir "$dataset" \
      --output-dir "$out" \
      --patchcore-output-dir "$patchcore" \
      --input-mode roi \
      --cluster-dim-reduction umap \
      --cluster-min-cluster-size 50 \
      --cluster-selection-method eom \
      --cluster-accept-min-size 20 \
      --cluster-accept-min-prob 0.5 \
      --cluster-anchor-quantile-alpha 0.05 \
      --train-noise-std 0 \
      --seed "$seed"
  done
  echo "=== ${tag} 3-seed no-gnoise done ==="
}

run_one_snr "snr2_8"   "C:/Users/Zhuoer/code/thesis/Datasets/SAS/rf_dataset_packed_impulsive_only_snr2_8_sirm12_8"  "./outputs/sas_impulsive_only_snr2_8_sirm12_8_patchcore_wide_resnet50_2_256_zscore_allpatches"
run_one_snr "snr8_14"  "C:/Users/Zhuoer/code/thesis/Datasets/SAS/rf_dataset_packed_impulsive_only_snr8_14_sirm8_4"  "./outputs/sas_impulsive_only_snr8_14_sirm8_4_patchcore_wide_resnet50_2_256_zscore_allpatches"
run_one_snr "snr14_20" "C:/Users/Zhuoer/code/thesis/Datasets/SAS/rf_dataset_packed_impulsive_only_snr14_20_sirm4_0" "./outputs/sas_impulsive_only_snr14_20_patchcore_wide_resnet50_2_256_zscore_allpatches"

echo "=== all 9 no-gnoise ablation runs done ==="
