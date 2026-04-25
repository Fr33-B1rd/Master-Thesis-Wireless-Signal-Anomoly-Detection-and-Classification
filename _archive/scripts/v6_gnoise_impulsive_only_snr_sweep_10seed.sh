#!/bin/bash
# v6 hypergeom-quantile + Gaussian noise — impulsive-only SNR sweep.
# 10 seeds per SNR regime = 30 total open-world runs, sequential (GPU contention).
# Output dirs: sas_impulsive_only_{snr_tag}_v6_gnoise_seed{N}
# Uses per-SNR PatchCore outputs already trained via the snr_sweep stage-1 runs.
set -e
cd "$(dirname "$0")/.."

PYEXE="/c/Users/Zhuoer/anaconda3/envs/IIS/python.exe"

run_one_snr() {
  local tag="$1"
  local dataset="$2"
  local patchcore="$3"
  for seed in 42 43 44 45 46 47 48 49 50 51; do
    local out="./outputs/sas_impulsive_only_${tag}_v6_gnoise_seed${seed}"
    echo "=== impulsive-only ${tag} v6 gnoise seed=${seed} -> ${out} ==="
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
      --seed "$seed"
  done
  echo "=== ${tag} 10-seed done ==="
}

run_one_snr "snr2_8"   "C:/Users/Zhuoer/code/thesis/Datasets/SAS/rf_dataset_packed_impulsive_only_snr2_8_sirm12_8"  "./outputs/sas_impulsive_only_snr2_8_sirm12_8_patchcore_wide_resnet50_2_256_zscore_allpatches"
run_one_snr "snr8_14"  "C:/Users/Zhuoer/code/thesis/Datasets/SAS/rf_dataset_packed_impulsive_only_snr8_14_sirm8_4"  "./outputs/sas_impulsive_only_snr8_14_sirm8_4_patchcore_wide_resnet50_2_256_zscore_allpatches"
run_one_snr "snr14_20" "C:/Users/Zhuoer/code/thesis/Datasets/SAS/rf_dataset_packed_impulsive_only_snr14_20_sirm4_0" "./outputs/sas_impulsive_only_snr14_20_patchcore_wide_resnet50_2_256_zscore_allpatches"

echo "=== all SNR sweep 30-seed runs done ==="
