#!/bin/bash
# Alt-class split: tone/comb/fsk known, chirp/pulse/ofdm emerging.
# 3 seeds. v6 hypergeom-quantile (alpha=0.05). Observation run only.
set -e
cd "$(dirname "$0")/.."

DATASET="C:/Users/Zhuoer/code/thesis/Datasets/SAS/rf_dataset_packed_blob_impulsive_snr14_20_sirm4_0"
PATCHCORE="./outputs/sas_blob_impulsive_snr14_20_patchcore_wide_resnet50_2_256_zscore_allpatches"
PYEXE="/c/Users/Zhuoer/anaconda3/envs/IIS/python.exe"

for seed in 42 43 44; do
  OUT="./outputs/sas_blob_impulsive_v6_altclass_tcf_seed${seed}"
  echo "=== alt-class tcf seed=${seed} -> ${OUT} ==="
  "$PYEXE" simulate_sas_convnextv2_open_world.py \
    --dataset-dir "$DATASET" \
    --output-dir "$OUT" \
    --patchcore-output-dir "$PATCHCORE" \
    --known-classes "tone,comb,fsk" \
    --emerging-classes "chirp,pulse,ofdm" \
    --input-mode roi \
    --cluster-dim-reduction umap \
    --cluster-min-cluster-size 50 \
    --cluster-selection-method eom \
    --cluster-accept-min-size 20 \
    --cluster-accept-min-prob 0.5 \
    --cluster-anchor-quantile-alpha 0.05 \
    --seed "$seed"
done
echo "=== alt-class tcf 3-seed done ==="
