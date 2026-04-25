#!/bin/bash
# Stage 1: PatchCore on impulsive-only snr14_20 dataset (no blob noise).
# Same hyperparameters as blob_impulsive_snr14_20 run for direct comparison.
set -e
cd "$(dirname "$0")/.."

DATASET="C:/Users/Zhuoer/code/thesis/Datasets/SAS/rf_dataset_packed_impulsive_only_snr14_20_sirm4_0"
OUT="./outputs/sas_impulsive_only_snr14_20_patchcore_wide_resnet50_2_256_zscore_allpatches"
PYEXE="/c/Users/Zhuoer/anaconda3/envs/IIS/python.exe"

echo "=== PatchCore impulsive-only snr14_20 -> ${OUT} ==="
"$PYEXE" train_eval_patchcore_sas.py \
  --dataset-dir "$DATASET" \
  --output-dir "$OUT" \
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
  # --export-test-patch-artifacts \
  # --export-selection all \
  # --export-embedding-dtype float16
  # ^^^ 以上三项生成 test_patch_artifacts.npz（6000×32×32×1536 float16 ≈ 18.9 GB）。
  # 当前 v6 pipeline 用 --input-mode roi，只消费 anomaly_outputs.npz，不需要 patch embeddings。
  # 上次写这个大文件时被 taskkill 打断，触发 Windows 清理了整个输出目录，导致 metrics.json/memory_bank.pt 丢失。
  # 保留注释：后续若要跑 patch 级聚类 (cluster_patchcore_sas.py)、few-shot patchcore 分类器、
  #          open-set 基线评估 (evaluate_sas_open_set_*) 或 --input-mode patch 的 open-world 实验，
  #          需要重新启用这三项导出。
echo "=== PatchCore done ==="
