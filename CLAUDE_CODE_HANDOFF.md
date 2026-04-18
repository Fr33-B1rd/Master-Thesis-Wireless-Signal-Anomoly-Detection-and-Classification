# Anomaly Localization: ConvNeXtV2 ROI Open-World Line Handoff

## 1. Repository and Entry Points

- Repo root:
  - `C:\Users\Zhuoer\code\thesis\Anomaly_Localization`
- Static open-set entry:
  - `C:\Users\Zhuoer\code\thesis\Anomaly_Localization\evaluate_sas_open_set_convnextv2.py`
- Static open-set implementation:
  - `C:\Users\Zhuoer\code\thesis\Anomaly_Localization\sas_open_world\experiments\convnextv2_open_set.py`
- Open-world entry:
  - `C:\Users\Zhuoer\code\thesis\Anomaly_Localization\simulate_sas_convnextv2_open_world.py`
- Open-world implementation:
  - `C:\Users\Zhuoer\code\thesis\Anomaly_Localization\sas_open_world\experiments\convnextv2_open_world.py`

## 2. System Architecture

### 2.0 Stage 1: PatchCore anomaly detection

The overall project contains a PatchCore-based anomaly detection stage that runs before the ConvNeXt open-set / open-world classifier line.

Stage 1 PatchCore flow:

1. Take spectrogram inputs from the SAS dataset.
2. Run PatchCore anomaly detection on spectrogram images.
3. Produce image-level anomaly scores.
4. Produce pixel/patch-level anomaly maps.
5. Optionally export patch-level intermediate artifacts for later feature construction.

PatchCore outputs used in this project:

- image-level anomaly score
- anomaly heatmap / anomaly map
- patch-level intermediate artifacts

How Stage 1 outputs are used downstream:

- anomaly maps are used to generate ROI boxes for `ROI-only` ConvNeXt input
- anomaly maps were also used in experiments such as:
  - soft attention
  - component mask
- patch-level artifacts were used in earlier clustering feature pipelines such as:
  - `PatchCore mean`
  - `PatchCore gram`
  - `PatchCore mean+gram`

Current relationship between PatchCore and the ConvNeXt line:

- PatchCore is still required for ROI localization.
- The current best open-world line does **not** use PatchCore features as the final clustering feature source.
- The current best clustering feature source is `ROI-only ConvNeXt embedding`.

Stage 1 relevant scripts:

- SAS PatchCore training/evaluation:
  - `C:\Users\Zhuoer\code\thesis\Anomaly_Localization\train_eval_patchcore_sas.py`
- SAS PatchCore clustering utility:
  - `C:\Users\Zhuoer\code\thesis\Anomaly_Localization\cluster_patchcore_sas.py`
- Generic PatchCore implementation used by SAS wrapper:
  - `C:\Users\Zhuoer\code\thesis\Anomaly_Localization\train_eval_patchcore_wsad.py`

Stage 1 implementation notes:

- `train_eval_patchcore_sas.py`
  - SAS-specific wrapper that loads packed SAS spectrogram arrays and calls the shared PatchCore implementation.
- `train_eval_patchcore_wsad.py`
  - shared PatchCore implementation used by the SAS wrapper
  - contains prediction, anomaly map export, visualization, and artifact export logic

### 2.1 Static open-set front end

Current front-end line:

1. Load SAS spectrograms.
2. Use PatchCore anomaly maps to localize the most suspicious region.
3. Crop ROI only from the spectrogram.
4. Feed ROI into `ConvNeXtV2-tiny`.
5. Train classifier on known classes only with cross-entropy.
6. At inference, compute `energy` from logits.
7. Reject samples as `unknown` when `energy > threshold`.

Important points:

- The following input variants have been tested:
  - full image
  - ROI-only
  - soft attention
  - component mask
  - dual full+ROI
- DINOv3-as-classifier has also been tested as a separate comparison line.
- The current experimental setup combines:
  - few-shot training
  - plain CE loss
  - train-calibrated energy threshold

### 2.2 Open-world line

Current best open-world pipeline:

1. Initial known-class training with `ROI-only ConvNeXtV2-tiny`.
2. Stream-time known/unknown decision by `global energy gate`.
3. All rejected unknown samples are accumulated and clustered at the end.
4. Clustering features come from `ROI-only ConvNeXt` embeddings, not PatchCore mean+gram.
5. HDBSCAN finds candidate new clusters.
6. High-purity accepted clusters are treated as discovered new classes.
7. Model is retrained with absorbed classes.
8. Final holdout evaluation is reported on all final known classes.

## 3. Dataset

### 3.1 Main dataset used by the current ConvNeXt line

- Dataset path:
  - `C:\Users\Zhuoer\code\thesis\Datasets\SAS\rf_dataset_packed_gaussian_6anom_db85_10`

### 3.2 Files in dataset

- `train_spectrogram.npy`
- `train_labels.npy`
- `train_manifest.csv`
- `train_iq.npy`
- `test_spectrogram.npy`
- `test_labels.npy`
- `test_manifest.csv`
- `test_iq.npy`

### 3.3 Basic label setup used in this line

Known classes:

- `chirp`
- `pulse`
- `tone`

Emerging / unknown classes:

- `comb`
- `ofdm`
- `fsk`

Important note:

- This experimental line does **not** include a separate normal/background class.
- It is anomaly-only known-vs-unknown classification and then open-world absorption.

### 3.4 Practical split conventions used in experiments

Typical static setup:

- `80-shot/class` for known-class training
- Remaining known samples for evaluation
- Unknown classes only appear in evaluation

Typical open-world setup:

- `80-shot/class` initial known training
- `600/class` stream flow for each class
- separate holdout evaluation at the end

## 4. PatchCore Dependency

ROI-only mode depends on precomputed PatchCore anomaly outputs.

Current PatchCore output directory used by this line:

- `C:\Users\Zhuoer\code\thesis\Anomaly_Localization\outputs\sas_gaussian_6anom_db85_10_patchcore_wide_resnet50_2_256_zscore_allpatches`

This directory must contain at least:

- `anomaly_outputs.npz`

Files typically present in the current SAS PatchCore output directory:

- `config.json`
- `metrics.json`
- `test_scores.csv`
- `anomaly_outputs.npz`
- `test_patch_artifacts.npz`
- `patchcore_memory_bank.pt`
- `confusion_matrix.png`
- `roc_curve.png`
- `anomaly_montage_per_type_top5.png`
- `test_label_breakdown.json`

The ConvNeXt scripts use PatchCore outputs for:

- ROI box generation
- not for final classification logits
- not for current best clustering features

PatchCore output file roles:

- `config.json`
  - PatchCore run configuration.
- `metrics.json`
  - Aggregate detector metrics and artifact export metadata.
- `test_scores.csv`
  - Per-sample image-level anomaly scores.
- `anomaly_outputs.npz`
  - Contains anomaly maps used by ROI-based ConvNeXt input modes.
- `test_patch_artifacts.npz`
  - Contains exported patch-level artifacts used in earlier feature pipelines such as `mean`, `gram`, and `mean+gram`.
- `patchcore_memory_bank.pt`
  - Saved PatchCore memory bank.
- `confusion_matrix.png`
  - Detector confusion-style visualization if generated in that run.
- `roc_curve.png`
  - Detector ROC visualization.
- `anomaly_montage_per_type_top5.png`
  - Qualitative montage of high-scoring anomalies.
- `test_label_breakdown.json`
  - Test-set label count summary for that PatchCore run.

Notes on Stage 1 visualizations:

- anomaly montage overlays are generated in:
  - `C:\Users\Zhuoer\code\thesis\Anomaly_Localization\train_eval_patchcore_wsad.py`
- the montage displays:
  - base spectrogram with `cmap=\"gray\"`
  - anomaly map overlay with `cmap=\"jet\"`
  - overlay alpha `0.45`
- confusion matrix plots use `cmap=\"Blues\"`

## 5. Current Best Configuration

### 5.1 Best single-run open-world result

Output directory:

- `C:\Users\Zhuoer\code\thesis\Anomaly_Localization\outputs\sas_gaussian_6anom_db85_10_convnextv2_energyglobal_balcal_open_world_80shot_flow600_allunknown_convnextcluster_roi_minsize20_v1`

Configuration:

- Backbone: `convnextv2_tiny`
- Input mode: `roi`
- Training loss: plain `CE`
- Epochs: `12`
- Batch size: `32`
- LR: `1e-4`
- Weight decay: `1e-4`
- Temperature: `2.0`
- Gate mode: `energy_global`
- Energy calibration: `balanced`
- Threshold quantile: `95`
- Flow per class: `600`
- Clustering feature source: `convnext`
- Cluster all unknown at end: `true`
- HDBSCAN:
  - `pca_dim = 8`
  - `min_cluster_size = 8`
  - `min_samples = 1`
- Cluster accept:
  - `purity >= 0.9`
  - `min_size = 20`
  - `max_anchor_fraction = 0.25`

Best single-run results:

- Holdout accuracy: `0.9104`
- Holdout macro-F1: `0.9540`

Holdout recognized rate by class:

- `chirp = 0.9833`
- `pulse = 0.8500`
- `tone = 0.9583`
- `comb = 0.8800`
- `ofdm = 0.9150`
- `fsk = 0.9000`

Accepted new-class samples in this run:

- `comb = 349`
- `ofdm = 585`
- `fsk = 284`

Source files:

- `metrics.json`
- `update_events.json`
- `update_01/cluster_summary.json`

### 5.2 Multi-run reference

10-run aggregate:

- `C:\Users\Zhuoer\code\thesis\Anomaly_Localization\outputs\sas_gaussian_openworld_roi_minsize20_repeat10_aggregate.json`

Aggregate from 10 runs:

- Holdout accuracy mean: `0.6297`
- Holdout macro-F1 mean: `0.7157`

## 6. Best Static Front-End Result

Best post-hoc static open-set ROI-only result:

- `C:\Users\Zhuoer\code\thesis\Anomaly_Localization\outputs\sas_gaussian_6anom_db85_10_convnextv2_tiny_energy_open_set_80shot_roi_v1`

Metrics:

- `known_coverage = 0.9968`
- `macro_f1_known = 0.9995`
- `avg_unknown_reject = 0.9471`
- `comb = 0.9363`
- `fsk = 0.9075`
- `ofdm = 0.9975`

Important warning:

- This result uses post-hoc threshold search on the evaluation set.
- It is **not** directly comparable to open-world `pre_update`.

## 7. Threshold Protocol Change Relevant to Coverage

One major experimental change was the threshold selection protocol:

- old static protocol:
  - post-hoc scan on evaluation set
- aligned protocol:
  - threshold = `95th percentile` of training-set energy

This protocol change makes static evaluation more comparable to open-world `pre_update`.

## 8. What Has Already Been Tried

The following variants have been implemented and evaluated at least once:

- Feature AE / VAE / PER-like variants
- Per-class energy threshold
- Single-prototype distance gate
- DINOv3 replacing ConvNeXt classifier
- Mahalanobis distance
- ArcFace + CE
- Center Loss (current tested forms)
- Early stopping variants

## 9. Recent Static Experiments Relevant to Coverage

### 9.1 Early stopping experiments

Multiple early stopping variants have been tested, including:

- internal validation splits such as `70+10`, `80+10`, `80+20`, `80+40`, `70+20`, `70+40`
- external held-out known validation
- `val_loss` monitoring
- `known_coverage` monitoring

### 9.2 ArcFace experiments

Recent ArcFace experiment setup:

- fixed `20 epochs`
- sweep of `arcface margin / scale`

### 9.3 Center Loss experiments

Weak Center Loss settings have also been tested under the aligned static protocol.

## 10. Runtime Environment

### 10.1 Python environment

Current working interpreter:

```powershell
C:\Users\Zhuoer\anaconda3\envs\IIS\python.exe
```

### 10.2 Repo root

```powershell
cd C:\Users\Zhuoer\code\thesis\Anomaly_Localization
```

## 11. Run Commands

### 11.1 Best current open-world configuration

```powershell
C:\Users\Zhuoer\anaconda3\envs\IIS\python.exe simulate_sas_convnextv2_open_world.py `
  --dataset-dir C:\Users\Zhuoer\code\thesis\Datasets\SAS\rf_dataset_packed_gaussian_6anom_db85_10 `
  --output-dir C:\Users\Zhuoer\code\thesis\Anomaly_Localization\outputs\YOUR_OUTPUT_DIR `
  --known-classes chirp,pulse,tone `
  --emerging-classes comb,ofdm,fsk `
  --shot 80 `
  --flow-per-class 600 `
  --anchor-per-class 30 `
  --model-name convnextv2_tiny `
  --epochs 12 `
  --batch-size 32 `
  --lr 1e-4 `
  --weight-decay 1e-4 `
  --temperature 2.0 `
  --gate-mode energy_global `
  --threshold-quantile 95 `
  --energy-calibration balanced `
  --input-mode roi `
  --roi-quantile 99 `
  --roi-pad-ratio 0.1 `
  --roi-min-side 32 `
  --clustering-feature-source convnext `
  --patchcore-output-dir C:\Users\Zhuoer\code\thesis\Anomaly_Localization\outputs\sas_gaussian_6anom_db85_10_patchcore_wide_resnet50_2_256_zscore_allpatches `
  --cluster-pca-dim 8 `
  --cluster-min-cluster-size 8 `
  --cluster-min-samples 1 `
  --cluster-accept-purity 0.9 `
  --cluster-accept-min-size 20 `
  --cluster-max-anchor-fraction 0.25 `
  --cluster-all-unknown-at-end `
  --seed 42
```

### 11.2 Static ROI-only open-set baseline

```powershell
C:\Users\Zhuoer\anaconda3\envs\IIS\python.exe evaluate_sas_open_set_convnextv2.py `
  --dataset-dir C:\Users\Zhuoer\code\thesis\Datasets\SAS\rf_dataset_packed_gaussian_6anom_db85_10 `
  --output-dir C:\Users\Zhuoer\code\thesis\Anomaly_Localization\outputs\YOUR_OUTPUT_DIR `
  --known-classes chirp,pulse,tone `
  --unknown-classes comb,fsk,ofdm `
  --shot 80 `
  --model-name convnextv2_tiny `
  --epochs 20 `
  --batch-size 32 `
  --lr 1e-4 `
  --weight-decay 1e-4 `
  --finetune-scope full `
  --reject-mode energy `
  --temperature 2.0 `
  --input-mode roi `
  --patchcore-output-dir C:\Users\Zhuoer\code\thesis\Anomaly_Localization\outputs\sas_gaussian_6anom_db85_10_patchcore_wide_resnet50_2_256_zscore_allpatches `
  --roi-quantile 99 `
  --roi-pad-ratio 0.1 `
  --roi-min-side 32 `
  --coverage-constraint 0.95 `
  --thresholds 0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95 `
  --seed 42
```

### 11.3 Static aligned threshold protocol

Use this if you want static results aligned with open-world thresholding:

```powershell
C:\Users\Zhuoer\anaconda3\envs\IIS\python.exe evaluate_sas_open_set_convnextv2.py `
  --dataset-dir C:\Users\Zhuoer\code\thesis\Datasets\SAS\rf_dataset_packed_gaussian_6anom_db85_10 `
  --output-dir C:\Users\Zhuoer\code\thesis\Anomaly_Localization\outputs\YOUR_OUTPUT_DIR `
  --known-classes chirp,pulse,tone `
  --unknown-classes comb,fsk,ofdm `
  --shot 80 `
  --model-name convnextv2_tiny `
  --epochs 12 `
  --batch-size 32 `
  --lr 1e-4 `
  --weight-decay 1e-4 `
  --finetune-scope full `
  --reject-mode energy `
  --temperature 2.0 `
  --input-mode roi `
  --patchcore-output-dir C:\Users\Zhuoer\code\thesis\Anomaly_Localization\outputs\sas_gaussian_6anom_db85_10_patchcore_wide_resnet50_2_256_zscore_allpatches `
  --roi-quantile 99 `
  --roi-pad-ratio 0.1 `
  --roi-min-side 32 `
  --thresholds train_calibrated `
  --threshold-quantile 95 `
  --threshold-source train `
  --seed 42
```

## 12. Execution Notes and Caveats

### 12.1 ROI modes require PatchCore output

If `--input-mode roi` is used, `--patchcore-output-dir` must be valid.

### 12.2 Use fresh output directories

Do not reuse output directories across incompatible runs.

### 12.3 Static metrics depend heavily on threshold protocol

There are two different static evaluation modes in this codebase:

- post-hoc threshold search
- train-calibrated threshold

Do not compare them directly without noting which protocol was used.
