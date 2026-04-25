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

### 2.0 Stage 0: 3-channel dataset packing

Raw I/Q recordings are packed into 3-channel spectrogram tensors before any training. The packer lives outside this repo:

- Packer script:
  - `C:\Users\Zhuoer\code\thesis\Datasets\SAS\build_rf_3channel.py`

Channel layout per packed tensor `*_spectrogram_3c.npy`:

- `Ch0`: `log_mag` (dB-normalized magnitude)
- `Ch1`: `IF` (instantaneous frequency, `∂φ/∂t / π`)
- `Ch2`: `GD` (group delay, `−∂φ/∂f / π`)

Phase channels (IF, GD) are masked by a soft gate so low-energy bins do not inject noise:

- Soft gate: `mask = |X| / (|X| + ε · peak|X|)`, `ε = 0.05`
- The `ε = 0.05` value is **support-adaptive** (effective threshold ranges from ~4σ to ~10σ across signal classes), not a fixed CFAR. See the ε-vs-σ conversion check in the `_archive/` commit history for the numerical check.

Also produced by the packer:

- `channel_stats.json` (per-channel mean/std for normalization)
- legacy single-channel `test_spectrogram.npy` (uint8 [0,255]) — consumed only by the `mag_1ch` input path

### 2.0.5 Dataset reproduction (regenerating `rf_dataset_packed_impulsive_only_snr8_14_sirm8_4`)

The dataset itself is **generated**, not shipped. To reproduce the experiments below from scratch, you must first regenerate the dataset with the exact CLI invocations recorded here. The five generator scripts live outside this repo at `C:\Users\Zhuoer\code\thesis\Datasets\SAS\` (or wherever you placed them). They are independent of the model code and run on the same `IIS` env (see §8.1).

Naming convention decode:
- `impulsive_only` → `--background-noise-models impulsive_mix` (only impulsive noise; blob noise disabled)
- `snr8_14` → wifi SNR drawn uniformly from `[8.0, 14.0]` dB
- `sirm8_4` → signal-to-interference ratio margin: `anomaly-to-wifi-ratio` drawn uniformly from `[-8.0, -4.0]` dB

The four classes-per-split layout (chirp, comb, fsk, ofdm, pulse, tone + normal) is built into `build_rf_dataset.py` defaults; the fields recovered from a sample's `.json` metadata confirm `wifi_snr_db_range=[8.0,14.0]`, `anomaly_to_wifi_ratio_db_range=[-8.0,-4.0]`, `spectrogram_normalization=fixed_db`, `spectrogram_db_range=[-85.0,-10.0]`, `spectrogram_size=256`, `stft_nfft=1024`, `sample_rate_hz=20e6`, `duration_ms=20.0`.

Three-step pipeline:

```bash
PYEXE="/c/Users/Zhuoer/anaconda3/envs/IIS/python.exe"
DATA_ROOT="C:/Users/Zhuoer/code/thesis/Datasets/SAS"
TAG="impulsive_only_snr8_14_sirm8_4"

# Step 1: build per-sample IQ + spectrogram + masks (writes rf_dataset_full_<TAG>/)
"$PYEXE" "$DATA_ROOT/build_rf_dataset.py" \
  --mode full \
  --output-dir "$DATA_ROOT/rf_dataset_full_${TAG}" \
  --background-noise-models impulsive_mix \
  --wifi-snr-min-db 8.0 --wifi-snr-max-db 14.0 \
  --anomaly-to-wifi-ratio-min-db -8.0 --anomaly-to-wifi-ratio-max-db -4.0 \
  --sample-rate 20000000 --duration-ms 20.0 \
  --nfft 1024 --image-size 256 \
  --spectrogram-normalization fixed_db \
  --spectrogram-db-min -85.0 --spectrogram-db-max -10.0 \
  --seed 2026
# (omit --enable-blob-noise → blob noise off; that's the "impulsive_only" part)

# Step 2: pack per-sample arrays into split-level numpy bundles
#         (writes rf_dataset_packed_<TAG>/{train,test}_iq.npy, _spectrogram.npy,
#          _labels.npy, _mask.npy, manifest.csv)
"$PYEXE" "$DATA_ROOT/pack_rf_dataset.py" \
  --input-dir "$DATA_ROOT/rf_dataset_full_${TAG}" \
  --output-dir "$DATA_ROOT/rf_dataset_packed_${TAG}"

# Step 3: compute 3-channel spectrograms (log-mag, IF, GD) + channel_stats.json
#         (writes *_spectrogram_3c.npy alongside the existing *_spectrogram.npy)
"$PYEXE" "$DATA_ROOT/build_rf_3channel.py" \
  --packed-dir "$DATA_ROOT/rf_dataset_packed_${TAG}" \
  --gate-mode soft --mask-epsilon-ratio 0.05 \
  --sample-rate 20000000 --nfft 1024 --image-size 256 \
  --db-min -85.0 --db-max -10.0
```

After step 3, `--dataset-dir` for all model-side scripts in this repo points at `$DATA_ROOT/rf_dataset_packed_${TAG}`.

Two utility scripts also live in `Datasets/SAS/` but are not part of the main reproduction path:

- `regenerate_masks.py` — re-derives `*_mask.npy` from sample-level `.json` metadata in-place (used after a mask-format fix; not needed for fresh generation).
- `build_multi_anomaly_testset.py` — generates an alternate test set with multiple anomalies per capture (used by some `_archive/` experiments).

Determinism: `build_rf_dataset.py --seed` controls the IQ/spectrogram generation. Steps 2 and 3 are deterministic given step 1's outputs. Different seeds → different captures (different anomaly draws, different WiFi noise realizations); for a given seed, step 1 is bit-exact reproducible on the same NumPy/SciPy versions.

### 2.1 Stage 1: PatchCore anomaly detection

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

- SAS PatchCore driver (CLI + main):
  - `C:\Users\Zhuoer\code\thesis\Anomaly_Localization\train_eval_patchcore_sas.py`
- SAS PatchCore clustering utility:
  - `C:\Users\Zhuoer\code\thesis\Anomaly_Localization\cluster_patchcore_sas.py`
- Dataset-neutral PatchCore core modules:
  - `C:\Users\Zhuoer\code\thesis\Anomaly_Localization\core\patchcore.py` (model + `ArrayDataset2D` + feature extractor)
  - `C:\Users\Zhuoer\code\thesis\Anomaly_Localization\core\viz.py` (evaluate_scores, ROC/CM/montage plots, score CSV)
  - `C:\Users\Zhuoer\code\thesis\Anomaly_Localization\core\metrics.py` (pixel-AUROC + AUPRO)
- SAS dataset loader:
  - `C:\Users\Zhuoer\code\thesis\Anomaly_Localization\dataio\sas\bundle.py` (`SASBundle`, `load_sas_bundle`, `save_label_breakdown`)

Stage 1 implementation notes:

- `train_eval_patchcore_sas.py`
  - SAS-specific driver: parses CLI args, builds datasets via `dataio.sas.bundle.load_sas_bundle`, runs the model from `core.patchcore.PatchCore`, and writes artifacts via `core.viz.*` and `core.metrics.compute_localization_metrics`.
- `core/patchcore.py`
  - Dataset-neutral `PatchCore` (ImageNet-pretrained ResNet backbone, k-center coreset memory bank, nearest-neighbor scoring, patch-artifact export) and `ArrayDataset2D` (2D-array → 3-channel tensor wrapper with z-score / per-image z-score normalization + edge masking + resize).
- `core/viz.py`
  - Visualization helpers (ROC, confusion matrix, anomaly montage) and `evaluate_scores` (image-level AUROC + CM at Youden-J threshold).
- `core/metrics.py`
  - Pixel-AUROC (histogram streaming) and MVTec-style AUPRO (quantile-spaced threshold grid, 8-connectivity regions), with optional per-class breakdown.

### 2.2 Static open-set front end

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

### 2.3 Open-world line (current main pipeline)

Current best open-world pipeline:

1. Initial known-class training with `ROI-only ConvNeXtV2-tiny`, `input-channels = mag_1ch`.
2. Stream-time known/unknown decision by `global energy gate`.
3. All rejected unknown samples are accumulated and clustered at the end.
4. Clustering features come from `ROI-only ConvNeXt` embeddings.
5. HDBSCAN + UMAP (`eom` selection, anchor-quantile α=0.05) finds candidate new clusters.
6. High-purity accepted clusters are treated as discovered new classes.
7. Model is retrained with absorbed classes.
8. Final holdout evaluation is reported on all final known classes.

### 2.4 Input-channels dispatch in the main pipeline

`--input-channels` selects the stem + input construction. The primary path is `mag_1ch`; the other modes exist for ablation only.

| Mode | Stem | Input | Purpose |
|---|---|---|---|
| `mag_1ch` (**main**) | Conv2d, in_channels=1, weight = `sum` of RGB pretrained weights + small Gaussian perturbation | Ch0 only, 1-channel tensor | Correct 1-channel baseline; avoids R=G=B color-opponent cancellation on grayscale |
| `mag` | Unmodified 3-channel ImageNet stem | Ch0 replicated to R=G=B | Naive baseline — demonstrates R=G=B stem collapse (seed-unstable) |
| `mag_if` | Unmodified 3-channel stem | Ch0+IF, Ch2 zeroed post-normalization | 2-channel ablation, IF contribution only |
| `mag_gd` | Unmodified 3-channel stem | Ch0+GD, Ch1 zeroed post-normalization | 2-channel ablation, GD contribution only |
| `mag_if_gd` | Unmodified 3-channel stem | Full 3-channel (Ch0+IF+GD) | Full reference; kept for future re-enablement |

The 2-channel ablations (`mag_if`, `mag_gd`, `mag_if_gd`) all run on the same `in_channels=3` model by zeroing the unused channel **after** normalization — the stem filters on a zeroed channel contribute nothing to the first activation, so there is no weight-surgery confound between them.

## 3. Dataset

### 3.1 Main dataset used by the current ConvNeXt line

- Dataset path:
  - `C:\Users\Zhuoer\code\thesis\Datasets\SAS\rf_dataset_packed_impulsive_only_snr8_14_sirm8_4`
- Secondary (higher-SNR comparison):
  - `C:\Users\Zhuoer\code\thesis\Datasets\SAS\rf_dataset_packed_impulsive_only_snr14_20_sirm4_0`

### 3.2 Files in dataset

3-channel packed arrays consumed by the main pipeline:

- `train_spectrogram_3c.npy` (float16, shape N×3×H×W, Ch0=log_mag, Ch1=IF, Ch2=GD)
- `test_spectrogram_3c.npy`
- `channel_stats.json` (per-channel mean/std)

Legacy single-channel arrays (consumed only by `--input-channels mag_1ch`):

- `train_spectrogram.npy` (uint8 [0,255])
- `test_spectrogram.npy` (uint8 [0,255])

Labels / manifests / raw IQ:

- `train_labels.npy`, `test_labels.npy` (string labels)
- `train_manifest.csv`, `test_manifest.csv`
- `train_iq.npy`, `test_iq.npy`

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

- `C:\Users\Zhuoer\code\thesis\Anomaly_Localization\outputs\sas_impulsive_only_snr8_14_sirm8_4_patchcore_wide_resnet50_2_256_zscore_allpatches`
- snr14_20 companion: `C:\Users\Zhuoer\code\thesis\Anomaly_Localization\outputs\sas_impulsive_only_snr14_20_patchcore_wide_resnet50_2_256_zscore_allpatches`

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

- anomaly montage overlays are generated by `save_anomaly_montage` in:
  - `C:\Users\Zhuoer\code\thesis\Anomaly_Localization\core\viz.py`
- the montage displays:
  - base spectrogram with `cmap=\"gray\"`
  - anomaly map overlay with `cmap=\"jet\"`
  - overlay alpha `0.45`
- confusion matrix plots use `cmap=\"Blues\"`

## 5. Current Best Configuration

### 5.1 Main pipeline

Backbone + training:

- Backbone: `convnextv2_tiny` (ImageNet pretrained, ROI-only input)
- **Input channels: `mag_1ch`** (1-channel stem surgery; see §2.4)
- Training loss: plain `CE`
- Epochs: `12`, batch size `32`, LR `1e-4`, weight decay `1e-4`, temperature `2.0`

Gate + stream:

- Gate mode: `energy_global`
- Energy calibration: `balanced`
- Flow per class: `600`, initial shot: `80`
- Threshold quantile: `95`

Clustering (HDBSCAN on ConvNeXt embeddings, anchor-assisted):

- Dim reduction: `umap`
- `min_cluster_size = 50`, selection method: `eom`
- Accept rule: `min_size = 20`, `min_prob = 0.5`, anchor-quantile `α = 0.05`

### 5.2 5-config × 3-seed ablation on snr8_14 (holdout)

| config | acc (mean±std) | macro_f1 | collapses | chirp | pulse | tone | comb | ofdm | fsk |
|---|---|---|---|---|---|---|---|---|---|
| `mag` (naive R=G=B) | 0.7903 ± 0.267 | 0.764 | 1/3 | 0.64 | 0.60 | 0.96 | 0.93 | 0.91 | 0.62 |
| `mag_1ch` (**main**) | **0.9264 ± 0.023** | 0.959 | 0/3 | 0.97 | 0.87 | 0.91 | 0.92 | 0.94 | 0.94 |
| `mag_if` | 0.8868 ± 0.111 | 0.926 | 1/3 | 0.93 | 0.92 | 0.97 | 0.95 | 0.95 | 0.67 |
| `mag_gd` | 0.9385 ± 0.017 | 0.967 | 0/3 | 0.96 | 0.93 | 0.93 | 0.92 | 0.91 | 0.98 |
| `mag_if_gd` (full 3ch) | 0.9427 ± 0.015 | 0.969 | 0/3 | 0.97 | 0.92 | 0.91 | 0.95 | 0.94 | 0.94 |

Key take-aways (documented in full in `_archive/` task notes):

- The `mag → mag_gd` +14.8 pp gap decomposes as **~13.6 pp from stem surgery** (`mag → mag_1ch`) and only ~1.2 pp from GD phase information (`mag_1ch → mag_gd`). The headline "3-channel beats 1-channel" narrative was mostly stem surgery.
- Seed stability is owned by stem surgery, not phase. `mag` and `mag_if` both have 1/3 seed collapses (seed=43 specifically). `mag_1ch`, `mag_gd`, `mag_if_gd` are all 0/3 collapses.
- IF preserves tone better (0.97 vs 0.93 for GD); GD stabilizes FSK (0.98 vs 0.67 for IF). Keeping both at once (`mag_if_gd`) is the best average but is the alternative/ablation path — not the shipped main.

### 5.3 Main pipeline output directory convention

- `C:\Users\Zhuoer\code\thesis\Anomaly_Localization\outputs\sas_impulsive_only_snr8_14_mag_1ch_seed{42,43,44}\`

Each contains `metrics.json`, `holdout_predictions.csv`, `update_events.json`, `update_01/cluster_summary.json`, confusion matrices, and the resolved `config.json`.

## 6. Threshold Protocol Change Relevant to Coverage

One major experimental change was the threshold selection protocol:

- old static protocol:
  - post-hoc scan on evaluation set
- aligned protocol:
  - threshold = `95th percentile` of training-set energy

This protocol change makes static evaluation more comparable to open-world `pre_update`.

## 7. Historical approaches (code archived)

The following directions were implemented at least once and then dropped. Their source code lives under `_archive/`, not on the active import path:

| Approach | Archived file |
|---|---|
| PatchCore + OCSVM + HDBSCAN open-world (pre-ConvNeXt) | `_archive/sas_open_world/experiments/open_world.py` |
| Per-class OCSVM gate open-set | `_archive/sas_open_world/experiments/open_set_gate.py` |
| OCSVM tone-gamma sweep | `_archive/sas_open_world/experiments/tone_gamma.py` |
| Known-class HDBSCAN separability diagnostic | `_archive/sas_open_world/experiments/convnextv2_known_hdbscan.py` |
| OCSVM helpers (`train_open_set_model`, `predict_label`) | `_archive/sas_open_world/openset/ocsvm.py` |
| Old training sweeps (mcs50_unsup_*, v5_*, v6_*, hdbscan_sweep) | `_archive/scripts/` |

Also tried in earlier iterations but not retained in code: Feature AE/VAE/PER variants, per-class energy threshold, single-prototype distance gate, DINOv3-as-classifier, Mahalanobis distance, ArcFace+CE, Center Loss, multiple early-stopping schemes.

Note: the generic helper `select_anchor_indices` was relocated from `ocsvm.py` to `sas_open_world/common.py` during cleanup — it is still called by the main pipeline.

## 8. Runtime Environment

### 8.1 Python environment

The reference environment is the conda env `IIS`. Two pinned-dependency files are committed at the repo root and are the canonical record of what produced the reference numbers in §12:

- `environment.yml` — `conda env export --from-history` snapshot (channels, conda-installed packages including `python=3.12.12`)
- `requirements-frozen.txt` — `pip freeze` snapshot (full transitive closure including `torch==2.9.1+cu130`, `timm==1.0.21`, `numpy==1.26.4`, `hdbscan==0.8.41`, `umap-learn==0.5.11`, `scikit-learn==1.7.2`)

Recreate from scratch:

```bash
# either:
conda env create -n IIS -f environment.yml
# then ensure pip-installed packages match exactly:
conda activate IIS
pip install -r requirements-frozen.txt

# or, simpler if you have an empty Python 3.12 env:
pip install -r requirements-frozen.txt
```

Current working interpreter on the development machine:

```powershell
C:\Users\Zhuoer\anaconda3\envs\IIS\python.exe        # = python 3.12.12
```

Hardware / driver during reference runs:

- GPU: `NVIDIA GeForce RTX 5070 Ti`
- NVIDIA driver: `591.86`
- CUDA runtime (via torch wheel): `cu130` (CUDA 13.0)
- OS: Windows 11 + Git Bash (the bash scripts in `scripts/` assume Git Bash semantics, e.g. forward slashes in `cd` and POSIX-style `/c/Users/...` paths)

### 8.1.1 Determinism caveats

Stage 1 (PatchCore) is **bit-exact reproducible** across runs on the same machine: identical AUROC / TN / FP / FN / TP / threshold across the original mag_1ch ablation and the post-refactor validation run on April 25 (see §12). The coreset-greedy memory-bank fit is deterministic given a numpy seed.

Stage 2 (ConvNeXtV2 open-world) is **not** bit-exact reproducible across runs even with identical CLI, identical Stage 1 outputs, identical seeds, and identical code. Drift envelope observed in practice on this hardware: ~±0.5–2pp on holdout `macro_f1_all_classes`. Sources:

- `torch.backends.cudnn.benchmark = True` is the default; cuDNN picks the fastest convolution algorithm per input shape, and that pick is non-deterministic across driver/library state.
- Atomic accumulators in cuBLAS / cuDNN (e.g., when computing convolution backward) are non-associative under floating-point reordering.
- `WeightedRandomSampler` consumes `torch`'s global RNG in a way that interacts with model parameter init order.

For an exact-match reproduction of Stage 2, set `torch.use_deterministic_algorithms(True)` and `cudnn.benchmark=False`, expect a ~30 % slowdown, and accept that the `mag_1ch` 3-seed mean will land within **0.95–0.98** macro-F1 — *not* exactly the reference 0.9735. The structural metrics (initial/final known classes, num_updates, processed_samples, gate_mode) are deterministic and must match the §12 reference exactly.

### 8.2 Repo root

```powershell
cd C:\Users\Zhuoer\code\thesis\Anomaly_Localization
```

## 9. Run Commands

### 9.1 Main config: mag_1ch, 3 seeds

Driver script: `scripts/ablation_mag_1ch.sh`. To reproduce a single seed manually (bash):

```bash
PYEXE="/c/Users/Zhuoer/anaconda3/envs/IIS/python.exe"
DATASET="C:/Users/Zhuoer/code/thesis/Datasets/SAS/rf_dataset_packed_impulsive_only_snr8_14_sirm8_4"
PATCHCORE="./outputs/sas_impulsive_only_snr8_14_sirm8_4_patchcore_wide_resnet50_2_256_zscore_allpatches"

"$PYEXE" simulate_sas_convnextv2_open_world.py \
  --dataset-dir "$DATASET" \
  --output-dir ./outputs/sas_impulsive_only_snr8_14_mag_1ch_seed42 \
  --patchcore-output-dir "$PATCHCORE" \
  --input-mode roi \
  --input-channels mag_1ch \
  --cluster-dim-reduction umap \
  --cluster-min-cluster-size 50 \
  --cluster-selection-method eom \
  --cluster-accept-min-size 20 \
  --cluster-accept-min-prob 0.5 \
  --cluster-anchor-quantile-alpha 0.05 \
  --seed 42
```

### 9.2 Ablation: mag_if / mag_gd / mag_if_gd

Driver script: `scripts/ablation_2ch_vs_3ch.sh`. Same command as §9.1 with `--input-channels` replaced by one of `mag_if`, `mag_gd`, or `mag_if_gd`.

### 9.3 Stage 1: PatchCore precompute

Driver script: `scripts/impulsive_only_snr14_20_patchcore.sh` (snr14_20 example). Use equivalent parameters with the snr8_14 dataset path for the main pipeline's PatchCore source. The resulting `anomaly_outputs.npz` is what `--patchcore-output-dir` consumes.

### 9.4 Static open-set baseline

The static ROI-only open-set baseline (`evaluate_sas_open_set_convnextv2.py`) is still in the repo and still works on the snr8_14 dataset, but it has not been re-tuned for the 3-channel pack. Treat its numbers as reference only until retuned.

## 10. Execution Notes and Caveats

### 10.1 ROI modes require PatchCore output

If `--input-mode roi` is used, `--patchcore-output-dir` must be valid.

### 10.2 Use fresh output directories

Do not reuse output directories across incompatible runs.

### 10.3 Static metrics depend heavily on threshold protocol

There are two different static evaluation modes in this codebase:

- post-hoc threshold search
- train-calibrated threshold

Do not compare them directly without noting which protocol was used.

### 10.4 mag_1ch has its own data path

`--input-channels mag_1ch` loads the legacy `test_spectrogram.npy` / `train_spectrogram.npy` (uint8 [0,255]), not `*_spectrogram_3c.npy`. The dataset pack must contain both or `mag_1ch` will fail. All other modes consume the 3-channel packed arrays.

### 10.5 Zero-channel ablation implementation detail

`mag_if` / `mag_gd` zero the unused channel **after** ImageNet normalization, not before. Zeroing before normalization would feed `(0 − mean) / std ≈ nonzero` into the stem, which is not the intended ablation. See `ThreeChannelTransform` in `sas_open_world/experiments/convnextv2_open_world.py`.

## 11. Repository Layout After Cleanup

Active pipeline files (see the file list produced during cleanup):

- Entry: `simulate_sas_convnextv2_open_world.py`
- Stage 2 driver (CLI + `main` only): `sas_open_world/experiments/convnextv2_open_world.py`
- Stage 2 model package: `sas_open_world/models/{transforms,dataset,convnextv2}.py` — see table below
- Static open-set reference: `evaluate_sas_open_set_convnextv2.py` + `sas_open_world/experiments/convnextv2_open_set.py`
- Few-shot subtype classifier on PatchCore features: `run_fewshot_patchcore_classifier_sas.py` + `sas_open_world/experiments/fewshot_patchcore_classifier.py` (moved from `support/` when that directory was collapsed; reusable feature builder `build_semantic_feature` lives in `sas_open_world/features/patchcore.py`)
- Stage 1 driver: `train_eval_patchcore_sas.py`
- Stage 1 core (dataset-neutral): `core/utils.py` (`log`, `set_seed`, `signed_log1p_np`, `compute_finite_mean_std`), `core/patchcore.py` (`ArrayDataset2D`, `PatchCore`, `ResNetFeatureExtractor`), `core/viz.py`, `core/metrics.py`
- Stage 1 dataset loader: `dataio/sas/bundle.py` (unified SAS disk format)
- Shared (Stage 2): `sas_open_world/common.py`, `clustering/unknown_buffer.py`, `features/{patchcore,dino}.py`, `pipeline/open_world.py`
- Scripts: `scripts/{ablation_mag_1ch,ablation_2ch_vs_3ch,impulsive_only_snr14_20_patchcore}.sh`

**Stage 2 model package** (`sas_open_world.models`): the historical 1276-line `experiments/convnextv2_open_world.py` monolith was split along three planes; the experiment file now only owns CLI + `main`.

| Module | Contents | Why this plane |
|---|---|---|
| `models/transforms.py` | `GaussianNoiseAugment`, `ThreeChannelTransform`, `MagOneChannelTransform`, `transform_is_mag_1ch_native`, `_apply_stem_surgery_for_{1,3}channel` | Input normalization + stem surgery are coupled: the surgery variant is determined by the channel mode, so they share a file. |
| `models/dataset.py` | `SpectrogramDataset` (3D / 2D-native / legacy PIL-RGB dispatch), `build_loader` | Pure DataLoader plumbing; no model or training code. |
| `models/convnextv2.py` | `build_protocol`, `create_model`, `train_model`, `infer_with_embeddings`, `predict_sample`, `train_convnext_open_world_model`, `_build_balanced_energy_values`, `_l2_normalize`, `_compute_class_distances` | All ConvNeXtV2-specific construction, training, calibration, and gate-aware inference. The orchestrator (`train_convnext_open_world_model`) is the reusable entry point for new experiments. |

**Unified SAS loader API** (`dataio.sas`, formerly `datasets.sas`; renamed to avoid shadowing HuggingFace `datasets`): a single dataclass-returning entry point per access pattern; all downstream code (Stage 1 driver, Stage 2 experiments, features, clustering, fewshot support) goes through these. No module hand-rolls `np.load("test_images.npy")` anymore.

| Function | Returns | Use when |
|---|---|---|
| `load_sas_bundle(dir)` | `SASBundle(train_images, train_labels, test_images, test_labels)` | Stage 1 PatchCore training (needs train split) |
| `load_sas_test_split(dir, *, load_images=True)` | `SASTestSplit(label_names, images, count)` | 1-channel test-set consumers; `load_images=False` for labels-only |
| `load_sas_test_split_3c(dir)` | `SASTestSplit3C(label_names, images, count)` | 3-channel `mag_if` / `mag_gd` / `mag_if_gd` consumers; requires `test_images_3c.npy` |
| `load_channel_stats(dir)` | `dict` of per-channel mean/std JSON | 3-channel normalization; raises with producer-script guidance if absent |

Layout principle: **dataset-neutral functional modules live in `core/`, dataset-specific loaders live in `dataio/<name>/`**. Drivers at the repo root are thin — they parse CLI args, call the dataset loader, then call the core. The old `train_eval_patchcore_wsad.py` (which bundled model + dataset + viz + dead WSAD CLI in one 886-line file) and `sas_localization_metrics.py` (top-level) were split accordingly; both are under `_archive/` for reference.

`sas_open_world/data/` (formerly held `build_flow_holdout_split` / `build_initial_training_indices` for an older two-step protocol) was deleted: its only callers were archived and the active pipeline uses `sas_open_world.models.convnextv2.build_protocol` instead.

Everything under `_archive/` is historical reference and is **not on the Python import path**. The `sas_open_world/openset/` package was archived in full; its one still-used helper (`select_anchor_indices`) is now in `sas_open_world/common.py`.

## 12. Reference Numbers (refactor-validation anchors, April 25 2026)

These are the canonical numbers an LLM agent / new contributor should reproduce after recreating the env (§8.1) and regenerating the dataset (§2.0.5). They were produced by `scripts/refactor_check_mag_1ch.sh` against the post-refactor codebase (Scope B Stage 1 split, Scope C Stage 2 split, `dataio/` rename, `support/` collapse). Side-by-side with the pre-refactor `scripts/ablation_mag_1ch.sh` runs from before the restructure, all structural metrics matched bit-for-bit; numerical drift is within the cuDNN-nondeterminism envelope (§8.1.1).

### 12.1 Stage 1 PatchCore (deterministic — must match exactly)

Run: `train_eval_patchcore_sas.py` with the parameters in `scripts/refactor_check_mag_1ch.sh` against `rf_dataset_packed_impulsive_only_snr8_14_sirm8_4`.

| Field | Reference value |
|---|---|
| `auroc` | 0.9993586805555555 |
| `threshold` | 39.10743713378906 |
| `tn` / `fp` / `fn` / `tp` | 1195 / 5 / 17 / 4783 |
| `tpr` / `tnr` | 0.9964583333333333 / 0.9958333333333333 |
| `train_count` / `test_count` | 4000 / 6000 |
| `train_mean` / `train_std` | 79.45186611557007 / 13.423637817751391 |
| `embedding_dim` / `total_train_patches` / `memory_bank_size` | 1536 / 4096000 / 200 |

If any of `auroc`, `threshold`, the four confusion-matrix counts, or `memory_bank_size` deviates, something is wrong (likely a numpy/scipy version mismatch or a bug in the coreset selection). `fit_seconds` and `predict_seconds` are timing fields and not part of the equality contract.

### 12.2 Stage 2 ConvNeXtV2 open-world, `mag_1ch`, 3 seeds (within ±2pp envelope)

Run: `simulate_sas_convnextv2_open_world.py` with `--input-channels mag_1ch --seed {42,43,44}` and the cluster knobs in `scripts/refactor_check_mag_1ch.sh`. Each seed reads the §12.1 Stage 1 output.

**Structural metrics — must match exactly across all three seeds:**

| Field | Reference value |
|---|---|
| `initial_known_classes` | `[chirp, pulse, tone]` |
| `final_known_classes` | `[chirp, pulse, tone, ofdm, comb, fsk]` (any order) |
| `num_updates` | 1 |
| `final_gate_mode` | `energy_global` |
| `processed_samples` | 3600 |

If a seed discovers fewer than all 3 emerging classes, fires more than one update event, or processes a different sample count, something is structurally wrong (likely a clustering hyperparameter mismatch, a dataio bug, or a Stage 1 anomaly map mismatch).

**Numerical metrics — should match within drift envelope:**

`holdout.macro_f1_all_classes` per seed and 3-seed mean ± std:

| Run | seed 42 | seed 43 | seed 44 | mean ± std (n=3) |
|---|---:|---:|---:|---:|
| Pre-refactor (`*_mag_1ch_seed{42,43,44}/`) | 0.9743 | 0.9522 | 0.9514 | 0.9593 ± 0.0130 |
| Post-refactor (`*_mag_1ch_refactor_seed{42,43,44}/`) | 0.9775 | 0.9704 | 0.9728 | 0.9735 ± 0.0036 |

Acceptance criterion for a fresh reproduction on a different machine: 3-seed mean within **0.94 – 0.99**, all individual seeds within **0.92 – 0.99**, no class missing from `final_known_classes`. The 0.014 mean shift between pre- and post-refactor is within the cuDNN drift envelope (§8.1.1) and is not a code-change effect — `set_seed` was simplified (dropped a `random.seed()` call) but the Python `random` module is dead code on the open-world path; Stage 1 outputs are bit-exact between runs.

### 12.3 Output directory provenance

| Reference output dir | What it pins |
|---|---|
| `outputs/sas_impulsive_only_snr8_14_sirm8_4_patchcore_refactor_check/` | §12.1 Stage 1 numbers |
| `outputs/sas_impulsive_only_snr8_14_mag_1ch_refactor_seed{42,43,44}/` | §12.2 Stage 2 numbers (post-refactor column) |
| `outputs/sas_impulsive_only_snr8_14_sirm8_4_patchcore_wide_resnet50_2_256_zscore_allpatches/` | Pre-refactor Stage 1 (kept for diff against §12.1; bit-exact match) |
| `outputs/sas_impulsive_only_snr8_14_mag_1ch_seed{42,43,44}/` | Pre-refactor Stage 2 (kept for diff against §12.2; pre-refactor column) |

If you regenerate the dataset and rerun, expect Stage 1 to be bit-exact against §12.1 (assuming pinned-version numpy/scipy/torch); Stage 2 will land somewhere in the §12.2 envelope. If the new Stage 1 is *not* bit-exact, suspect first that the dataset itself is different — diff `train_mean` / `train_std` against §12.1; if those drift, your `build_rf_dataset.py --seed` or your numpy/scipy version is different from the reference.
