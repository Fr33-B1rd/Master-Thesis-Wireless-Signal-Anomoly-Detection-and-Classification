# Anomaly Localization

This repository contains PatchCore-based experiments for RF spectrogram anomaly detection, localization, clustering, and open-set analysis.

## Scope

The code is organized around three dataset families:

- `IAD`: anomaly detection on PKL-based modulation datasets
- `WSAD`: PatchCore experiments on packed array splits
- `SAS`: spectrogram anomaly detection, clustering, and open-set rejection

The main workflow is:

1. Train `PatchCore` on normal training data.
2. Evaluate anomaly scores and localization maps on test data.
3. Optionally export patch-level artifacts.
4. Reuse those artifacts for clustering or downstream open-set analysis.

## Main Scripts

- [`train_eval_patchcore.py`](C:/Users/Zhuoer/code/thesis/Anomaly_Localization/train_eval_patchcore.py)  
  PatchCore training and evaluation for the original IAD PKL datasets.

- [`train_eval_patchcore_iad_shared.py`](C:/Users/Zhuoer/code/thesis/Anomaly_Localization/train_eval_patchcore_iad_shared.py)  
  Shared-feature-space PatchCore for combined IAD normal training sets.

- [`train_eval_patchcore_wsad.py`](C:/Users/Zhuoer/code/thesis/Anomaly_Localization/train_eval_patchcore_wsad.py)  
  PatchCore implementation for WSAD array datasets.

- [`train_eval_patchcore_sas.py`](C:/Users/Zhuoer/code/thesis/Anomaly_Localization/train_eval_patchcore_sas.py)  
  PatchCore training and evaluation for packed SAS spectrogram datasets.

- [`cluster_patchcore_anomalies.py`](C:/Users/Zhuoer/code/thesis/Anomaly_Localization/cluster_patchcore_anomalies.py)  
  Clustering pipeline for IAD anomaly samples.

- [`cluster_patchcore_sas.py`](C:/Users/Zhuoer/code/thesis/Anomaly_Localization/cluster_patchcore_sas.py)  
  Clustering pipeline for SAS anomaly samples.

- [`evaluate_sas_open_set_ocsvm_gate.py`](C:/Users/Zhuoer/code/thesis/Anomaly_Localization/evaluate_sas_open_set_ocsvm_gate.py)  
  Open-set evaluation for SAS using a known-class classifier plus per-class OCSVM rejection.

- [`evaluate_sas_open_set_ocsvm_tone_gamma.py`](C:/Users/Zhuoer/code/thesis/Anomaly_Localization/evaluate_sas_open_set_ocsvm_tone_gamma.py)  
  Targeted sweep of the tone-class OCSVM gamma in the SAS open-set pipeline.

## Typical Outputs

Generated results are written to [`outputs`](C:/Users/Zhuoer/code/thesis/Anomaly_Localization/outputs) and usually include:

- metrics JSON files
- score CSV files
- ROC and confusion matrix figures
- anomaly heatmap montages
- clustering summaries and UMAP plots
- exported patch artifacts for downstream analysis

## Notes

- The `support` directory stores secondary or exploratory utilities that are not part of the main root workflow.
- Most scripts are command-line entry points and are intended to be run directly with Python.
- The current project emphasis is on PatchCore feature reuse rather than end-to-end deep classification.
