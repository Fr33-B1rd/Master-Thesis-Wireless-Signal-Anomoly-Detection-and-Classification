# Anomaly Localization

ConvNeXtV2 open-world pipeline for SAS RF spectrogram anomaly detection and unknown-class discovery. Stage 1 uses PatchCore for ROI localization; Stage 2 is a 1-channel ConvNeXtV2-tiny classifier with energy-based open-set gating and HDBSCAN+UMAP unknown-class absorption.

For operational details (dataset paths, input-channel modes, run commands, results, caveats) see [HANDOFF.md](HANDOFF.md).

## Repository layout

```
.
├── HANDOFF.md                              # operational source of truth
├── simulate_sas_convnextv2_open_world.py   # main entry (Stage 2 open-world)
├── evaluate_sas_open_set_convnextv2.py     # static open-set baseline
├── train_eval_patchcore_sas.py             # Stage 1 PatchCore driver (CLI + main)
├── core/                                   # dataset-neutral functional modules
│   ├── patchcore.py                        #   PatchCore model, ArrayDataset2D, feature extractor
│   ├── viz.py                              #   AUROC/ROC/CM plotting, score CSV, anomaly montage
│   └── metrics.py                          #   pixel-AUROC + AUPRO localization metrics
├── dataio/                                 # dataset-specific loaders (renamed from datasets/
│   └── sas/bundle.py                       #     to avoid shadowing HuggingFace `datasets`)
├── cluster_patchcore_sas.py                # patch-feature clustering driver
├── sas_open_world/                         # Stage 2 package (incl. models/, experiments/)
├── scripts/                                # shell drivers for repeatable runs
├── outputs/                                # all run artifacts (not versioned)
└── _archive/                               # retired code and scripts, not on import path
```

## Main entry point

```bash
python simulate_sas_convnextv2_open_world.py \
  --dataset-dir <path-to-packed-SAS-dataset> \
  --output-dir <run-output-dir> \
  --patchcore-output-dir <stage1-patchcore-output-dir> \
  --input-mode roi \
  --input-channels mag_1ch \
  --seed 42
```

Reproducible 3-seed runs live in `scripts/ablation_mag_1ch.sh` (main) and `scripts/ablation_2ch_vs_3ch.sh` (ablation).

## Notes

- Stage 1 PatchCore output (`anomaly_outputs.npz`) must exist before any ROI-mode Stage 2 run.
- `outputs/`, `__pycache__/`, and raw `*.npy` / `*.npz` / `*.pt` artifacts are excluded from version control (see `.gitignore`).
- Historical code for OCSVM gates, PatchCore-only open-world, DINO feature clustering, and old IAD PatchCore variants is preserved under `_archive/` but is not importable.
