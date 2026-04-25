# Core functional modules shared across datasets.
#
# Layout:
#   core.utils      — generic helpers (log, set_seed, signed_log1p_np, finite mean/std)
#   core.patchcore  — PatchCore model, feature extractor, ArrayDataset2D wrapper
#   core.viz        — AUROC/CM/ROC plotting, score CSV, anomaly montage
#   core.metrics    — pixel-AUROC and AUPRO localization metrics
