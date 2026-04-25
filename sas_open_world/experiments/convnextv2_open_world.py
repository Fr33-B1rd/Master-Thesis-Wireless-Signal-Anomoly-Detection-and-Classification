"""Pseudo-online open-world experiment with ConvNeXtV2 classification and unknown-buffer clustering.

Driver for the Stage 2 ConvNeXtV2 open-world pipeline. All reusable pieces
(transforms, stem surgery, dataset wrappers, model training, inference, and
the ``train_convnext_open_world_model`` orchestrator) live in
``sas_open_world.models``; this file owns only the CLI surface and the
top-level orchestration that ties loaders, model, stream inference, unknown
clustering, and holdout evaluation together.
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import torch

from core.utils import set_seed
from dataio.sas import (
    load_channel_stats,
    load_sas_test_split,
    load_sas_test_split_3c,
)
from sas_open_world.clustering import cluster_unknown_buffer
from sas_open_world.common import log, save_confusion, select_anchor_indices
from sas_open_world.experiments.convnextv2_open_set import build_roi_boxes
from sas_open_world.features import build_dino_feature_cache, build_feature_cache
from sas_open_world.models.convnextv2 import (
    build_protocol,
    predict_sample,
    train_convnext_open_world_model,
)
from sas_open_world.pipeline import compute_holdout_metrics, compute_phase_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pseudo-online ConvNeXtV2 open-world experiment on SAS anomaly classes"
    )
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--known-classes", default="chirp,pulse,tone")
    parser.add_argument("--emerging-classes", default="comb,ofdm,fsk")
    parser.add_argument("--shot", type=int, default=80)
    parser.add_argument("--flow-per-class", type=int, default=600)
    parser.add_argument("--anchor-per-class", type=int, default=30)
    parser.add_argument("--model-name", default="convnextv2_tiny")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--gate-mode", default="energy_global", choices=["energy_global", "distance"])
    parser.add_argument("--threshold-quantile", type=float, default=95.0)
    parser.add_argument("--energy-calibration", default="balanced", choices=["balanced", "pooled"])
    parser.add_argument("--distance-quantile", type=float, default=95.0)
    parser.add_argument("--distance-metric", default="cosine", choices=["cosine", "l2"])
    parser.add_argument("--input-mode", default="full", choices=["full", "roi"])
    parser.add_argument("--roi-quantile", type=float, default=99.0)
    parser.add_argument("--roi-pad-ratio", type=float, default=0.10)
    parser.add_argument("--roi-min-side", type=int, default=32)
    parser.add_argument(
        "--clustering-feature-source",
        default="convnext",
        choices=["convnext", "patchcore_mean_plus_gram", "dino"],
    )
    parser.add_argument("--patchcore-output-dir")
    parser.add_argument("--dino-model-name", default="vit_small_patch16_dinov3")
    parser.add_argument("--dino-batch-size", type=int, default=32)
    parser.add_argument("--top-k-patches", type=int, default=64)
    parser.add_argument("--gram-dim", type=int, default=32)
    parser.add_argument("--cluster-pca-dim", type=int, default=8)
    parser.add_argument("--cluster-min-cluster-size", type=int, default=50)
    parser.add_argument("--cluster-min-samples", type=int, default=1)
    parser.add_argument("--cluster-accept-purity", type=float, default=0.9)
    parser.add_argument("--cluster-accept-min-size", type=int, default=40)
    parser.add_argument("--cluster-max-anchor-fraction", type=float, default=0.10)
    parser.add_argument("--cluster-dim-reduction", default="pca", choices=["pca", "umap"])
    parser.add_argument("--cluster-umap-n-neighbors", type=int, default=15)
    parser.add_argument("--cluster-umap-min-dist", type=float, default=0.0)
    parser.add_argument("--cluster-selection-method", default="eom", choices=["eom", "leaf"])
    parser.add_argument("--cluster-accept-min-prob", type=float, default=-1.0,
                        help="Unsupervised acceptance: min mean HDBSCAN membership probability. "
                             "Set >=0 to enable unsupervised mode (disables oracle purity check). "
                             "Default -1 = legacy oracle mode.")
    parser.add_argument("--cluster-anchor-null-alpha", type=float, default=-1.0,
                        help="Scale-free known-class-leakage test. If >=0, replaces "
                             "--cluster-max-anchor-fraction with a per-class hypergeometric "
                             "test: reject cluster if any known class anchor count is "
                             "significantly higher than random at level alpha/M (Bonferroni "
                             "over M known classes). Default -1 = use absolute fraction threshold.")
    parser.add_argument("--anchor-core-prob-threshold", type=float, default=0.5,
                        help="Under --cluster-anchor-null-alpha, reject leakage only when "
                             "mean HDBSCAN membership probability of the cluster's anchors "
                             "is >= this threshold (i.e. anchors are core members, not "
                             "periphery points dragged in by UMAP).")
    parser.add_argument("--cluster-anchor-quantile-alpha", type=float, default=-1.0,
                        help="Statistically rigorous anchor_fraction threshold. Per-cluster "
                             "threshold = hypergeom.ppf(1-alpha, N, K_a, n) / n, where K_a "
                             "is total anchor count and n is cluster size. Equivalent to a "
                             "one-sided permutation test on total anchor_fraction at level "
                             "alpha. Default -1 = disabled.")
    parser.add_argument("--cluster-anchor-fraction-scale", type=float, default=-1.0,
                        help="Scale-free reformulation of --cluster-max-anchor-fraction. "
                             "If >=0, reject cluster when anchor_fraction exceeds "
                             "scale x (n_anchor_total / n_total). Physical meaning: "
                             "'cluster has more than <scale>x the null-expected anchor "
                             "density'. Auto-scales with anchor count and buffer size. "
                             "Default -1 = use absolute --cluster-max-anchor-fraction.")
    parser.add_argument("--cluster-only", action="store_true")
    parser.add_argument("--train-noise-std", type=float, default=0.05,
                        help="Std of Gaussian noise added to normalized spectrogram "
                             "tensors during classifier training (initial + retrain). "
                             "Not applied during calibration, stream inference, or "
                             "holdout eval. Set 0 to disable.")
    parser.add_argument("--input-channels", default="mag",
                        choices=["mag", "mag_1ch", "mag_if", "mag_gd", "mag_if_gd"],
                        help="Classifier input representation. 'mag' = legacy "
                             "single-channel magnitude replicated to RGB (no "
                             "stem surgery; baseline that suffers R=G=B stem "
                             "collapse). 'mag_1ch' = true 1-channel: replaces "
                             "first Conv2d with in_channels=1 initialized from "
                             "the SUM of pretrained RGB weights, isolates the "
                             "effect of proper 1-channel fine-tuning from the "
                             "phase-information contribution of IF/GD. "
                             "'mag_if_gd' = 3 physically distinct channels "
                             "(log-magnitude, instantaneous frequency, group "
                             "delay). 'mag_if' / 'mag_gd' = 2-channel ablation: "
                             "loads the 3c pack, keeps Ch0 + the named phase "
                             "channel, zeros the other phase channel AFTER "
                             "normalization so stem filters on it contribute 0. "
                             "All non-'mag'/'mag_1ch' modes read "
                             "<dataset-dir>/test_spectrogram_3c.npy with "
                             "per-channel stats from channel_stats.json. "
                             "PatchCore (stage-1) is unaffected in any mode.")
    parser.add_argument("--stem-perturb-std", type=float, default=0.01,
                        help="Scale of Gaussian perturbation added to each "
                             "stem-channel slice when initializing the 3-channel "
                             "stem from averaged pretrained RGB weights. Relative "
                             "to the pretrained weight std. Only used when "
                             "--input-channels=mag_if_gd.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    known_classes = [part.strip() for part in args.known_classes.split(",") if part.strip()]
    emerging_classes = [part.strip() for part in args.emerging_classes.split(",") if part.strip()]
    all_eval_classes = [*known_classes, *emerging_classes]

    channel_stats: dict | None = None
    if args.input_channels in {"mag_if", "mag_gd", "mag_if_gd"}:
        split_3c = load_sas_test_split_3c(args.dataset_dir)
        channel_stats = load_channel_stats(args.dataset_dir)
        images = split_3c.images
        label_names = split_3c.label_names
    elif args.input_channels == "mag_1ch":
        # True 1-channel: 2D magnitude array + Ch0 stats from channel_stats.json.
        split = load_sas_test_split(args.dataset_dir)
        channel_stats = load_channel_stats(args.dataset_dir)
        images = split.images
        label_names = split.label_names
    else:
        split = load_sas_test_split(args.dataset_dir)
        images = split.images
        label_names = split.label_names
    roi_boxes = None
    if args.input_mode == "roi":
        if not args.patchcore_output_dir:
            raise ValueError("--patchcore-output-dir is required when input-mode=roi")
        anomaly_outputs = np.load(os.path.join(args.patchcore_output_dir, "anomaly_outputs.npz"))
        anomaly_maps = anomaly_outputs["anomaly_maps"]
        roi_boxes = build_roi_boxes(
            anomaly_maps=anomaly_maps,
            sample_indices=np.arange(label_names.shape[0], dtype=np.int64),
            quantile=args.roi_quantile,
            pad_ratio=args.roi_pad_ratio,
            min_side=args.roi_min_side,
        )
        roi_heights = np.asarray([box.bottom - box.top for box in roi_boxes.values()], dtype=np.int64)
        roi_widths = np.asarray([box.right - box.left for box in roi_boxes.values()], dtype=np.int64)
        with open(os.path.join(args.output_dir, "roi_summary.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "num_boxes": int(len(roi_boxes)),
                    "quantile": args.roi_quantile,
                    "pad_ratio": args.roi_pad_ratio,
                    "min_side": args.roi_min_side,
                    "mean_height": float(np.mean(roi_heights)),
                    "mean_width": float(np.mean(roi_widths)),
                    "min_height": int(np.min(roi_heights)),
                    "max_height": int(np.max(roi_heights)),
                    "min_width": int(np.min(roi_widths)),
                    "max_width": int(np.max(roi_widths)),
                },
                f,
                indent=2,
            )
    x_cluster_all = None
    if args.clustering_feature_source == "patchcore_mean_plus_gram":
        if not args.patchcore_output_dir:
            raise ValueError("--patchcore-output-dir is required when clustering-feature-source=patchcore_mean_plus_gram")
        cache_path = os.path.join(
            args.output_dir,
            f"feature_cache_patchcore_mean_plus_gram_topk{args.top_k_patches}_gram{args.gram_dim}.npz",
        )
        x_cluster_all, cache_labels = build_feature_cache(
            dataset_dir=args.dataset_dir,
            patchcore_output_dir=args.patchcore_output_dir,
            feature_mode="mean_plus_gram",
            top_k_patches=args.top_k_patches,
            gram_dim=args.gram_dim,
            seed=args.seed,
            cache_path=cache_path,
            augment_temporal_diff=False,
            augment_dynamic_range=False,
            augment_entropy=False,
            augment_frequency_profile=False,
        )
        if not np.array_equal(label_names, cache_labels.astype(str)):
            raise ValueError("PatchCore feature cache labels do not align with dataset labels")
    elif args.clustering_feature_source == "dino":
        cache_path = os.path.join(
            args.output_dir,
            f"feature_cache_dino_{args.dino_model_name.replace('/', '_')}.npz",
        )
        x_cluster_all, cache_labels = build_dino_feature_cache(
            dataset_dir=args.dataset_dir,
            dino_model_name=args.dino_model_name,
            cache_path=cache_path,
            batch_size=args.dino_batch_size,
            device=device,
        )
        if not np.array_equal(label_names, cache_labels.astype(str)):
            raise ValueError("DINO feature cache labels do not align with dataset labels")

    (
        train_indices_by_label,
        flow_indices_by_label,
        holdout_indices_by_label,
        split_summary,
        stream_indices,
    ) = build_protocol(
        label_names=label_names,
        known_classes=known_classes,
        emerging_classes=emerging_classes,
        shot=args.shot,
        flow_per_class=args.flow_per_class,
        rng=rng,
    )
    with open(os.path.join(args.output_dir, "split_protocol.json"), "w", encoding="utf-8") as f:
        json.dump(split_summary, f, indent=2)

    current_known_classes = list(known_classes)
    (
        model,
        transform,
        class_to_id,
        gate_info,
        history,
        train_class_counts,
    ) = train_convnext_open_world_model(
        images=images,
        label_names=label_names,
        train_indices_by_label=train_indices_by_label,
        current_known_classes=current_known_classes,
        model_name=args.model_name,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        gate_mode=args.gate_mode,
        threshold_quantile=args.threshold_quantile,
        energy_calibration=args.energy_calibration,
        distance_quantile=args.distance_quantile,
        distance_metric=args.distance_metric,
        device=device,
        roi_boxes=roi_boxes,
        train_noise_std=args.train_noise_std,
        input_channels=args.input_channels,
        channel_stats=channel_stats,
        stem_perturb_std=args.stem_perturb_std,
        stem_surgery_seed=args.seed,
    )
    model_class_names = [name for name, _ in sorted(class_to_id.items(), key=lambda item: item[1])]
    with open(os.path.join(args.output_dir, "initial_train_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(args.output_dir, "initial_train_class_counts.json"), "w", encoding="utf-8") as f:
        json.dump(train_class_counts, f, indent=2)

    processed_rows = []
    unknown_sample_indices: list[int] = []
    unknown_embeddings: list[np.ndarray] = []
    update_events = []

    # Pre-update stream pass: classify each sample; collect those flagged as unknown.
    for stream_pos, sample_index in enumerate(stream_indices):
        true_label = str(label_names[sample_index])
        pred_label, energy, embedding = predict_sample(
            model=model,
            transform=transform,
            image=np.asarray(images[sample_index]),
            model_class_names=model_class_names,
            gate_info=gate_info,
            temperature=args.temperature,
            device=device,
            roi_box=None if roi_boxes is None else roi_boxes.get(int(sample_index)),
        )
        processed_rows.append(
            {
                "stream_position": int(stream_pos),
                "sample_index": int(sample_index),
                "phase": "pre_update",
                "true_label": true_label,
                "pred_label": pred_label,
                "energy": energy,
            }
        )
        if pred_label == "unknown":
            unknown_sample_indices.append(int(sample_index))
            unknown_embeddings.append(embedding)

    # End-of-pre-update single clustering event with known-class anchors.
    if unknown_sample_indices:
        update_id = 1
        if x_cluster_all is None:
            x_buffer = np.stack(unknown_embeddings, axis=0).astype(np.float32)
        else:
            x_buffer = x_cluster_all[np.asarray(unknown_sample_indices, dtype=np.int64)].astype(np.float32, copy=False)
        anchor_indices = select_anchor_indices(
            train_indices_by_label=train_indices_by_label,
            current_known_classes=current_known_classes,
            excluded_indices=unknown_sample_indices,
            anchor_per_class=args.anchor_per_class,
            seed=args.seed + update_id,
        )
        if x_cluster_all is None:
            x_anchor_list = []
            for anchor_idx in anchor_indices:
                _, _, anchor_embedding = predict_sample(
                    model=model,
                    transform=transform,
                    image=np.asarray(images[anchor_idx]),
                    model_class_names=model_class_names,
                    gate_info=gate_info,
                    temperature=args.temperature,
                    device=device,
                    roi_box=None if roi_boxes is None else roi_boxes.get(int(anchor_idx)),
                )
                x_anchor_list.append(anchor_embedding)
            x_anchor = (
                np.stack(x_anchor_list, axis=0).astype(np.float32)
                if x_anchor_list
                else np.empty((0, x_buffer.shape[1]), dtype=np.float32)
            )
        else:
            x_anchor = x_cluster_all[np.asarray(anchor_indices, dtype=np.int64)].astype(np.float32, copy=False)

        assignment_df, accepted_by_label, cluster_summary = cluster_unknown_buffer(
            x_buffer=x_buffer,
            true_labels=label_names[np.asarray(unknown_sample_indices, dtype=np.int64)],
            buffer_sample_indices=np.asarray(unknown_sample_indices, dtype=np.int64),
            x_anchor=x_anchor,
            anchor_labels=label_names[np.asarray(anchor_indices, dtype=np.int64)],
            anchor_sample_indices=np.asarray(anchor_indices, dtype=np.int64),
            current_known_classes=current_known_classes,
            emerging_classes=emerging_classes,
            pca_dim=args.cluster_pca_dim,
            min_cluster_size=args.cluster_min_cluster_size,
            min_samples=args.cluster_min_samples,
            cluster_accept_purity=args.cluster_accept_purity,
            cluster_accept_min_size=args.cluster_accept_min_size,
            cluster_max_anchor_fraction=args.cluster_max_anchor_fraction,
            seed=args.seed,
            dim_reduction=args.cluster_dim_reduction,
            umap_n_neighbors=args.cluster_umap_n_neighbors,
            umap_min_dist=args.cluster_umap_min_dist,
            cluster_selection_method=args.cluster_selection_method,
            cluster_accept_min_prob=args.cluster_accept_min_prob,
            cluster_anchor_null_alpha=args.cluster_anchor_null_alpha,
            anchor_core_prob_threshold=args.anchor_core_prob_threshold,
            cluster_anchor_fraction_scale=args.cluster_anchor_fraction_scale,
            cluster_anchor_quantile_alpha=args.cluster_anchor_quantile_alpha,
        )

        event_dir = os.path.join(args.output_dir, f"update_{update_id:02d}")
        os.makedirs(event_dir, exist_ok=True)
        assignment_df.to_csv(os.path.join(event_dir, "cluster_assignments.csv"), index=False)

        discovered_classes = []
        accepted_counts = {}
        for class_name, local_indices in accepted_by_label.items():
            global_indices = [unknown_sample_indices[local_idx] for local_idx in sorted(set(local_indices))]
            train_indices_by_label.setdefault(class_name, [])
            train_indices_by_label[class_name].extend(global_indices)
            train_indices_by_label[class_name] = sorted(set(train_indices_by_label[class_name]))
            accepted_counts[class_name] = len(global_indices)
            if class_name not in current_known_classes:
                current_known_classes.append(class_name)
                discovered_classes.append(class_name)

        cluster_summary["accepted_counts"] = accepted_counts
        cluster_summary["discovered_classes"] = discovered_classes
        cluster_summary["buffer_sample_indices"] = list(unknown_sample_indices)
        with open(os.path.join(event_dir, "cluster_summary.json"), "w", encoding="utf-8") as f:
            json.dump(cluster_summary, f, indent=2)

        if args.cluster_only:
            log(f"cluster-only mode: stopping after clustering (discovered={discovered_classes}, accepted={accepted_counts})")
            return

        update_events.append(
            {
                "update_id": update_id,
                "trigger_stream_position": int(len(stream_indices) - 1),
                "unknown_count": len(unknown_sample_indices),
                "discovered_classes": discovered_classes,
                "accepted_counts": accepted_counts,
            }
        )

        if discovered_classes:
            (
                model,
                transform,
                class_to_id,
                gate_info,
                history,
                train_class_counts,
            ) = train_convnext_open_world_model(
                images=images,
                label_names=label_names,
                train_indices_by_label=train_indices_by_label,
                current_known_classes=current_known_classes,
                model_name=args.model_name,
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                temperature=args.temperature,
                gate_mode=args.gate_mode,
                threshold_quantile=args.threshold_quantile,
                energy_calibration=args.energy_calibration,
                distance_quantile=args.distance_quantile,
                distance_metric=args.distance_metric,
                device=device,
                roi_boxes=roi_boxes,
                train_noise_std=args.train_noise_std,
                input_channels=args.input_channels,
                channel_stats=channel_stats,
                stem_perturb_std=args.stem_perturb_std,
                stem_surgery_seed=args.seed + update_id,
            )
            model_class_names = [name for name, _ in sorted(class_to_id.items(), key=lambda item: item[1])]
            with open(os.path.join(event_dir, "retrain_history.json"), "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
            with open(os.path.join(event_dir, "train_class_counts.json"), "w", encoding="utf-8") as f:
                json.dump(train_class_counts, f, indent=2)
            log(
                f"Update {update_id}: discovered {discovered_classes}, "
                f"retrained with classes {current_known_classes}, "
                f"gate_mode={gate_info['mode']}, "
                f"gate_metric={gate_info['metric']}, "
                f"global_gate_value={gate_info['global']:.4f}"
            )
        else:
            log(f"Update {update_id}: no new class accepted from unknown set")

    records_df = pd.DataFrame(processed_rows)
    records_df.to_csv(os.path.join(args.output_dir, "stream_predictions.csv"), index=False)
    with open(os.path.join(args.output_dir, "update_events.json"), "w", encoding="utf-8") as f:
        json.dump(update_events, f, indent=2)

    holdout_indices = []
    for class_name in all_eval_classes:
        holdout_indices.extend(holdout_indices_by_label[class_name])
    holdout_rows = []
    for sample_index in holdout_indices:
        pred_label, energy, _ = predict_sample(
            model=model,
            transform=transform,
            image=np.asarray(images[sample_index]),
            model_class_names=model_class_names,
            gate_info=gate_info,
            temperature=args.temperature,
            device=device,
            roi_box=None if roi_boxes is None else roi_boxes.get(int(sample_index)),
        )
        row = {
            "sample_index": int(sample_index),
            "true_label": str(label_names[sample_index]),
            "pred_label": pred_label,
            "energy": energy,
        }
        holdout_rows.append(row)
    holdout_df = pd.DataFrame(holdout_rows)
    holdout_df.to_csv(os.path.join(args.output_dir, "holdout_predictions.csv"), index=False)

    metrics = {
        "processed_samples": int(records_df.shape[0]),
        "initial_known_classes": known_classes,
        "final_known_classes": current_known_classes,
        "num_updates": len(update_events),
        "final_gate_mode": str(gate_info["mode"]),
        "final_gate_metric": str(gate_info["metric"]),
        "final_global_gate_value": float(gate_info["global"]),
        "final_gate_calibration": str(gate_info.get("calibration", "")),
        "final_gate_calibration_counts": {
            str(class_name): int(value)
            for class_name, value in gate_info.get("calibration_counts", {}).items()
        },
        "final_distance_thresholds_by_class": {
            str(class_name): float(value)
            for class_name, value in gate_info.get("by_class", {}).items()
        },
        "overall": compute_phase_metrics(
            records_df=records_df,
            known_classes=known_classes,
            emerging_classes=emerging_classes,
        ),
        "holdout": compute_holdout_metrics(
            records_df=holdout_df,
            final_known_classes=current_known_classes,
            all_eval_classes=all_eval_classes,
        ),
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    save_confusion(
        y_true=records_df["true_label"].tolist(),
        y_pred=records_df["pred_label"].tolist(),
        label_order=["unknown", *all_eval_classes],
        output_path=os.path.join(args.output_dir, "stream_confusion_matrix.png"),
        title="ConvNeXt Open-World Stream Confusion Matrix",
    )
    save_confusion(
        y_true=holdout_df["true_label"].tolist(),
        y_pred=holdout_df["pred_label"].tolist(),
        label_order=["unknown", *all_eval_classes],
        output_path=os.path.join(args.output_dir, "holdout_confusion_matrix.png"),
        title="ConvNeXt Open-World Holdout Confusion Matrix",
    )
    log(f"Finished ConvNeXtV2 open-world experiment with {len(update_events)} update(s)")


if __name__ == "__main__":
    main()
