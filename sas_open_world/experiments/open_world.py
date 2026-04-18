"""Entry point for SAS open-world simulation with buffering and retraining."""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from sas_open_world.clustering import cluster_unknown_buffer
from sas_open_world.common import log, save_confusion
from sas_open_world.data import build_flow_holdout_split, build_initial_training_indices
from sas_open_world.features import build_feature_cache
from sas_open_world.openset import predict_label, select_anchor_indices, train_open_set_model
from sas_open_world.pipeline import build_stream_indices, compute_holdout_metrics, compute_phase_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pseudo-online SAS open-world simulation with unknown buffering and retraining"
    )
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--patchcore-output-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--known-classes", default="chirp,pulse,tone")
    parser.add_argument("--emerging-classes", default="comb,ofdm")
    parser.add_argument("--ignore-classes", default="fsk")
    parser.add_argument("--flow-per-class", type=int, default=600)
    parser.add_argument("--holdout-per-class", type=int, default=200)
    parser.add_argument("--shot", type=int, default=60)
    parser.add_argument("--feature-mode", default="mean_plus_gram", choices=["mean", "gram", "mean_plus_gram"])
    parser.add_argument("--top-k-patches", type=int, default=64)
    parser.add_argument("--gram-dim", type=int, default=32)
    parser.add_argument("--classifier", default="logreg", choices=["logreg", "linear_svm"])
    parser.add_argument("--classifier-c", type=float, default=1.0)
    parser.add_argument("--class-weight", default="none", choices=["none", "balanced"])
    parser.add_argument("--augment-entropy", action="store_true")
    parser.add_argument("--augment-frequency-profile", action="store_true")
    parser.add_argument("--nu", type=float, default=0.06)
    parser.add_argument("--target-class", default="tone")
    parser.add_argument("--target-gamma-multiplier", type=float, default=0.5)
    parser.add_argument("--gate-threshold", type=float, default=-0.35)
    parser.add_argument("--unknown-buffer-size", type=int, default=400)
    parser.add_argument("--anchor-per-class", type=int, default=30)
    parser.add_argument("--cluster-pca-dim", type=int, default=8)
    parser.add_argument("--cluster-min-cluster-size", type=int, default=8)
    parser.add_argument("--cluster-min-samples", type=int, default=1)
    parser.add_argument("--cluster-accept-purity", type=float, default=0.9)
    parser.add_argument("--cluster-accept-min-size", type=int, default=40)
    parser.add_argument("--cluster-max-anchor-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_cache_path(args: argparse.Namespace) -> str:
    suffix_parts = [args.feature_mode, "open_world"]
    if args.augment_entropy:
        suffix_parts.append("entropy")
    if args.augment_frequency_profile:
        suffix_parts.append("fprofile")
    return os.path.join(args.output_dir, f"feature_cache_{'_'.join(suffix_parts)}.npz")


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    known_classes = [part.strip() for part in args.known_classes.split(",") if part.strip()]
    emerging_classes = [part.strip() for part in args.emerging_classes.split(",") if part.strip()]
    rng = np.random.default_rng(args.seed)
    label_array = np.load(os.path.join(args.dataset_dir, "test_labels.npy")).astype(str)
    detector_scores = pd.read_csv(os.path.join(args.patchcore_output_dir, "test_scores.csv"))
    detector_predictions = detector_scores["prediction"].astype(int).to_numpy()

    x_all, cache_labels = build_feature_cache(
        dataset_dir=args.dataset_dir,
        patchcore_output_dir=args.patchcore_output_dir,
        feature_mode=args.feature_mode,
        top_k_patches=args.top_k_patches,
        gram_dim=args.gram_dim,
        seed=args.seed,
        cache_path=build_cache_path(args),
        augment_temporal_diff=False,
        augment_dynamic_range=False,
        augment_entropy=args.augment_entropy,
        augment_frequency_profile=args.augment_frequency_profile,
    )
    if not np.array_equal(label_array, cache_labels.astype(str)):
        raise ValueError("Feature cache labels do not align with dataset labels")

    split_classes = [*known_classes, *emerging_classes]
    flow_indices_by_label, holdout_indices_by_label, split_summary, normal_stream_indices = build_flow_holdout_split(
        label_array=label_array,
        split_classes=split_classes,
        flow_per_class=args.flow_per_class,
        holdout_per_class=args.holdout_per_class,
        rng=rng,
    )
    train_indices_by_label, initial_train_indices = build_initial_training_indices(
        flow_indices_by_label=flow_indices_by_label,
        known_classes=known_classes,
        shot=args.shot,
    )
    stream_indices = build_stream_indices(
        flow_indices_by_label=flow_indices_by_label,
        split_classes=split_classes,
        initial_train_indices=initial_train_indices,
        normal_stream_indices=normal_stream_indices,
        rng=rng,
    )

    with open(os.path.join(args.output_dir, "split_protocol.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "flow_per_class": args.flow_per_class,
                "holdout_per_class": args.holdout_per_class,
                "shot": args.shot,
                "split_summary": split_summary,
            },
            f,
            indent=2,
        )

    current_known_classes = list(known_classes)
    model = train_open_set_model(
        x_all=x_all,
        train_indices_by_label=train_indices_by_label,
        known_classes=current_known_classes,
        classifier_name=args.classifier,
        classifier_c=args.classifier_c,
        class_weight_name=args.class_weight,
        target_class=args.target_class,
        target_gamma_multiplier=args.target_gamma_multiplier,
        nu=args.nu,
        seed=args.seed,
    )
    log(f"Initial model trained with classes: {current_known_classes}")

    processed_rows = []
    unknown_buffer: list[int] = []
    update_events = []
    phase_name = "pre_update"
    update_id = 0

    for stream_pos, sample_index in enumerate(stream_indices):
        true_label = str(label_array[sample_index])
        pred_label, gate_score = predict_label(
            model=model,
            x_vector=x_all[sample_index],
            detector_prediction=int(detector_predictions[sample_index]),
            gate_threshold=args.gate_threshold,
        )

        processed_rows.append(
            {
                "stream_position": stream_pos,
                "sample_index": sample_index,
                "phase": phase_name,
                "true_label": true_label,
                "pred_label": pred_label,
                "detector_prediction": int(detector_predictions[sample_index]),
                "gate_score": gate_score,
            }
        )

        if pred_label == "unknown" and int(detector_predictions[sample_index]) == 1:
            unknown_buffer.append(sample_index)

        while len(unknown_buffer) >= args.unknown_buffer_size:
            update_id += 1
            buffer_slice = unknown_buffer[: args.unknown_buffer_size]
            anchor_indices = select_anchor_indices(
                train_indices_by_label=train_indices_by_label,
                current_known_classes=current_known_classes,
                excluded_indices=buffer_slice,
                anchor_per_class=args.anchor_per_class,
                seed=args.seed + update_id,
            )
            assignment_df, accepted_by_label, cluster_summary = cluster_unknown_buffer(
                x_buffer=x_all[buffer_slice],
                true_labels=label_array[buffer_slice],
                buffer_sample_indices=np.asarray(buffer_slice, dtype=np.int64),
                x_anchor=x_all[anchor_indices],
                anchor_labels=label_array[anchor_indices],
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
            )

            event_dir = os.path.join(args.output_dir, f"update_{update_id:02d}")
            os.makedirs(event_dir, exist_ok=True)
            assignment_df.to_csv(os.path.join(event_dir, "cluster_assignments.csv"), index=False)

            discovered_classes = []
            accepted_counts = {}
            for class_name, local_indices in accepted_by_label.items():
                global_indices = [buffer_slice[local_idx] for local_idx in sorted(set(local_indices))]
                train_indices_by_label.setdefault(class_name, [])
                train_indices_by_label[class_name].extend(global_indices)
                train_indices_by_label[class_name] = sorted(set(train_indices_by_label[class_name]))
                accepted_counts[class_name] = len(global_indices)
                if class_name not in current_known_classes:
                    current_known_classes.append(class_name)
                    discovered_classes.append(class_name)

            cluster_summary["accepted_counts"] = accepted_counts
            cluster_summary["discovered_classes"] = discovered_classes
            cluster_summary["buffer_sample_indices"] = buffer_slice
            with open(os.path.join(event_dir, "cluster_summary.json"), "w", encoding="utf-8") as f:
                json.dump(cluster_summary, f, indent=2)

            update_events.append(
                {
                    "update_id": update_id,
                    "trigger_stream_position": stream_pos,
                    "buffer_size": len(buffer_slice),
                    "discovered_classes": discovered_classes,
                    "accepted_counts": accepted_counts,
                    "cluster_summary_path": os.path.join(event_dir, "cluster_summary.json"),
                }
            )

            if discovered_classes:
                model = train_open_set_model(
                    x_all=x_all,
                    train_indices_by_label=train_indices_by_label,
                    known_classes=current_known_classes,
                    classifier_name=args.classifier,
                    classifier_c=args.classifier_c,
                    class_weight_name=args.class_weight,
                    target_class=args.target_class,
                    target_gamma_multiplier=args.target_gamma_multiplier,
                    nu=args.nu,
                    seed=args.seed,
                )
                phase_name = f"post_update_{update_id:02d}"
                log(f"Update {update_id}: retrained with classes {current_known_classes}")
            else:
                log(f"Update {update_id}: no new class accepted from buffer")

            unknown_buffer = unknown_buffer[args.unknown_buffer_size :]

    records_df = pd.DataFrame(processed_rows)
    records_df.to_csv(os.path.join(args.output_dir, "stream_predictions.csv"), index=False)
    with open(os.path.join(args.output_dir, "update_events.json"), "w", encoding="utf-8") as f:
        json.dump(update_events, f, indent=2)

    metrics = {
        "processed_samples": int(records_df.shape[0]),
        "initial_known_classes": known_classes,
        "final_known_classes": current_known_classes,
        "num_updates": len(update_events),
        "residual_unknown_buffer": int(len(unknown_buffer)),
    }
    metrics["overall"] = compute_phase_metrics(records_df, known_classes=known_classes, emerging_classes=emerging_classes)
    if (records_df["phase"] == "pre_update").any():
        metrics["pre_update"] = compute_phase_metrics(
            records_df[records_df["phase"] == "pre_update"],
            known_classes=known_classes,
            emerging_classes=emerging_classes,
        )
    post_mask = records_df["phase"].str.startswith("post_update_")
    if post_mask.any():
        metrics["post_update"] = compute_phase_metrics(
            records_df[post_mask],
            known_classes=known_classes + [c for c in emerging_classes if c in current_known_classes],
            emerging_classes=emerging_classes,
        )

    holdout_indices = []
    for class_name in split_classes:
        holdout_indices.extend(holdout_indices_by_label[class_name])
    holdout_rows = []
    for sample_index in holdout_indices:
        pred_label, gate_score = predict_label(
            model=model,
            x_vector=x_all[sample_index],
            detector_prediction=int(detector_predictions[sample_index]),
            gate_threshold=args.gate_threshold,
        )
        holdout_rows.append(
            {
                "sample_index": int(sample_index),
                "true_label": str(label_array[sample_index]),
                "pred_label": pred_label,
                "detector_prediction": int(detector_predictions[sample_index]),
                "gate_score": gate_score,
            }
        )
    holdout_df = pd.DataFrame(holdout_rows)
    holdout_df.to_csv(os.path.join(args.output_dir, "holdout_predictions.csv"), index=False)
    metrics["holdout"] = compute_holdout_metrics(
        records_df=holdout_df,
        final_known_classes=current_known_classes,
        all_eval_classes=split_classes,
    )

    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    label_order = ["normal", "unknown", *known_classes, *emerging_classes]
    save_confusion(
        y_true=records_df["true_label"].tolist(),
        y_pred=records_df["pred_label"].tolist(),
        label_order=label_order,
        output_path=os.path.join(args.output_dir, "overall_confusion_matrix.png"),
        title="Open-World Stream Confusion Matrix",
    )
    save_confusion(
        y_true=holdout_df["true_label"].tolist(),
        y_pred=holdout_df["pred_label"].tolist(),
        label_order=["unknown", *split_classes],
        output_path=os.path.join(args.output_dir, "holdout_confusion_matrix.png"),
        title="Final Holdout Confusion Matrix",
    )
    log(f"Finished open-world simulation with {len(update_events)} update(s)")


if __name__ == "__main__":
    main()
