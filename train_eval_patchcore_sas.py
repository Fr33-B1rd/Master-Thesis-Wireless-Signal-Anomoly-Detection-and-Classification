# Trains and evaluates PatchCore on packed SAS spectrogram datasets, with optional
# export of test-time patch artifacts for clustering or open-set analysis.

import argparse
import json
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_eval_patchcore_wsad import (
    WSADArrayDataset,
    WSADPatchCore,
    compute_finite_mean_std,
    evaluate_scores,
    log,
    save_anomaly_montage,
    save_confusion_matrix,
    save_roc_curve,
    save_score_csv,
    set_seed,
)


@dataclass
class SASBundle:
    train_data: np.ndarray
    test_data: np.ndarray
    train_labels: np.ndarray
    test_labels: np.ndarray
    test_label_names: np.ndarray
    train_count: int
    test_count: int
    mean: float
    std: float


def load_sas_bundle(dataset_dir: str) -> SASBundle:
    train_data = np.load(
        os.path.join(dataset_dir, "train_spectrogram.npy"),
        mmap_mode="r",
    )
    test_data = np.load(
        os.path.join(dataset_dir, "test_spectrogram.npy"),
        mmap_mode="r",
    )
    train_label_names = np.load(os.path.join(dataset_dir, "train_labels.npy"))
    test_label_names = np.load(os.path.join(dataset_dir, "test_labels.npy"))

    if train_data.ndim != 3 or test_data.ndim != 3:
        raise ValueError("Expected SAS spectrogram arrays with shape [N, H, W]")

    train_label_names = np.asarray(train_label_names).astype(str)
    test_label_names = np.asarray(test_label_names).astype(str)
    train_labels = (train_label_names != "normal").astype(np.int64)
    test_labels = (test_label_names != "normal").astype(np.int64)

    train_unique = sorted(np.unique(train_label_names).tolist())
    test_unique = sorted(np.unique(test_label_names).tolist())
    if train_unique != ["normal"]:
        raise ValueError(
            f"SAS PatchCore expects normal-only training data, got train labels: {train_unique}"
        )
    if "normal" not in test_unique or len(test_unique) < 2:
        raise ValueError(
            "Expected SAS test split to contain normal plus at least one abnormal class"
        )

    mean, std = compute_finite_mean_std(train_data, use_log_transform=False)
    if std <= 0:
        raise ValueError("Training data std must be > 0")

    return SASBundle(
        train_data=train_data,
        test_data=test_data,
        train_labels=train_labels,
        test_labels=test_labels,
        test_label_names=test_label_names,
        train_count=int(train_data.shape[0]),
        test_count=int(test_data.shape[0]),
        mean=mean,
        std=std,
    )


def save_label_breakdown(
    label_names: np.ndarray,
    scores: np.ndarray,
    output_path: str,
) -> None:
    unique_labels = sorted(np.unique(label_names).tolist())
    rows = []
    for label_name in unique_labels:
        mask = label_names == label_name
        rows.append(
            {
                "label_name": label_name,
                "count": int(mask.sum()),
                "mean_score": float(scores[mask].mean()),
                "std_score": float(scores[mask].std()),
            }
        )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate PatchCore on SAS packed spectrogram data"
    )
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--backbone",
        default="wide_resnet50_2",
        choices=["resnet18", "wide_resnet50_2"],
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        default=["layer2", "layer3"],
        choices=["layer1", "layer2", "layer3", "layer4"],
    )
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clip-z", type=float, default=5.0)
    parser.add_argument("--sample-patches-per-image", type=int, default=32)
    parser.add_argument("--candidate-pool-size", type=int, default=20000)
    parser.add_argument("--coreset-ratio", type=float, default=0.01)
    parser.add_argument("--max-memory-bank-size", type=int, default=2048)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--query-chunk-size", type=int, default=2048)
    parser.add_argument("--num-vis", type=int, default=10)
    parser.add_argument("--mask-top-rows", type=int, default=0)
    parser.add_argument("--mask-bottom-rows", type=int, default=0)
    parser.add_argument("--mask-left-cols", type=int, default=0)
    parser.add_argument("--mask-right-cols", type=int, default=0)
    parser.add_argument("--export-test-patch-artifacts", action="store_true")
    parser.add_argument(
        "--export-selection",
        default="label1",
        choices=["all", "label1"],
    )
    parser.add_argument(
        "--export-embedding-dtype",
        default="float16",
        choices=["float16", "float32"],
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)

    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    bundle = load_sas_bundle(args.dataset_dir)
    log(
        f"SAS loaded: train={bundle.train_count}, test={bundle.test_count}, "
        f"raw_shape={tuple(bundle.train_data.shape[1:])}, "
        f"train mean={bundle.mean:.6f}, train std={bundle.std:.6f}, "
        f"mask(top={args.mask_top_rows}, bottom={args.mask_bottom_rows}, "
        f"left={args.mask_left_cols}, right={args.mask_right_cols})"
    )

    train_dataset = WSADArrayDataset(
        data_parts=[bundle.train_data],
        labels=bundle.train_labels,
        mean=bundle.mean,
        std=bundle.std,
        input_size=args.input_size,
        clip_z=args.clip_z,
        normalization="zscore",
        mask_top_rows=args.mask_top_rows,
        mask_bottom_rows=args.mask_bottom_rows,
        mask_left_cols=args.mask_left_cols,
        mask_right_cols=args.mask_right_cols,
    )
    test_dataset = WSADArrayDataset(
        data_parts=[bundle.test_data],
        labels=bundle.test_labels,
        mean=bundle.mean,
        std=bundle.std,
        input_size=args.input_size,
        clip_z=args.clip_z,
        normalization="zscore",
        mask_top_rows=args.mask_top_rows,
        mask_bottom_rows=args.mask_bottom_rows,
        mask_left_cols=args.mask_left_cols,
        mask_right_cols=args.mask_right_cols,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = WSADPatchCore(
        backbone=args.backbone,
        layers=args.layers,
        device=device,
        sample_patches_per_image=args.sample_patches_per_image,
        candidate_pool_size=args.candidate_pool_size,
        coreset_ratio=args.coreset_ratio,
        max_memory_bank_size=args.max_memory_bank_size,
        projection_dim=args.projection_dim,
        query_chunk_size=args.query_chunk_size,
        seed=args.seed,
        pretrained=not args.no_pretrained,
    )

    fit_start = time.time()
    model.fit(train_loader)
    fit_seconds = time.time() - fit_start

    predict_start = time.time()
    scores, labels, indices, anomaly_maps = model.predict(test_loader)
    predict_seconds = time.time() - predict_start

    if not np.array_equal(indices, np.arange(indices.shape[0])):
        raise ValueError("Unexpected test ordering")

    eval_result = evaluate_scores(labels, scores)
    tpr = float(eval_result["tp"] / max(eval_result["tp"] + eval_result["fn"], 1))
    tnr = float(eval_result["tn"] / max(eval_result["tn"] + eval_result["fp"], 1))
    save_roc_curve(labels, scores, os.path.join(args.output_dir, "roc_curve.png"))
    save_confusion_matrix(
        eval_result["confusion_matrix"],
        os.path.join(args.output_dir, "confusion_matrix.png"),
    )
    vis_indices = save_anomaly_montage(
        dataset=test_dataset,
        anomaly_maps=anomaly_maps,
        scores=scores,
        labels=labels,
        output_path=os.path.join(args.output_dir, "anomaly_montage_per_type_top5.png"),
        num_images=5,
        label_names=bundle.test_label_names,
        title_prefix="Top 5 Test Samples Per Anomaly Type",
    )
    save_score_csv(
        scores,
        labels,
        eval_result["predictions"],
        os.path.join(args.output_dir, "test_scores.csv"),
    )
    np.savez_compressed(
        os.path.join(args.output_dir, "anomaly_outputs.npz"),
        scores=scores,
        labels=labels,
        anomaly_maps=anomaly_maps.astype(np.float16),
        label_names=bundle.test_label_names.astype("U16"),
    )
    patch_export_info = None
    if args.export_test_patch_artifacts:
        patch_export_info = model.export_test_patch_artifacts(
            test_loader=test_loader,
            output_path=os.path.join(args.output_dir, "test_patch_artifacts.npz"),
            selection=args.export_selection,
            embedding_dtype=args.export_embedding_dtype,
        )
    save_label_breakdown(
        bundle.test_label_names,
        scores,
        os.path.join(args.output_dir, "test_label_breakdown.json"),
    )

    metrics = {
        "dataset_dir": args.dataset_dir,
        "output_dir": args.output_dir,
        "backbone": args.backbone,
        "layers": args.layers,
        "device": args.device,
        "input_size": args.input_size,
        "train_count": bundle.train_count,
        "test_count": bundle.test_count,
        "raw_input_shape": list(bundle.train_data.shape[1:]),
        "normalization": "zscore",
        "train_mean": bundle.mean,
        "train_std": bundle.std,
        "mask_top_rows": args.mask_top_rows,
        "mask_bottom_rows": args.mask_bottom_rows,
        "mask_left_cols": args.mask_left_cols,
        "mask_right_cols": args.mask_right_cols,
        "embedding_dim": model.embedding_dim,
        "patch_grid_size": model.patch_grid_size,
        "total_train_patches": model.total_train_patches,
        "memory_bank_size": int(model.memory_bank.shape[0]),
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "auroc": eval_result["auroc"],
        "threshold": eval_result["threshold"],
        "tn": eval_result["tn"],
        "fp": eval_result["fp"],
        "fn": eval_result["fn"],
        "tp": eval_result["tp"],
        "tpr": tpr,
        "tnr": tnr,
        "selected_visualization_indices": vis_indices,
        "patch_export": patch_export_info,
        "test_label_distribution": {
            label_name: int((bundle.test_label_names == label_name).sum())
            for label_name in sorted(np.unique(bundle.test_label_names).tolist())
        },
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    torch.save(
        {
            "memory_bank": model.memory_bank.cpu(),
            "layers": args.layers,
            "embedding_dim": model.embedding_dim,
            "patch_grid_size": model.patch_grid_size,
            "input_size": args.input_size,
            "normalization": "zscore",
            "train_mean": bundle.mean,
            "train_std": bundle.std,
            "mask_top_rows": args.mask_top_rows,
            "mask_bottom_rows": args.mask_bottom_rows,
            "mask_left_cols": args.mask_left_cols,
            "mask_right_cols": args.mask_right_cols,
        },
        os.path.join(args.output_dir, "patchcore_memory_bank.pt"),
    )

    log(
        f"Finished SAS PatchCore. AUROC={metrics['auroc']:.4f}, "
        f"TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']} TP={metrics['tp']}"
    )
    log(f"Artifacts saved to {args.output_dir}")


if __name__ == "__main__":
    main()
