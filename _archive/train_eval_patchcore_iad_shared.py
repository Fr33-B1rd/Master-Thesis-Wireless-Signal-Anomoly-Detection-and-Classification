# Trains one shared PatchCore model on the combined normal IAD training splits and
# evaluates that shared feature space on each modulation subset separately.

import argparse
import json
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from train_eval_patchcore import (
    IADSpectrogramDataset,
    PatchCoreModel,
    evaluate_scores,
    load_iad_dataset,
    log,
    save_anomaly_montage,
    save_confusion_matrix,
    save_roc_curve,
    save_score_csv,
    set_seed,
)


@dataclass
class NamedBundle:
    name: str
    dataset_pkl: str
    train_data: np.ndarray
    train_label: np.ndarray
    test_data: np.ndarray
    test_label: np.ndarray


def parse_dataset_specs(specs: list[str]) -> list[tuple[str, str]]:
    parsed = []
    for spec in specs:
        if "::" not in spec:
            raise ValueError(f"Invalid dataset spec: {spec}")
        name, dataset_pkl = spec.split("::", 1)
        parsed.append((name, dataset_pkl))
    return parsed


def compute_shared_mean_std(train_arrays: list[np.ndarray]) -> tuple[float, float]:
    total_sum = 0.0
    total_sq_sum = 0.0
    total_count = 0
    for array in train_arrays:
        values = np.asarray(array, dtype=np.float32)
        total_sum += float(values.sum(dtype=np.float64))
        total_sq_sum += float(np.square(values, dtype=np.float64).sum(dtype=np.float64))
        total_count += int(values.size)
    mean = total_sum / total_count
    variance = max(total_sq_sum / total_count - mean * mean, 1e-12)
    return float(mean), float(np.sqrt(variance))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train one shared PatchCore on merged IAD normal sets and evaluate per dataset"
    )
    parser.add_argument(
        "--dataset-specs",
        nargs="+",
        required=True,
        help="Repeated specs of the form DATASET_NAME::PATH_TO_PKL",
    )
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
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clip-z", type=float, default=5.0)
    parser.add_argument("--coreset-ratio", type=float, default=0.01)
    parser.add_argument("--max-memory-bank-size", type=int, default=2048)
    parser.add_argument("--candidate-pool-size", type=int, default=20000)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--query-chunk-size", type=int, default=2048)
    parser.add_argument("--num-vis", type=int, default=10)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--no-pretrained", action="store_true")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    dataset_specs = parse_dataset_specs(args.dataset_specs)
    bundles: list[NamedBundle] = []
    train_arrays = []
    train_labels = []

    for name, dataset_pkl in dataset_specs:
        log(f"Loading {name} from {dataset_pkl}")
        bundle = load_iad_dataset(dataset_pkl)
        bundles.append(
            NamedBundle(
                name=name,
                dataset_pkl=dataset_pkl,
                train_data=bundle.train_data,
                train_label=bundle.train_label,
                test_data=bundle.test_data,
                test_label=bundle.test_label,
            )
        )
        train_arrays.append(bundle.train_data)
        train_labels.append(bundle.train_label)

    shared_train_data = np.concatenate(train_arrays, axis=0).astype(np.float32, copy=False)
    shared_train_label = np.concatenate(train_labels, axis=0).astype(np.int64, copy=False)
    shared_mean, shared_std = compute_shared_mean_std(train_arrays)

    root_metrics = {
        "datasets": [bundle.name for bundle in bundles],
        "train_samples_total": int(shared_train_data.shape[0]),
        "shared_train_mean": shared_mean,
        "shared_train_std": shared_std,
        "backbone": args.backbone,
        "layers": args.layers,
        "device": args.device,
    }
    with open(os.path.join(args.output_dir, "shared_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(root_metrics, f, indent=2)

    train_dataset = IADSpectrogramDataset(
        shared_train_data,
        shared_train_label,
        mean=shared_mean,
        std=shared_std,
        clip_z=args.clip_z,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = PatchCoreModel(
        backbone=args.backbone,
        layers=args.layers,
        device=device,
        coreset_ratio=args.coreset_ratio,
        max_memory_bank_size=args.max_memory_bank_size,
        candidate_pool_size=args.candidate_pool_size,
        projection_dim=args.projection_dim,
        query_chunk_size=args.query_chunk_size,
        seed=args.seed,
        pretrained=not args.no_pretrained,
    )

    log(f"Building shared PatchCore memory bank on device={device}")
    fit_start = time.time()
    model.fit(train_loader)
    fit_seconds = time.time() - fit_start

    torch.save(
        {
            "memory_bank": model.memory_bank.cpu(),
            "layers": args.layers,
            "embedding_dim": model.embedding_dim,
            "patch_grid_size": model.patch_grid_size,
            "mean": shared_mean,
            "std": shared_std,
            "clip_z": args.clip_z,
            "datasets": [bundle.name for bundle in bundles],
        },
        os.path.join(args.output_dir, "shared_patchcore_memory_bank.pt"),
    )

    summary_rows = []
    for bundle in bundles:
        dataset_output_dir = os.path.join(args.output_dir, bundle.name)
        os.makedirs(dataset_output_dir, exist_ok=True)

        test_dataset = IADSpectrogramDataset(
            bundle.test_data,
            bundle.test_label,
            mean=shared_mean,
            std=shared_std,
            clip_z=args.clip_z,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

        log(f"Scoring shared-space test set for {bundle.name}")
        predict_start = time.time()
        scores, labels, indices, anomaly_maps = model.predict(test_loader)
        predict_seconds = time.time() - predict_start

        if not np.array_equal(indices, np.arange(indices.shape[0])):
            raise ValueError(f"{bundle.name}: unexpected test index ordering")

        eval_result = evaluate_scores(labels, scores)
        save_roc_curve(labels, scores, os.path.join(dataset_output_dir, "roc_curve.png"))
        save_confusion_matrix(
            eval_result["confusion_matrix"],
            os.path.join(dataset_output_dir, "confusion_matrix.png"),
        )
        selected_vis_indices = save_anomaly_montage(
            raw_test_data=bundle.test_data,
            anomaly_maps=anomaly_maps,
            scores=scores,
            labels=labels,
            output_path=os.path.join(dataset_output_dir, "anomaly_montage_top10.png"),
            num_images=args.num_vis,
        )
        save_score_csv(
            scores,
            labels,
            eval_result["predictions"],
            os.path.join(dataset_output_dir, "test_scores.csv"),
        )
        np.savez_compressed(
            os.path.join(dataset_output_dir, "anomaly_outputs.npz"),
            scores=scores,
            labels=labels,
            anomaly_maps=anomaly_maps,
        )

        patch_export_info = None
        if args.export_test_patch_artifacts:
            log(f"Exporting shared-space patch artifacts for {bundle.name}")
            patch_export_info = model.export_test_patch_artifacts(
                test_loader=test_loader,
                output_path=os.path.join(dataset_output_dir, "test_patch_artifacts.npz"),
                selection=args.export_selection,
                embedding_dtype=args.export_embedding_dtype,
            )

        dataset_metrics = {
            "dataset_name": bundle.name,
            "dataset_pkl": bundle.dataset_pkl,
            "shared_output_root": args.output_dir,
            "device": args.device,
            "backbone": args.backbone,
            "layers": args.layers,
            "train_samples_total": int(shared_train_data.shape[0]),
            "test_samples": int(bundle.test_data.shape[0]),
            "shared_train_mean": shared_mean,
            "shared_train_std": shared_std,
            "embedding_dim": model.embedding_dim,
            "patch_grid_size": model.patch_grid_size,
            "total_train_patches": model.total_train_patches,
            "memory_bank_size": int(model.memory_bank.shape[0]),
            "fit_seconds_shared": fit_seconds,
            "predict_seconds": predict_seconds,
            "auroc": eval_result["auroc"],
            "threshold": eval_result["threshold"],
            "tn": eval_result["tn"],
            "fp": eval_result["fp"],
            "fn": eval_result["fn"],
            "tp": eval_result["tp"],
            "tpr": eval_result["tpr"],
            "tnr": eval_result["tnr"],
            "selected_visualization_indices": selected_vis_indices,
            "patch_export": patch_export_info,
        }
        with open(os.path.join(dataset_output_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(dataset_metrics, f, indent=2)

        summary_rows.append(
            {
                "dataset_name": bundle.name,
                "auroc": eval_result["auroc"],
                "tn": eval_result["tn"],
                "fp": eval_result["fp"],
                "fn": eval_result["fn"],
                "tp": eval_result["tp"],
                "tpr": eval_result["tpr"],
                "tnr": eval_result["tnr"],
            }
        )

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(args.output_dir, "shared_iad_summary.csv"),
        index=False,
    )
    log(f"Shared IAD PatchCore artifacts saved to {args.output_dir}")


if __name__ == "__main__":
    main()
