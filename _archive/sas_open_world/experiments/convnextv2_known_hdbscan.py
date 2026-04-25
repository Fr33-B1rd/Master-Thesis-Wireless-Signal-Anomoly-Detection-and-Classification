"""Train a ConvNeXtV2 encoder on known SAS classes and cluster its embeddings."""

import argparse
import json
import os

import hdbscan
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import timm
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from timm.data import resolve_model_data_config
from timm.data.transforms_factory import create_transform
from torch.utils.data import DataLoader

from sas_open_world.common import log
from sas_open_world.experiments.convnextv2_open_set import (
    SpectrogramDataset,
    apply_finetune_scope,
    build_roi_boxes,
    set_seed,
    train_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Known-class HDBSCAN tuning with ConvNeXtV2 embeddings on SAS"
    )
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--patchcore-output-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--known-classes", default="chirp,pulse,tone")
    parser.add_argument("--shot", type=int, default=0)
    parser.add_argument("--model-name", default="convnextv2_tiny")
    parser.add_argument("--input-mode", default="roi", choices=["full", "roi", "roi_letterbox"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--finetune-scope",
        default="full",
        choices=["head_only", "last_stage", "last_two_stages", "full"],
    )
    parser.add_argument("--roi-quantile", type=float, default=99.0)
    parser.add_argument("--roi-pad-ratio", type=float, default=0.10)
    parser.add_argument("--roi-min-side", type=int, default=32)
    parser.add_argument("--pca-dims", default="2,3,4,5,6,8,10,12")
    parser.add_argument("--min-cluster-sizes", default="5,8,10,12,15,20,25")
    parser.add_argument("--min-samples-list", default="1,2,3,5,8")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


@torch.no_grad()
def extract_embeddings(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    features = []
    labels = []
    sample_indices = []
    for images, batch_labels, batch_indices in loader:
        images = images.to(device, non_blocking=True)
        feat = model.forward_features(images)
        if hasattr(model, "forward_head"):
            feat = model.forward_head(feat, pre_logits=True)
        elif feat.ndim > 2:
            feat = feat.mean(dim=(2, 3))
        features.append(feat.detach().cpu().numpy().astype(np.float32, copy=False))
        labels.append(batch_labels.numpy().astype(np.int64, copy=False))
        sample_indices.append(batch_indices.numpy().astype(np.int64, copy=False))
    return (
        np.concatenate(features, axis=0),
        np.concatenate(labels, axis=0),
        np.concatenate(sample_indices, axis=0),
    )


def cluster_purity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    clusters = [cid for cid in np.unique(y_pred).tolist() if cid != -1]
    if not clusters:
        return 0.0
    total = 0
    correct = 0
    for cid in clusters:
        idx = np.where(y_pred == cid)[0]
        counts = pd.Series(y_true[idx]).value_counts()
        total += idx.size
        correct += int(counts.iloc[0])
    return float(correct / max(total, 1))


def overall_majority_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mapped = np.asarray(["unknown"] * len(y_true), dtype=object)
    for cid in [cid for cid in np.unique(y_pred).tolist() if cid != -1]:
        idx = np.where(y_pred == cid)[0]
        majority = str(pd.Series(y_true[idx]).value_counts().index[0])
        mapped[idx] = majority
    return float(np.mean(mapped == y_true))


def scan_hdbscan(
    features: np.ndarray,
    y_true: np.ndarray,
    pca_dims: list[int],
    min_cluster_sizes: list[int],
    min_samples_list: list[int],
    seed: int,
) -> pd.DataFrame:
    x_scaled = StandardScaler().fit_transform(features)
    rows = []
    for pca_dim in pca_dims:
        n_components = min(int(pca_dim), x_scaled.shape[1], x_scaled.shape[0] - 1)
        x_proj = PCA(n_components=n_components, random_state=seed).fit_transform(x_scaled)
        for min_cluster_size in min_cluster_sizes:
            for min_samples in min_samples_list:
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=int(min_cluster_size),
                    min_samples=int(min_samples),
                    metric="euclidean",
                )
                y_pred = clusterer.fit_predict(x_proj)
                rows.append(
                    {
                        "pca_dim": int(n_components),
                        "min_cluster_size": int(min_cluster_size),
                        "min_samples": int(min_samples),
                        "n_clusters": int(np.sum(np.unique(y_pred) != -1)),
                        "noise_ratio": float(np.mean(y_pred == -1)),
                        "overall_accuracy_with_noise": overall_majority_accuracy(y_true, y_pred),
                        "purity_clustered": cluster_purity(y_true, y_pred),
                        "ari": float(adjusted_rand_score(y_true, y_pred)),
                        "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
                    }
                )
    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["ari", "nmi", "overall_accuracy_with_noise", "purity_clustered", "noise_ratio"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)
    return df


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    set_seed(args.seed)
    known_classes = [part.strip() for part in args.known_classes.split(",") if part.strip()]
    class_to_id = {name: idx for idx, name in enumerate(known_classes)}

    images = np.load(os.path.join(args.dataset_dir, "test_spectrogram.npy"), mmap_mode="r")
    label_names = np.load(os.path.join(args.dataset_dir, "test_labels.npy")).astype(str)
    known_mask = np.isin(label_names, known_classes)
    known_indices = np.where(known_mask)[0].astype(np.int64)

    rng = np.random.default_rng(args.seed)
    train_indices_list = []
    cluster_indices_list = []
    for class_name in known_classes:
        class_indices = np.where(label_names == class_name)[0].astype(np.int64)
        shuffled = rng.permutation(class_indices)
        if args.shot > 0:
            if shuffled.shape[0] <= args.shot:
                raise ValueError(
                    f"{class_name} has only {shuffled.shape[0]} known samples, not enough for shot={args.shot} plus clustering"
                )
            train_indices_list.append(shuffled[: args.shot])
            cluster_indices_list.append(shuffled[args.shot :])
        else:
            train_indices_list.append(shuffled)
            cluster_indices_list.append(shuffled)

    train_indices = np.concatenate(train_indices_list).astype(np.int64)
    cluster_indices = np.concatenate(cluster_indices_list).astype(np.int64)
    train_label_ids = np.asarray([class_to_id[str(label_names[idx])] for idx in train_indices], dtype=np.int64)
    cluster_label_ids = np.asarray([class_to_id[str(label_names[idx])] for idx in cluster_indices], dtype=np.int64)
    cluster_label_names = np.asarray([str(label_names[idx]) for idx in cluster_indices], dtype="U16")

    anomaly_outputs = np.load(os.path.join(args.patchcore_output_dir, "anomaly_outputs.npz"))
    anomaly_maps = anomaly_outputs["anomaly_maps"]
    roi_boxes = None
    if args.input_mode in {"roi", "roi_letterbox"}:
        roi_boxes = build_roi_boxes(
            anomaly_maps=anomaly_maps,
            sample_indices=np.unique(np.concatenate([train_indices, cluster_indices])).astype(np.int64),
            quantile=args.roi_quantile,
            pad_ratio=args.roi_pad_ratio,
            min_side=args.roi_min_side,
        )

    model = timm.create_model(args.model_name, pretrained=True, num_classes=len(known_classes))
    trainable_params = apply_finetune_scope(model, args.finetune_scope)
    data_config = resolve_model_data_config(model)
    transform = create_transform(**data_config, is_training=False)

    dataset = SpectrogramDataset(
        images=images,
        labels=train_label_ids,
        indices=train_indices,
        transform=transform,
        roi_boxes=roi_boxes,
        letterbox_roi=(args.input_mode == "roi_letterbox"),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    log(
        f"Training {args.model_name} on {device.type} with {train_indices.shape[0]} known samples "
        f"(input_mode={args.input_mode}, finetune_scope={args.finetune_scope}, trainable_params={trainable_params})"
    )
    history = train_model(
        model=model,
        train_loader=loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    with open(os.path.join(args.output_dir, "train_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    embed_loader = DataLoader(
        SpectrogramDataset(
            images=images,
            labels=cluster_label_ids,
            indices=cluster_indices,
            transform=transform,
            roi_boxes=roi_boxes,
            letterbox_roi=(args.input_mode == "roi_letterbox"),
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    embeddings, _, sample_indices = extract_embeddings(model=model, loader=embed_loader, device=device)
    np.savez_compressed(
        os.path.join(args.output_dir, "known_embeddings.npz"),
        X=embeddings.astype(np.float32),
        label_names=cluster_label_names.astype("U16"),
        sample_indices=sample_indices.astype(np.int64),
    )

    pca_dims = [int(part.strip()) for part in args.pca_dims.split(",") if part.strip()]
    min_cluster_sizes = [int(part.strip()) for part in args.min_cluster_sizes.split(",") if part.strip()]
    min_samples_list = [int(part.strip()) for part in args.min_samples_list.split(",") if part.strip()]
    scan_df = scan_hdbscan(
        features=embeddings,
        y_true=cluster_label_names,
        pca_dims=pca_dims,
        min_cluster_sizes=min_cluster_sizes,
        min_samples_list=min_samples_list,
        seed=args.seed,
    )
    scan_df.to_csv(os.path.join(args.output_dir, "scan_results.csv"), index=False)
    scan_df.head(20).to_csv(os.path.join(args.output_dir, "top20_configs.csv"), index=False)

    best = scan_df.iloc[0].to_dict()
    with open(os.path.join(args.output_dir, "best_config.json"), "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    x_scaled = StandardScaler().fit_transform(embeddings)
    x_proj = PCA(
        n_components=min(int(best["pca_dim"]), x_scaled.shape[1], x_scaled.shape[0] - 1),
        random_state=args.seed,
    ).fit_transform(x_scaled)
    y_pred = hdbscan.HDBSCAN(
        min_cluster_size=int(best["min_cluster_size"]),
        min_samples=int(best["min_samples"]),
        metric="euclidean",
    ).fit_predict(x_proj)

    assignments = pd.DataFrame(
        {
            "sample_index": sample_indices,
            "true_label": cluster_label_names,
            "cluster": y_pred,
        }
    )
    assignments.to_csv(os.path.join(args.output_dir, "best_cluster_assignments.csv"), index=False)

    summary_rows = []
    for cid in sorted(np.unique(y_pred).tolist()):
        mask = y_pred == cid
        labels_c = cluster_label_names[mask]
        counts = pd.Series(labels_c).value_counts()
        summary_rows.append(
            {
                "cluster": int(cid),
                "size": int(mask.sum()),
                "majority_label": "noise" if cid == -1 else str(counts.index[0]),
                "purity": 0.0 if cid == -1 else float(counts.iloc[0] / max(mask.sum(), 1)),
                "chirp": int(np.sum(labels_c == "chirp")),
                "pulse": int(np.sum(labels_c == "pulse")),
                "tone": int(np.sum(labels_c == "tone")),
            }
        )
    cluster_summary = pd.DataFrame(summary_rows)
    cluster_summary.to_csv(os.path.join(args.output_dir, "best_cluster_summary.csv"), index=False)

    perplexity = min(30, max(5, (len(cluster_label_names) - 1) // 3))
    x_tsne = TSNE(
        n_components=2,
        random_state=args.seed,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
    ).fit_transform(x_proj)
    tsne_df = pd.DataFrame(
        {
            "tsne_x": x_tsne[:, 0],
            "tsne_y": x_tsne[:, 1],
            "true_label": cluster_label_names,
            "cluster": y_pred,
        }
    )
    tsne_df.to_csv(os.path.join(args.output_dir, "best_tsne_projection.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    for class_name in known_classes:
        sub = tsne_df[tsne_df["true_label"] == class_name]
        axes[0].scatter(sub["tsne_x"], sub["tsne_y"], s=10, alpha=0.7, label=class_name)
    axes[0].set_title("t-SNE by True Label")
    axes[0].legend()

    clusters = sorted(tsne_df["cluster"].unique().tolist())
    palette = sns.color_palette("tab10", n_colors=max(1, len(clusters)))
    color_map = {cid: palette[i % len(palette)] for i, cid in enumerate(clusters)}
    for cid in clusters:
        sub = tsne_df[tsne_df["cluster"] == cid]
        axes[1].scatter(
            sub["tsne_x"],
            sub["tsne_y"],
            s=10,
            alpha=0.7,
            label=("noise" if cid == -1 else f"cluster {cid}"),
            color=color_map[cid],
        )
    axes[1].set_title("t-SNE by HDBSCAN Cluster")
    axes[1].legend(fontsize=8)
    fig.suptitle("Known-Class HDBSCAN on ROI-only ConvNeXt Embeddings")
    fig.savefig(os.path.join(args.output_dir, "best_tsne_projection.png"), dpi=220)
    plt.close(fig)

    comp = assignments.groupby(["cluster", "true_label"]).size().unstack(fill_value=0)
    plt.figure(figsize=(8, 5))
    sns.heatmap(comp, annot=True, fmt="d", cmap="Blues")
    plt.title("Best Config Cluster Composition")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "best_cluster_composition_heatmap.png"), dpi=220)
    plt.close()

    summary = {
        "dataset_dir": args.dataset_dir,
        "patchcore_output_dir": args.patchcore_output_dir,
        "known_classes": known_classes,
        "shot": int(args.shot),
        "n_train_samples": int(train_indices.shape[0]),
        "n_cluster_samples": int(cluster_indices.shape[0]),
        "input_mode": args.input_mode,
        "model_name": args.model_name,
        "best_config": best,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log(f"Finished known-class ConvNeXt HDBSCAN tuning. Best ARI={best['ari']:.4f}")


if __name__ == "__main__":
    main()
