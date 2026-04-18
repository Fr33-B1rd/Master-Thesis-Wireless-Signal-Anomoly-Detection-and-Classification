"""ConvNeXtV2 open-set experiment on SAS anomaly classes only."""

import argparse
import copy
import json
import math
import os
import random
from collections import deque
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import f1_score
from timm.data import resolve_model_data_config
from timm.data.transforms_factory import create_transform
from torch import nn
from torch.utils.data import DataLoader, Dataset

from sas_open_world.common import log, save_rectangular_confusion


@dataclass(frozen=True)
class RoiBox:
    top: int
    bottom: int
    left: int
    right: int


class WeakSpectrogramAugment:
    """Applies small time/frequency masks and shifts without altering semantics."""

    def __init__(
        self,
        time_mask_max: int = 24,
        freq_mask_max: int = 24,
        shift_max: int = 8,
        gain_jitter: float = 0.05,
        seed: int = 42,
    ) -> None:
        self.time_mask_max = int(time_mask_max)
        self.freq_mask_max = int(freq_mask_max)
        self.shift_max = int(shift_max)
        self.gain_jitter = float(gain_jitter)
        self.rng = np.random.default_rng(seed)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        x = tensor.clone()
        _, height, width = x.shape

        if self.shift_max > 0:
            shift = int(self.rng.integers(-self.shift_max, self.shift_max + 1))
            if shift != 0:
                x = torch.roll(x, shifts=shift, dims=2)

        if self.time_mask_max > 0:
            mask_width = int(self.rng.integers(0, self.time_mask_max + 1))
            if 0 < mask_width < width:
                start = int(self.rng.integers(0, width - mask_width + 1))
                x[:, :, start : start + mask_width] = 0.0

        if self.freq_mask_max > 0:
            mask_height = int(self.rng.integers(0, self.freq_mask_max + 1))
            if 0 < mask_height < height:
                start = int(self.rng.integers(0, height - mask_height + 1))
                x[:, start : start + mask_height, :] = 0.0

        if self.gain_jitter > 0:
            gain = float(self.rng.uniform(1.0 - self.gain_jitter, 1.0 + self.gain_jitter))
            x = x * gain

        return x


class SpectrogramDataset(Dataset):
    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
        transform,
        augment=None,
        roi_boxes: dict[int, RoiBox] | None = None,
        return_full_and_roi: bool = False,
        attention_maps: np.ndarray | None = None,
        attention_floor: float = 0.15,
        attention_power: float = 2.0,
        component_mask_maps: np.ndarray | None = None,
        mask_quantile: float = 99.0,
        mask_background_floor: float = 0.05,
        letterbox_roi: bool = False,
    ) -> None:
        self.images = images
        self.labels = labels
        self.indices = indices
        self.transform = transform
        self.augment = augment
        self.roi_boxes = roi_boxes or {}
        self.return_full_and_roi = return_full_and_roi
        self.attention_maps = attention_maps
        self.attention_floor = float(attention_floor)
        self.attention_power = float(attention_power)
        self.component_mask_maps = component_mask_maps
        self.mask_quantile = float(mask_quantile)
        self.mask_background_floor = float(mask_background_floor)
        self.letterbox_roi = bool(letterbox_roi)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int):
        sample_index = int(self.indices[idx])
        image = self.images[sample_index]
        roi_box = self.roi_boxes.get(sample_index)
        if self.attention_maps is not None:
            image = apply_soft_attention(
                image=image,
                anomaly_map=self.attention_maps[sample_index],
                floor=self.attention_floor,
                power=self.attention_power,
            )
        if self.component_mask_maps is not None:
            image = apply_component_mask(
                image=image,
                anomaly_map=self.component_mask_maps[sample_index],
                quantile=self.mask_quantile,
                background_floor=self.mask_background_floor,
            )
        if self.return_full_and_roi:
            full_pil = Image.fromarray(image).convert("RGB")
            roi_image = image
            if roi_box is not None:
                roi_image = image[roi_box.top : roi_box.bottom, roi_box.left : roi_box.right]
            roi_pil = Image.fromarray(roi_image).convert("RGB")
            full_tensor = self.transform(full_pil)
            roi_tensor = self.transform(roi_pil)
            if self.augment is not None:
                full_tensor = self.augment(full_tensor)
                roi_tensor = self.augment(roi_tensor)
            tensor = (full_tensor, roi_tensor)
        else:
            if roi_box is not None:
                image = image[roi_box.top : roi_box.bottom, roi_box.left : roi_box.right]
                if self.letterbox_roi:
                    image = letterbox_square(image)
            pil = Image.fromarray(image).convert("RGB")
            tensor = self.transform(pil)
            if self.augment is not None:
                tensor = self.augment(tensor)
        label = int(self.labels[idx])
        return tensor, label, sample_index


class SharedBackboneDualClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(feature_dim * 2, num_classes)

    def forward(self, inputs):
        full_images, roi_images = inputs
        full_features = self.backbone(full_images)
        roi_features = self.backbone(roi_images)
        fused = torch.cat([full_features, roi_features], dim=1)
        return self.head(fused)


class CenterLoss(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int) -> None:
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, embedding_dim, dtype=torch.float32))

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        centers_batch = self.centers.index_select(0, labels)
        diff = embeddings - centers_batch
        return 0.5 * diff.pow(2).sum(dim=1).mean()


class ArcFaceHead(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        scale: float,
        margin: float,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim, dtype=torch.float32))
        nn.init.xavier_uniform_(self.weight)
        self.scale = float(scale)
        self.margin = float(margin)
        self.cos_margin = math.cos(self.margin)
        self.sin_margin = math.sin(self.margin)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        normalized_embeddings = F.normalize(embeddings, dim=1)
        normalized_weight = F.normalize(self.weight, dim=1)
        cosine = torch.matmul(normalized_embeddings, normalized_weight.T).clamp(-1.0, 1.0)
        if labels is None:
            return cosine * self.scale

        sine = torch.sqrt(torch.clamp(1.0 - cosine.pow(2), min=1e-7))
        phi = cosine * self.cos_margin - sine * self.sin_margin
        one_hot = F.one_hot(labels, num_classes=normalized_weight.size(0)).to(dtype=cosine.dtype)
        logits = one_hot * phi + (1.0 - one_hot) * cosine
        return logits * self.scale


def find_largest_connected_component(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    best_coords: list[tuple[int, int]] = []
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for start_y, start_x in np.argwhere(mask):
        if visited[start_y, start_x]:
            continue
        queue: deque[tuple[int, int]] = deque([(int(start_y), int(start_x))])
        visited[start_y, start_x] = True
        coords: list[tuple[int, int]] = []
        while queue:
            y, x = queue.popleft()
            coords.append((y, x))
            for dy, dx in neighbors:
                ny = y + dy
                nx = x + dx
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    queue.append((ny, nx))
        if len(coords) > len(best_coords):
            best_coords = coords

    component = np.zeros_like(mask, dtype=bool)
    for y, x in best_coords:
        component[y, x] = True
    return component


def apply_soft_attention(
    image: np.ndarray,
    anomaly_map: np.ndarray,
    floor: float,
    power: float,
) -> np.ndarray:
    image_float = image.astype(np.float32, copy=False)
    attention = anomaly_map.astype(np.float32, copy=False)
    attention = attention - float(attention.min())
    max_value = float(attention.max())
    if max_value > 1e-6:
        attention = attention / max_value
    else:
        attention = np.ones_like(attention, dtype=np.float32)
    attention = np.power(np.clip(attention, 0.0, 1.0), power)
    attention = float(floor) + (1.0 - float(floor)) * attention

    weighted = image_float * attention
    weighted_max = float(weighted.max())
    image_max = float(image_float.max())
    if weighted_max > 1e-6 and image_max > 1e-6:
        weighted = weighted * (image_max / weighted_max)
    return np.clip(weighted, 0.0, 255.0).astype(np.uint8)


def apply_component_mask(
    image: np.ndarray,
    anomaly_map: np.ndarray,
    quantile: float,
    background_floor: float,
) -> np.ndarray:
    threshold = float(np.percentile(anomaly_map.astype(np.float32, copy=False), quantile))
    binary_mask = anomaly_map >= threshold
    if not np.any(binary_mask):
        return image.astype(np.uint8, copy=False)
    component_mask = find_largest_connected_component(binary_mask)
    if not np.any(component_mask):
        return image.astype(np.uint8, copy=False)

    image_float = image.astype(np.float32, copy=False)
    mask_float = component_mask.astype(np.float32)
    weighted = image_float * (float(background_floor) + (1.0 - float(background_floor)) * mask_float)
    return np.clip(weighted, 0.0, 255.0).astype(np.uint8)


def letterbox_square(image: np.ndarray) -> np.ndarray:
    height, width = image.shape
    side = max(int(height), int(width))
    canvas = np.zeros((side, side), dtype=image.dtype)
    top = (side - height) // 2
    left = (side - width) // 2
    canvas[top : top + height, left : left + width] = image
    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ConvNeXtV2 open-set classification on SAS anomaly classes"
    )
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--known-classes", default="chirp,pulse,tone")
    parser.add_argument("--unknown-classes", default="comb,fsk,ofdm")
    parser.add_argument("--shot", type=int, default=60)
    parser.add_argument("--model-name", default="convnextv2_tiny")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--finetune-scope",
        default="full",
        choices=["head_only", "last_stage", "last_two_stages", "full"],
    )
    parser.add_argument(
        "--reject-mode",
        default="max_softmax",
        choices=["max_softmax", "energy", "energy_prototype"],
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--train-loss", default="ce", choices=["ce", "ce_supcon", "ce_center", "ce_arcface"])
    parser.add_argument("--supcon-weight", type=float, default=0.2)
    parser.add_argument("--supcon-temperature", type=float, default=0.07)
    parser.add_argument("--center-loss-weight", type=float, default=0.01)
    parser.add_argument("--center-loss-lr", type=float, default=0.5)
    parser.add_argument("--arcface-scale", type=float, default=30.0)
    parser.add_argument("--arcface-margin", type=float, default=0.3)
    parser.add_argument("--distance-quantile", type=float, default=95.0)
    parser.add_argument("--distance-metric", default="cosine", choices=["cosine", "l2"])
    parser.add_argument("--train-augment", default="none", choices=["none", "weak_spec"])
    parser.add_argument(
        "--input-mode",
        default="full",
        choices=["full", "roi", "roi_letterbox", "dual", "soft_attention", "component_mask"],
    )
    parser.add_argument("--patchcore-output-dir", default="")
    parser.add_argument("--roi-quantile", type=float, default=99.0)
    parser.add_argument("--roi-pad-ratio", type=float, default=0.10)
    parser.add_argument("--roi-min-side", type=int, default=32)
    parser.add_argument("--attention-floor", type=float, default=0.15)
    parser.add_argument("--attention-power", type=float, default=2.0)
    parser.add_argument("--mask-quantile", type=float, default=99.0)
    parser.add_argument("--mask-background-floor", type=float, default=0.05)
    parser.add_argument("--coverage-constraint", type=float, default=0.95)
    parser.add_argument("--thresholds", default="0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95")
    parser.add_argument("--threshold-quantile", type=float, default=95.0)
    parser.add_argument("--calibration-per-class", type=int, default=0)
    parser.add_argument(
        "--threshold-source",
        default="auto",
        choices=["auto", "train", "external_calibration"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stopping", action="store_true")
    parser.add_argument("--val-per-class", type=int, default=10)
    parser.add_argument("--early-stopping-use-calibration", action="store_true")
    parser.add_argument(
        "--early-stopping-monitor",
        default="val_loss",
        choices=["val_loss", "known_coverage"],
    )
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--label-flip-rate", type=float, default=0.0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_model(
    model: nn.Module,
    center_loss_module: CenterLoss | None,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    temperature: float,
    train_loss: str,
    supcon_weight: float,
    supcon_temperature: float,
    center_loss_weight: float,
    center_loss_lr: float,
    early_stopping: bool,
    early_stopping_monitor: str,
    early_stopping_patience: int,
    early_stopping_min_delta: float,
    threshold_quantile: float,
) -> list[dict]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    center_optimizer = None
    if center_loss_module is not None:
        center_optimizer = torch.optim.SGD(center_loss_module.parameters(), lr=center_loss_lr)
    history = []
    best_state = None
    best_val_loss = float("inf")
    best_monitor_value = None
    best_monitor_tiebreak = None
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_ce_loss = 0.0
        running_supcon_loss = 0.0
        running_center_loss = 0.0
        running_correct = 0
        running_total = 0
        for images, labels, _ in train_loader:
            if isinstance(images, (list, tuple)):
                images = tuple(item.to(device, non_blocking=True) for item in images)
            else:
                images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if center_optimizer is not None:
                center_optimizer.zero_grad(set_to_none=True)
            logits, embeddings = forward_logits_and_embeddings(model, images, labels)
            ce_loss = F.cross_entropy(logits, labels)
            if train_loss == "ce_supcon":
                supcon_loss = supervised_contrastive_loss(
                    embeddings=embeddings,
                    labels=labels,
                    temperature=supcon_temperature,
                )
                center_loss = torch.zeros((), device=device)
                loss = ce_loss + float(supcon_weight) * supcon_loss
            elif train_loss == "ce_center":
                if center_loss_module is None:
                    raise ValueError("Center loss training requested without a center loss module")
                supcon_loss = torch.zeros((), device=device)
                center_loss = center_loss_module(embeddings, labels)
                loss = ce_loss + float(center_loss_weight) * center_loss
            else:
                supcon_loss = torch.zeros((), device=device)
                center_loss = torch.zeros((), device=device)
                loss = ce_loss
            loss.backward()
            optimizer.step()
            if center_optimizer is not None:
                center_optimizer.step()

            running_loss += float(loss.item()) * int(labels.size(0))
            running_ce_loss += float(ce_loss.item()) * int(labels.size(0))
            running_supcon_loss += float(supcon_loss.item()) * int(labels.size(0))
            running_center_loss += float(center_loss.item()) * int(labels.size(0))
            preds = logits.argmax(dim=1)
            running_correct += int((preds == labels).sum().item())
            running_total += int(labels.size(0))

        epoch_record = {
            "epoch": epoch + 1,
            "loss": running_loss / max(running_total, 1),
            "ce_loss": running_ce_loss / max(running_total, 1),
            "supcon_loss": running_supcon_loss / max(running_total, 1),
            "center_loss": running_center_loss / max(running_total, 1),
            "train_accuracy": running_correct / max(running_total, 1),
        }

        if val_loader is not None:
            model.eval()
            val_loss_sum = 0.0
            val_ce_loss_sum = 0.0
            val_supcon_loss_sum = 0.0
            val_center_loss_sum = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for images, labels, _ in val_loader:
                    if isinstance(images, (list, tuple)):
                        images = tuple(item.to(device, non_blocking=True) for item in images)
                    else:
                        images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    logits, embeddings = forward_logits_and_embeddings(model, images, labels)
                    ce_loss = F.cross_entropy(logits, labels)
                    if train_loss == "ce_supcon":
                        supcon_loss = supervised_contrastive_loss(
                            embeddings=embeddings,
                            labels=labels,
                            temperature=supcon_temperature,
                        )
                        center_loss = torch.zeros((), device=device)
                        loss = ce_loss + float(supcon_weight) * supcon_loss
                    elif train_loss == "ce_center":
                        if center_loss_module is None:
                            raise ValueError("Center loss training requested without a center loss module")
                        supcon_loss = torch.zeros((), device=device)
                        center_loss = center_loss_module(embeddings, labels)
                        loss = ce_loss + float(center_loss_weight) * center_loss
                    else:
                        supcon_loss = torch.zeros((), device=device)
                        center_loss = torch.zeros((), device=device)
                        loss = ce_loss

                    batch_size = int(labels.size(0))
                    val_loss_sum += float(loss.item()) * batch_size
                    val_ce_loss_sum += float(ce_loss.item()) * batch_size
                    val_supcon_loss_sum += float(supcon_loss.item()) * batch_size
                    val_center_loss_sum += float(center_loss.item()) * batch_size
                    preds = logits.argmax(dim=1)
                    val_correct += int((preds == labels).sum().item())
                    val_total += batch_size

            epoch_record["val_loss"] = val_loss_sum / max(val_total, 1)
            epoch_record["val_ce_loss"] = val_ce_loss_sum / max(val_total, 1)
            epoch_record["val_supcon_loss"] = val_supcon_loss_sum / max(val_total, 1)
            epoch_record["val_center_loss"] = val_center_loss_sum / max(val_total, 1)
            epoch_record["val_accuracy"] = val_correct / max(val_total, 1)

            if early_stopping_monitor == "known_coverage":
                train_eval_df = infer(
                    model=model,
                    loader=train_loader,
                    device=device,
                    temperature=temperature,
                )
                val_eval_df = infer(
                    model=model,
                    loader=val_loader,
                    device=device,
                    temperature=temperature,
                )
                train_threshold = float(
                    np.percentile(train_eval_df["energy"].to_numpy(), threshold_quantile)
                )
                val_pred_known = (val_eval_df["energy"].to_numpy() <= train_threshold).astype(np.float32)
                epoch_record["val_known_coverage"] = float(np.mean(val_pred_known)) if val_pred_known.size > 0 else 0.0
                monitor_value = float(epoch_record["val_known_coverage"])
                tiebreak_value = float(epoch_record["val_accuracy"])
                if (
                    best_monitor_value is None
                    or monitor_value > (best_monitor_value + float(early_stopping_min_delta))
                    or (
                        abs(monitor_value - best_monitor_value) <= float(early_stopping_min_delta)
                        and (
                            best_monitor_tiebreak is None
                            or tiebreak_value > best_monitor_tiebreak + 1e-12
                        )
                    )
                ):
                    best_monitor_value = monitor_value
                    best_monitor_tiebreak = tiebreak_value
                    best_val_loss = float(epoch_record["val_loss"])
                    best_epoch = epoch + 1
                    stale_epochs = 0
                    best_state = copy.deepcopy(model.state_dict())
                else:
                    stale_epochs += 1
            else:
                if epoch_record["val_loss"] < (best_val_loss - float(early_stopping_min_delta)):
                    best_val_loss = float(epoch_record["val_loss"])
                    best_monitor_value = float(epoch_record["val_loss"])
                    best_monitor_tiebreak = float(epoch_record["val_accuracy"])
                    best_epoch = epoch + 1
                    stale_epochs = 0
                    best_state = copy.deepcopy(model.state_dict())
                else:
                    stale_epochs += 1

        history.append(epoch_record)
        log(
            f"epoch {epoch + 1}/{epochs}: "
            f"loss={history[-1]['loss']:.4f}, "
            f"ce={history[-1]['ce_loss']:.4f}, "
            f"supcon={history[-1]['supcon_loss']:.4f}, "
            f"center={history[-1]['center_loss']:.4f}, "
            f"train_acc={history[-1]['train_accuracy']:.4f}"
        )
        if val_loader is not None:
            extra = ""
            if "val_known_coverage" in history[-1]:
                extra = f", val_known_coverage={history[-1]['val_known_coverage']:.4f}"
            log(
                f"  val_loss={history[-1]['val_loss']:.4f}, "
                f"val_acc={history[-1]['val_accuracy']:.4f}{extra}, "
                f"best_epoch={best_epoch}"
            )
            if early_stopping and stale_epochs >= int(early_stopping_patience):
                log(f"early stopping at epoch {epoch + 1}, restoring best epoch {best_epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    if history:
        history[-1]["best_epoch"] = int(best_epoch) if best_epoch > 0 else int(history[-1]["epoch"])
        history[-1]["best_val_loss"] = float(best_val_loss) if best_epoch > 0 else None
        history[-1]["early_stopping_monitor"] = early_stopping_monitor
        history[-1]["best_monitor_value"] = best_monitor_value
    return history


def apply_finetune_scope(module: nn.Module, scope: str) -> int:
    for param in module.parameters():
        param.requires_grad = False

    if scope == "head_only":
        train_prefixes = ["head", "head_drop", "norm_pre"]
    elif scope == "last_stage":
        train_prefixes = ["stages.3", "head", "head_drop", "norm_pre"]
    elif scope == "last_two_stages":
        train_prefixes = ["stages.2", "stages.3", "head", "head_drop", "norm_pre"]
    elif scope == "full":
        train_prefixes = [""]
    else:
        raise ValueError(f"Unsupported finetune scope: {scope}")

    trainable = 0
    for name, param in module.named_parameters():
        if any(name.startswith(prefix) for prefix in train_prefixes):
            param.requires_grad = True
            trainable += param.numel()
    return int(trainable)


def forward_logits_and_embeddings(model: nn.Module, images, labels: torch.Tensor | None = None):
    if isinstance(images, (list, tuple)):
        if not hasattr(model, "backbone"):
            raise ValueError("Tuple inputs require a model with a shared backbone")
        full_images, roi_images = images
        full_features = model.backbone(full_images)
        roi_features = model.backbone(roi_images)
        embeddings = torch.cat([full_features, roi_features], dim=1)
        if hasattr(model, "arcface_head") and model.arcface_head is not None:
            logits = model.arcface_head(embeddings, labels)
        else:
            logits = model.head(embeddings)
        return logits, embeddings
    features = model.forward_features(images)
    embeddings = model.forward_head(features, pre_logits=True)
    if hasattr(model, "arcface_head") and model.arcface_head is not None:
        logits = model.arcface_head(embeddings, labels)
    else:
        logits = model.forward_head(features)
    return logits, embeddings


def get_embedding_dim(model: nn.Module, input_mode: str) -> int:
    if input_mode == "dual":
        return int(model.head.in_features)
    if hasattr(model, "num_features"):
        return int(model.num_features)
    if hasattr(model, "head") and hasattr(model.head, "in_features"):
        return int(model.head.in_features)
    raise ValueError("Unable to infer embedding dimension for center loss")


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    normalized = F.normalize(embeddings, dim=1)
    logits = torch.matmul(normalized, normalized.T) / max(float(temperature), 1e-6)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    labels = labels.view(-1, 1)
    positive_mask = torch.eq(labels, labels.T).to(dtype=normalized.dtype)
    self_mask = torch.eye(labels.size(0), device=labels.device, dtype=normalized.dtype)
    positive_mask = positive_mask - self_mask

    exp_logits = torch.exp(logits) * (1.0 - self_mask)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    positive_count = positive_mask.sum(dim=1)
    valid_mask = positive_count > 0
    if not bool(valid_mask.any()):
        return torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)
    mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / positive_count.clamp_min(1.0)
    loss = -mean_log_prob_pos[valid_mask].mean()
    return loss


def build_roi_boxes(
    anomaly_maps: np.ndarray,
    sample_indices: np.ndarray,
    quantile: float,
    pad_ratio: float,
    min_side: int,
) -> dict[int, RoiBox]:
    roi_boxes: dict[int, RoiBox] = {}
    for sample_index in sample_indices.tolist():
        anomaly_map = anomaly_maps[int(sample_index)].astype(np.float32, copy=False)
        threshold = float(np.percentile(anomaly_map, quantile))
        mask = anomaly_map >= threshold
        ys, xs = np.nonzero(mask)
        height, width = anomaly_map.shape
        if ys.size == 0 or xs.size == 0:
            roi_boxes[int(sample_index)] = RoiBox(0, height, 0, width)
            continue

        top = int(ys.min())
        bottom = int(ys.max()) + 1
        left = int(xs.min())
        right = int(xs.max()) + 1

        box_height = bottom - top
        box_width = right - left

        pad_h = max(1, int(round(box_height * pad_ratio)))
        pad_w = max(1, int(round(box_width * pad_ratio)))
        top = max(0, top - pad_h)
        bottom = min(height, bottom + pad_h)
        left = max(0, left - pad_w)
        right = min(width, right + pad_w)

        box_height = bottom - top
        box_width = right - left
        if box_height < min_side:
            deficit = min_side - box_height
            expand_top = deficit // 2
            expand_bottom = deficit - expand_top
            top = max(0, top - expand_top)
            bottom = min(height, bottom + expand_bottom)
            if bottom - top < min_side:
                if top == 0:
                    bottom = min(height, min_side)
                else:
                    top = max(0, height - min_side)
        if box_width < min_side:
            deficit = min_side - box_width
            expand_left = deficit // 2
            expand_right = deficit - expand_left
            left = max(0, left - expand_left)
            right = min(width, right + expand_right)
            if right - left < min_side:
                if left == 0:
                    right = min(width, min_side)
                else:
                    left = max(0, width - min_side)

        roi_boxes[int(sample_index)] = RoiBox(top, bottom, left, right)
    return roi_boxes


@torch.no_grad()
def infer(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    temperature: float,
) -> pd.DataFrame:
    model.eval()
    rows = []
    for images, labels, sample_indices in loader:
        if isinstance(images, (list, tuple)):
            images = tuple(item.to(device, non_blocking=True) for item in images)
        else:
            images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        max_probs, pred_ids = probs.max(dim=1)
        energy = (-temperature * torch.logsumexp(logits / temperature, dim=1)).detach()
        batch_size = int(labels.size(0))
        for i in range(batch_size):
            rows.append(
                {
                    "sample_index": int(sample_indices[i].item()),
                    "true_known_id": int(labels[i].item()),
                    "pred_known_id": int(pred_ids[i].item()),
                    "max_softmax": float(max_probs[i].item()),
                    "energy": float(energy[i].item()),
                }
            )
    return pd.DataFrame(rows)


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms


def _compute_class_distances(
    embeddings: np.ndarray,
    prototype: np.ndarray,
    metric: str,
) -> np.ndarray:
    if metric == "cosine":
        norm_embeddings = _l2_normalize(embeddings)
        norm_prototype = prototype / max(float(np.linalg.norm(prototype)), 1e-12)
        return 1.0 - np.matmul(norm_embeddings, norm_prototype.astype(np.float32))
    if metric == "l2":
        return np.linalg.norm(embeddings - prototype[None, :], axis=1)
    raise ValueError(f"Unsupported distance metric: {metric}")


@torch.no_grad()
def infer_with_embeddings(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    temperature: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    model.eval()
    rows = []
    embedding_batches = []
    for images, labels, sample_indices in loader:
        if isinstance(images, (list, tuple)):
            images = tuple(item.to(device, non_blocking=True) for item in images)
        else:
            images = images.to(device, non_blocking=True)
        logits, embeddings = forward_logits_and_embeddings(model, images)
        probs = torch.softmax(logits, dim=1)
        max_probs, pred_ids = probs.max(dim=1)
        energy = (-temperature * torch.logsumexp(logits / temperature, dim=1)).detach()
        embedding_batches.append(embeddings.detach().cpu().numpy().astype(np.float32))
        batch_size = int(labels.size(0))
        for i in range(batch_size):
            rows.append(
                {
                    "sample_index": int(sample_indices[i].item()),
                    "true_known_id": int(labels[i].item()),
                    "pred_known_id": int(pred_ids[i].item()),
                    "max_softmax": float(max_probs[i].item()),
                    "energy": float(energy[i].item()),
                }
            )
    embedding_matrix = np.concatenate(embedding_batches, axis=0).astype(np.float32)
    return pd.DataFrame(rows), embedding_matrix


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    set_seed(args.seed)

    known_classes = [part.strip() for part in args.known_classes.split(",") if part.strip()]
    unknown_classes = [part.strip() for part in args.unknown_classes.split(",") if part.strip()]
    all_anomaly_classes = [*known_classes, *unknown_classes]

    images = np.load(os.path.join(args.dataset_dir, "test_spectrogram.npy"), mmap_mode="r")
    label_names = np.load(os.path.join(args.dataset_dir, "test_labels.npy")).astype(str)
    anomaly_mask = label_names != "normal"
    anomaly_indices = np.where(anomaly_mask)[0]

    known_to_id = {name: idx for idx, name in enumerate(known_classes)}
    train_indices = []
    train_split_indices = []
    val_split_indices = []
    calibration_indices = []
    rng = np.random.default_rng(args.seed)
    for class_name in known_classes:
        class_indices = np.where(label_names == class_name)[0]
        shuffled = rng.permutation(class_indices)
        if shuffled.shape[0] < args.shot:
            raise ValueError(f"{class_name} has only {shuffled.shape[0]} samples, fewer than shot={args.shot}")
        sampled = shuffled[: args.shot]
        train_indices.extend(sampled.tolist())
        if args.early_stopping:
            if args.early_stopping_use_calibration:
                train_split_indices.extend(sampled.tolist())
            else:
                if args.val_per_class <= 0:
                    raise ValueError("--val-per-class must be positive when --early-stopping is enabled")
                if args.shot <= args.val_per_class:
                    raise ValueError(
                        f"shot={args.shot} must be greater than val_per_class={args.val_per_class}"
                    )
                val_split_indices.extend(sampled[: args.val_per_class].tolist())
                train_split_indices.extend(sampled[args.val_per_class :].tolist())
        else:
            train_split_indices.extend(sampled.tolist())
        if args.calibration_per_class > 0:
            remaining = shuffled[args.shot :]
            if remaining.shape[0] < args.calibration_per_class:
                raise ValueError(
                    f"{class_name} has only {remaining.shape[0]} held-out known samples, "
                    f"fewer than calibration_per_class={args.calibration_per_class}"
                )
            calibration_indices.extend(remaining[: args.calibration_per_class].tolist())
    train_indices = np.asarray(sorted(train_indices), dtype=np.int64)
    train_split_indices = np.asarray(sorted(train_split_indices), dtype=np.int64)
    val_split_indices = np.asarray(sorted(val_split_indices), dtype=np.int64)
    calibration_indices = np.asarray(sorted(calibration_indices), dtype=np.int64)
    train_set = set(int(idx) for idx in train_indices.tolist())
    calibration_set = set(int(idx) for idx in calibration_indices.tolist())
    eval_indices = np.asarray(
        [idx for idx in anomaly_indices.tolist() if int(idx) not in train_set and int(idx) not in calibration_set],
        dtype=np.int64,
    )

    train_label_ids = np.asarray([known_to_id[str(label_names[idx])] for idx in train_split_indices], dtype=np.int64)
    if args.label_flip_rate > 0:
        num_classes = len(known_classes)
        flip_rng = np.random.default_rng(args.seed + 9999)
        flip_mask = flip_rng.random(train_label_ids.shape[0]) < args.label_flip_rate
        n_flipped = int(flip_mask.sum())
        offsets = flip_rng.integers(1, num_classes, size=n_flipped)
        train_label_ids[flip_mask] = (train_label_ids[flip_mask] + offsets) % num_classes
        log(f"Label flip: flipped {n_flipped}/{train_label_ids.shape[0]} labels (rate={args.label_flip_rate})")
    val_label_ids = np.asarray([known_to_id[str(label_names[idx])] for idx in val_split_indices], dtype=np.int64)
    calibration_label_ids = np.asarray(
        [known_to_id[str(label_names[idx])] for idx in calibration_indices],
        dtype=np.int64,
    )
    eval_true_names = np.asarray([str(label_names[idx]) for idx in eval_indices], dtype="U16")
    eval_label_ids = np.asarray([known_to_id.get(name, -1) for name in eval_true_names], dtype=np.int64)

    roi_boxes = None
    attention_maps = None
    component_mask_maps = None
    if args.input_mode in {"roi", "roi_letterbox", "dual", "soft_attention", "component_mask"}:
        if not args.patchcore_output_dir:
            raise ValueError("--patchcore-output-dir is required when input-mode uses PatchCore guidance")
        anomaly_outputs = np.load(os.path.join(args.patchcore_output_dir, "anomaly_outputs.npz"))
        anomaly_maps = anomaly_outputs["anomaly_maps"]
        roi_sample_indices = np.unique(np.concatenate([train_indices, eval_indices])).astype(np.int64)
        if args.input_mode in {"roi", "roi_letterbox", "dual"}:
            roi_boxes = build_roi_boxes(
                anomaly_maps=anomaly_maps,
                sample_indices=roi_sample_indices,
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
        if args.input_mode == "soft_attention":
            attention_maps = anomaly_maps
        if args.input_mode == "component_mask":
            component_mask_maps = anomaly_maps

    if args.input_mode == "dual":
        backbone = timm.create_model(args.model_name, pretrained=True, num_classes=0)
        feature_dim = int(backbone.num_features)
        model = SharedBackboneDualClassifier(
            backbone=backbone,
            feature_dim=feature_dim,
            num_classes=len(known_classes),
        )
        trainable_params = apply_finetune_scope(model.backbone, args.finetune_scope)
        for param in model.head.parameters():
            param.requires_grad = True
        trainable_params += int(sum(param.numel() for param in model.head.parameters()))
        data_config = resolve_model_data_config(backbone)
    else:
        model = timm.create_model(args.model_name, pretrained=True, num_classes=len(known_classes))
        trainable_params = apply_finetune_scope(model, args.finetune_scope)
        data_config = resolve_model_data_config(model)
    model.arcface_head = None
    if args.train_loss == "ce_arcface":
        embedding_dim = get_embedding_dim(model, args.input_mode)
        if hasattr(model, "head") and isinstance(model.head, nn.Module):
            for param in model.head.parameters():
                param.requires_grad = False
        model.arcface_head = ArcFaceHead(
            embedding_dim=embedding_dim,
            num_classes=len(known_classes),
            scale=args.arcface_scale,
            margin=args.arcface_margin,
        )
        trainable_params = int(sum(param.numel() for param in model.parameters() if param.requires_grad))
    transform = create_transform(**data_config, is_training=False)
    train_augment = None
    if args.train_augment == "weak_spec":
        train_augment = WeakSpectrogramAugment(seed=args.seed)

    train_ds = SpectrogramDataset(
        images,
        train_label_ids,
        train_split_indices,
        transform,
        augment=train_augment,
        roi_boxes=roi_boxes,
        return_full_and_roi=(args.input_mode == "dual"),
        attention_maps=attention_maps,
        attention_floor=args.attention_floor,
        attention_power=args.attention_power,
        component_mask_maps=component_mask_maps,
        mask_quantile=args.mask_quantile,
        mask_background_floor=args.mask_background_floor,
        letterbox_roi=(args.input_mode == "roi_letterbox"),
    )
    val_ds = None
    if args.early_stopping and val_split_indices.size > 0:
        val_ds = SpectrogramDataset(
            images,
            val_label_ids,
            val_split_indices,
            transform,
            roi_boxes=roi_boxes,
            return_full_and_roi=(args.input_mode == "dual"),
            attention_maps=attention_maps,
            attention_floor=args.attention_floor,
            attention_power=args.attention_power,
            component_mask_maps=component_mask_maps,
            mask_quantile=args.mask_quantile,
            mask_background_floor=args.mask_background_floor,
            letterbox_roi=(args.input_mode == "roi_letterbox"),
        )
    eval_ds = SpectrogramDataset(
        images,
        eval_label_ids,
        eval_indices,
        transform,
        roi_boxes=roi_boxes,
        return_full_and_roi=(args.input_mode == "dual"),
        attention_maps=attention_maps,
        attention_floor=args.attention_floor,
        attention_power=args.attention_power,
        component_mask_maps=component_mask_maps,
        mask_quantile=args.mask_quantile,
        mask_background_floor=args.mask_background_floor,
        letterbox_roi=(args.input_mode == "roi_letterbox"),
    )
    calibration_ds = None
    if calibration_indices.size > 0:
        calibration_ds = SpectrogramDataset(
            images,
            calibration_label_ids,
            calibration_indices,
            transform,
            roi_boxes=roi_boxes,
            return_full_and_roi=(args.input_mode == "dual"),
            attention_maps=attention_maps,
            attention_floor=args.attention_floor,
            attention_power=args.attention_power,
            component_mask_maps=component_mask_maps,
            mask_quantile=args.mask_quantile,
            mask_background_floor=args.mask_background_floor,
            letterbox_roi=(args.input_mode == "roi_letterbox"),
        )
    if args.early_stopping and args.early_stopping_use_calibration:
        if calibration_ds is None:
            raise ValueError(
                "--early-stopping-use-calibration requires --calibration-per-class > 0"
            )
        val_ds = calibration_ds

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    calibration_loader = None
    if calibration_ds is not None:
        calibration_loader = DataLoader(
            calibration_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    center_loss_module = None
    if args.train_loss == "ce_center":
        center_loss_module = CenterLoss(
            num_classes=len(known_classes),
            embedding_dim=get_embedding_dim(model, args.input_mode),
        ).to(device)
    log(
        f"Training {args.model_name} on {device.type} with {train_split_indices.shape[0]} train samples"
        f"{f' and {(calibration_indices.shape[0] if args.early_stopping_use_calibration else val_split_indices.shape[0])} val samples' if args.early_stopping else ''} "
        f"{f' and {calibration_indices.shape[0]} calibration samples' if calibration_indices.size > 0 else ''} "
        f"(sampled={train_indices.shape[0]}, finetune_scope={args.finetune_scope}, "
        f"trainable_params={trainable_params})"
    )
    history = train_model(
        model=model,
        center_loss_module=center_loss_module,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        train_loss=args.train_loss,
        supcon_weight=args.supcon_weight,
        supcon_temperature=args.supcon_temperature,
        center_loss_weight=args.center_loss_weight,
        center_loss_lr=args.center_loss_lr,
        early_stopping=args.early_stopping,
        early_stopping_monitor=args.early_stopping_monitor,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        threshold_quantile=args.threshold_quantile,
    )
    with open(os.path.join(args.output_dir, "train_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(args.output_dir, "trainable_params.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "finetune_scope": args.finetune_scope,
                "trainable_params": trainable_params,
            },
            f,
            indent=2,
        )

    prototype_gate = None
    if args.reject_mode == "energy_prototype":
        train_eval_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        calib_df, calib_embeddings = infer_with_embeddings(
            model=model,
            loader=train_eval_loader,
            device=device,
            temperature=args.temperature,
        )
        prototypes_by_class = {}
        thresholds_by_class = {}
        for class_name, class_id in known_to_id.items():
            class_mask = calib_df["true_known_id"].to_numpy() == class_id
            class_embeddings = calib_embeddings[class_mask]
            if args.distance_metric == "cosine":
                normalized = _l2_normalize(class_embeddings)
                prototype = normalized.mean(axis=0).astype(np.float32)
                prototype = prototype / max(float(np.linalg.norm(prototype)), 1e-12)
            else:
                prototype = class_embeddings.mean(axis=0).astype(np.float32)
            distances = _compute_class_distances(
                embeddings=class_embeddings,
                prototype=prototype,
                metric=args.distance_metric,
            )
            prototypes_by_class[class_name] = prototype
            thresholds_by_class[class_name] = float(np.percentile(distances, args.distance_quantile))
        prototype_gate = {
            "metric": args.distance_metric,
            "quantile": float(args.distance_quantile),
            "thresholds_by_class": thresholds_by_class,
        }
        with open(os.path.join(args.output_dir, "prototype_gate.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metric": args.distance_metric,
                    "quantile": float(args.distance_quantile),
                    "thresholds_by_class": thresholds_by_class,
                },
                f,
                indent=2,
            )
        eval_df, eval_embeddings = infer_with_embeddings(
            model=model,
            loader=eval_loader,
            device=device,
            temperature=args.temperature,
        )
    else:
        eval_df = infer(
            model=model,
            loader=eval_loader,
            device=device,
            temperature=args.temperature,
        )
        eval_embeddings = None
    eval_df["true_label"] = eval_true_names
    eval_df.to_csv(os.path.join(args.output_dir, "eval_logits.csv"), index=False)

    if args.reject_mode in {"energy", "energy_prototype"} and args.thresholds.strip().lower() == "train_calibrated":
        threshold_loader = None
        if args.threshold_source == "external_calibration":
            threshold_loader = calibration_loader
            threshold_source = "external_calibration"
        elif args.threshold_source == "train":
            threshold_source = "train"
        else:
            threshold_loader = calibration_loader
            threshold_source = "external_calibration" if calibration_loader is not None else "train"

        if threshold_loader is None:
            threshold_loader = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )
            threshold_source = "train"
        train_calib_df = infer(
            model=model,
            loader=threshold_loader,
            device=device,
            temperature=args.temperature,
        )
        thresholds = [float(np.percentile(train_calib_df["energy"].to_numpy(), args.threshold_quantile))]
        with open(os.path.join(args.output_dir, "train_calibrated_threshold.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "threshold_source": threshold_source,
                    "threshold_quantile": float(args.threshold_quantile),
                    "threshold": float(thresholds[0]),
                },
                f,
                indent=2,
            )
    elif args.reject_mode in {"energy", "energy_prototype"} and args.thresholds.strip().lower() == "auto":
        percentiles = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 97, 99]
        thresholds = [float(np.percentile(eval_df["energy"].to_numpy(), p)) for p in percentiles]
    else:
        thresholds = [float(part.strip()) for part in args.thresholds.split(",") if part.strip()]

    true_label_names = all_anomaly_classes
    pred_label_names = ["unknown", *known_classes]
    threshold_rows = []
    confusion_sums = {}
    for threshold in thresholds:
        pred_names = []
        for row_idx, (_, row) in enumerate(eval_df.iterrows()):
            if args.reject_mode == "max_softmax":
                reject = float(row["max_softmax"]) < threshold
            elif args.reject_mode == "energy":
                reject = float(row["energy"]) > threshold
            else:
                pred_known_name = known_classes[int(row["pred_known_id"])]
                class_threshold = float(prototype_gate["thresholds_by_class"][pred_known_name])
                sample_distance = float(
                    _compute_class_distances(
                        embeddings=eval_embeddings[row_idx : row_idx + 1],
                        prototype=np.asarray(prototypes_by_class[pred_known_name], dtype=np.float32),
                        metric=args.distance_metric,
                    )[0]
                )
                reject = (float(row["energy"]) > threshold) or (sample_distance > class_threshold)
            if reject:
                pred_names.append("unknown")
            else:
                pred_names.append(known_classes[int(row["pred_known_id"])])

        known_mask = eval_df["true_label"].isin(known_classes).to_numpy()
        known_true = eval_df.loc[known_mask, "true_label"].tolist()
        known_pred = [pred for pred, keep in zip(pred_names, known_mask.tolist()) if keep]
        known_coverage = float(np.mean([pred != "unknown" for pred in known_pred])) if known_pred else 0.0
        accepted_true = [t for t, p in zip(known_true, known_pred) if p != "unknown"]
        accepted_pred = [p for p in known_pred if p != "unknown"]
        macro_f1_known = float(
            f1_score(accepted_true, accepted_pred, labels=known_classes, average="macro", zero_division=0)
        ) if accepted_true else 0.0

        unknown_reject_rates = {}
        for class_name in unknown_classes:
            class_mask = eval_df["true_label"] == class_name
            class_pred = [pred for pred, keep in zip(pred_names, class_mask.tolist()) if keep]
            unknown_reject_rates[class_name] = float(np.mean([pred == "unknown" for pred in class_pred]))
        avg_unknown_reject = float(np.mean(list(unknown_reject_rates.values())))

        row_to_idx = {name: idx for idx, name in enumerate(true_label_names)}
        col_to_idx = {name: idx for idx, name in enumerate(pred_label_names)}
        cm = np.zeros((len(true_label_names), len(pred_label_names)), dtype=np.float64)
        for true_name, pred_name in zip(eval_df["true_label"].tolist(), pred_names):
            cm[row_to_idx[true_name], col_to_idx[pred_name]] += 1
        confusion_sums[threshold] = cm

        row = {
            "threshold": threshold,
            "known_coverage": known_coverage,
            "macro_f1_known": macro_f1_known,
            "avg_unknown_reject_rate": avg_unknown_reject,
        }
        if args.reject_mode == "energy_prototype":
            row["distance_quantile"] = float(args.distance_quantile)
            row["distance_metric"] = args.distance_metric
        for class_name in unknown_classes:
            row[f"{class_name}_reject_rate"] = unknown_reject_rates[class_name]
        threshold_rows.append(row)

    summary_df = pd.DataFrame(threshold_rows).sort_values(
        ["avg_unknown_reject_rate", "macro_f1_known", "known_coverage"],
        ascending=[False, False, False],
    )
    summary_df.to_csv(os.path.join(args.output_dir, "threshold_summary.csv"), index=False)

    feasible_df = summary_df[summary_df["known_coverage"] >= args.coverage_constraint]
    if feasible_df.empty:
        best = {"feasible": False, "coverage_constraint": args.coverage_constraint}
    else:
        best = feasible_df.iloc[0].to_dict()
        best["feasible"] = True
        best["coverage_constraint"] = args.coverage_constraint
        best_threshold = float(best["threshold"])
        save_rectangular_confusion(
            matrix=confusion_sums[best_threshold],
            row_labels=true_label_names,
            col_labels=pred_label_names,
            output_path=os.path.join(args.output_dir, "best_confusion_matrix.png"),
            title=f"ConvNeXtV2 Open-Set {args.reject_mode} (thr={best_threshold:.2f})",
        )
    with open(os.path.join(args.output_dir, "best_under_coverage_constraint.json"), "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    log(
        "Finished ConvNeXtV2 open-set experiment. "
        f"Best feasible={bool(best.get('feasible', False))}"
    )


if __name__ == "__main__":
    main()
