"""DINO feature extraction for SAS clustering experiments."""

import os

import numpy as np
import timm
import torch
from PIL import Image
from timm.data import resolve_model_data_config
from timm.data.transforms_factory import create_transform
from torch.utils.data import DataLoader, Dataset

from dataio.sas import load_sas_test_split
from sas_open_world.common import log


class DinoSpectrogramDataset(Dataset):
    def __init__(self, images: np.ndarray, indices: np.ndarray, transform) -> None:
        self.images = images
        self.indices = indices.astype(np.int64, copy=False)
        self.transform = transform

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int):
        sample_index = int(self.indices[idx])
        image = self.images[sample_index]
        tensor = self.transform(Image.fromarray(image).convert("RGB"))
        return tensor, sample_index


@torch.no_grad()
def build_dino_feature_cache(
    dataset_dir: str,
    dino_model_name: str,
    cache_path: str,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    if os.path.exists(cache_path):
        cache = np.load(cache_path)
        return cache["X"].astype(np.float32), cache["label_names"].astype(str)

    split = load_sas_test_split(dataset_dir)
    test_label_names = split.label_names
    images = split.images
    indices = np.arange(test_label_names.shape[0], dtype=np.int64)

    model = timm.create_model(dino_model_name, pretrained=True, num_classes=0).to(device)
    model.eval()
    data_config = resolve_model_data_config(model)
    transform = create_transform(**data_config, is_training=False)

    loader = DataLoader(
        DinoSpectrogramDataset(images=images, indices=indices, transform=transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    log(f"Building DINO features with {dino_model_name} for {indices.shape[0]} test samples")
    features = []
    sample_indices = []
    for batch, batch_indices in loader:
        batch = batch.to(device, non_blocking=True)
        embedding = model(batch)
        features.append(embedding.detach().cpu().numpy().astype(np.float32, copy=False))
        sample_indices.append(batch_indices.numpy().astype(np.int64, copy=False))

    x_all = np.concatenate(features, axis=0)
    sample_indices = np.concatenate(sample_indices, axis=0)
    order = np.argsort(sample_indices)
    x_all = x_all[order]
    np.savez_compressed(cache_path, X=x_all, label_names=test_label_names.astype("U32"))
    return x_all, test_label_names
