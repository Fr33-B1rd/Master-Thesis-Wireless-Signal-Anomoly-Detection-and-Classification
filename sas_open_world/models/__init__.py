"""ConvNeXtV2 classifier model, transforms, and training utilities.

Split from the historical ``sas_open_world/experiments/convnextv2_open_world.py``
monolith along three planes:

- ``transforms``: input normalization (1-channel and 3-channel) + stem surgery
  (in_channels=3 -> in_channels=1 or RGB averaging). These are tightly coupled
  because the surgery path is determined by the channel mode.
- ``dataset``: the ``SpectrogramDataset`` / ``build_loader`` wrappers that
  dispatch between 3D, 2D-native, and legacy PIL-RGB image paths.
- ``convnextv2``: the ConvNeXtV2 model construction + training + inference +
  the full ``train_convnext_open_world_model`` orchestrator. This is where
  experiment code should hook in if it wants to reuse the trainer.
"""

from sas_open_world.models.convnextv2 import (
    build_protocol,
    create_model,
    infer_with_embeddings,
    predict_sample,
    train_convnext_open_world_model,
    train_model,
)
from sas_open_world.models.dataset import SpectrogramDataset, build_loader
from sas_open_world.models.transforms import (
    GaussianNoiseAugment,
    MagOneChannelTransform,
    ThreeChannelTransform,
)

__all__ = [
    "GaussianNoiseAugment",
    "MagOneChannelTransform",
    "SpectrogramDataset",
    "ThreeChannelTransform",
    "build_loader",
    "build_protocol",
    "create_model",
    "infer_with_embeddings",
    "predict_sample",
    "train_convnext_open_world_model",
    "train_model",
]
