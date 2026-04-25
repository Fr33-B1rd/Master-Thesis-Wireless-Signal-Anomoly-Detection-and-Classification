"""Dataset + DataLoader wrappers for the ConvNeXtV2 classifier.

``SpectrogramDataset`` dispatches between three input paths at runtime:

- 3D ndarray (C, H, W) -> consumed by ``ThreeChannelTransform`` (numpy in,
  tensor out, no PIL)
- 2D ndarray + ``_mag_1ch_native=True`` -> consumed by ``MagOneChannelTransform``
  (numpy in, tensor out, no PIL)
- 2D ndarray + ``_mag_1ch_native=False`` -> legacy path, replicate to RGB
  via PIL and run a timm ImageNet transform

The ``_mag_1ch_native`` flag is set by ``build_loader`` based on whether the
transform (possibly wrapped in a torchvision Compose for training-time noise
augmentation) contains a ``MagOneChannelTransform``.
"""

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from sas_open_world.experiments.convnextv2_open_set import RoiBox
from sas_open_world.models.transforms import transform_is_mag_1ch_native


class SpectrogramDataset(Dataset):
    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
        transform,
        roi_boxes: dict[int, RoiBox] | None = None,
    ) -> None:
        self.images = images
        self.labels = labels
        self.indices = indices
        self.transform = transform
        self.roi_boxes = roi_boxes or {}

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int):
        sample_index = int(self.indices[idx])
        image = self.images[sample_index]
        roi_box = self.roi_boxes.get(sample_index)
        # Dispatch rule: 3D ndarray -> ThreeChannelTransform (consumes ndarray).
        # 2D ndarray -> either MagOneChannelTransform (ndarray) or legacy
        # PIL-RGB path. The transform may be wrapped in a Compose (when
        # train_noise_std>0), so we detect mag_1ch via _mag_1ch_native flag
        # set by build_loader rather than via isinstance.
        if image.ndim == 3:
            image = np.asarray(image)
            if roi_box is not None:
                image = image[:, roi_box.top : roi_box.bottom, roi_box.left : roi_box.right]
            tensor = self.transform(image)
        elif getattr(self, "_mag_1ch_native", False):
            # True 1-channel magnitude path: 2D ndarray consumed directly.
            image = np.asarray(image)
            if roi_box is not None:
                image = image[roi_box.top : roi_box.bottom, roi_box.left : roi_box.right]
            tensor = self.transform(image)
        else:
            # Legacy single-channel magnitude path: replicate to RGB via PIL.
            if roi_box is not None:
                image = image[roi_box.top : roi_box.bottom, roi_box.left : roi_box.right]
            tensor = self.transform(Image.fromarray(image).convert("RGB"))
        return tensor, int(self.labels[idx]), sample_index


def build_loader(
    images,
    indices: list[int],
    label_names: np.ndarray,
    class_to_id: dict[str, int],
    transform,
    batch_size: int,
    roi_boxes: dict[int, RoiBox] | None = None,
):
    indices_array = np.asarray(indices, dtype=np.int64)
    labels_array = np.asarray(
        [class_to_id[str(label_names[idx])] for idx in indices_array],
        dtype=np.int64,
    )
    dataset = SpectrogramDataset(
        images=images,
        labels=labels_array,
        indices=indices_array,
        transform=transform,
        roi_boxes=roi_boxes,
    )
    dataset._mag_1ch_native = transform_is_mag_1ch_native(transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
