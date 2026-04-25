"""Input transforms and stem surgery for the ConvNeXtV2 classifier.

Two input paths:
- 3-channel pre-packed arrays (log-mag, IF, GD) via ``ThreeChannelTransform``
- true 1-channel magnitude via ``MagOneChannelTransform``

Two stem-surgery variants, matched to those paths:
- ``_apply_stem_surgery_for_1channel``: replace the first Conv2d with
  in_channels=1 initialized from the SUM of pretrained RGB weights, preserving
  the R=G=B equivalent behavior while freeing fine-tuning from the
  color-opponent-cancelling init.
- ``_apply_stem_surgery_for_3channel``: keep in_channels=3 but re-initialize
  each channel from the average of pretrained RGB weights plus a small
  perturbation, giving the three physically distinct channels a symmetric
  prior instead of the R=G=B collapse implicit in timm's default.
"""

import numpy as np
import torch
import torch.nn.functional as F


class GaussianNoiseAugment:
    """Add zero-mean Gaussian noise to a normalized (C, H, W) tensor.

    Physical analogue for RF: simulates SNR fluctuations. Does not delete
    discriminative features — preserves all localized F/T patterns while
    teaching the model robustness to amplitude perturbations.
    """

    def __init__(self, std: float = 0.05, prob: float = 1.0) -> None:
        self.std = std
        self.prob = prob

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() >= self.prob:
            return tensor
        return tensor + torch.randn_like(tensor) * self.std


class ThreeChannelTransform:
    """Tensor-in / tensor-out transform for pre-packed 3-channel spectrograms.

    Takes a numpy (3, H, W) float16 or float32 array, returns a (3, out, out)
    float32 tensor with per-channel (mean/std) standardization applied. The
    per-channel statistics come from channel_stats.json, fit on the train
    split by build_rf_3channel.py. Mimics the ImageNet Normalize step of
    timm's PIL-based transform but skips all PIL / RGB replication.
    """

    def __init__(
        self,
        mean: tuple[float, float, float] | list[float],
        std: tuple[float, float, float] | list[float],
        out_size: int,
        zero_channels: tuple[int, ...] = (),
    ) -> None:
        self._mean = torch.as_tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self._std = torch.as_tensor(std, dtype=torch.float32).view(3, 1, 1)
        self._std = torch.clamp(self._std, min=1e-6)
        self._out_size = int(out_size)
        # Channels in ``zero_channels`` are forced to 0 AFTER normalization so
        # the first-conv filters on those channels multiply zero and contribute
        # nothing to downstream activations. This simulates a "true N-channel"
        # input without replacing the model's stem. Used for mag+IF-only and
        # mag+GD-only ablations against mag+IF+GD.
        self._zero_channels = tuple(int(c) for c in zero_channels)

    def __call__(self, arr_chw: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(np.ascontiguousarray(arr_chw)).float()
        if tensor.ndim != 3 or tensor.shape[0] != 3:
            raise ValueError(
                f"ThreeChannelTransform expects (3, H, W) array, got {tuple(tensor.shape)}"
            )
        # Match timm's default behavior of resizing to the model's input size.
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=(self._out_size, self._out_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        tensor = (tensor - self._mean) / self._std
        for c in self._zero_channels:
            tensor[c].zero_()
        return tensor


class MagOneChannelTransform:
    """Tensor-in / tensor-out transform for true 1-channel magnitude input.

    Takes a numpy (H, W) float array (the pre-packed grayscale magnitude
    spectrogram), returns a (1, out, out) float32 tensor with single-channel
    (mean/std) standardization applied. Used with ``--input-channels=mag_1ch``
    to avoid the R=G=B PIL-replication path that depresses the ImageNet
    pretrained stem via color-opponent filter cancellation.
    """

    def __init__(self, mean: float, std: float, out_size: int) -> None:
        self._mean = torch.as_tensor([mean], dtype=torch.float32).view(1, 1, 1)
        self._std = torch.as_tensor([std], dtype=torch.float32).view(1, 1, 1)
        self._std = torch.clamp(self._std, min=1e-6)
        self._out_size = int(out_size)

    def __call__(self, arr_hw: np.ndarray) -> torch.Tensor:
        arr = np.ascontiguousarray(arr_hw)
        if arr.ndim != 2:
            raise ValueError(
                f"MagOneChannelTransform expects (H, W) array, got {arr.shape}"
            )
        # The legacy test_spectrogram.npy is stored as uint8 in [0, 255] while
        # channel_stats (mean/std) was fit on the float16 3c pack Ch0 in [0, 1].
        # Rescale uint8 to [0, 1] so normalization matches.
        if arr.dtype == np.uint8:
            tensor = torch.from_numpy(arr).float() / 255.0
        else:
            tensor = torch.from_numpy(arr).float()
        tensor = tensor.unsqueeze(0)  # (1, H, W)
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=(self._out_size, self._out_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        tensor = (tensor - self._mean) / self._std
        return tensor


def _apply_stem_surgery_for_1channel(model, perturb_std_ratio: float, seed: int) -> None:
    """Replace the first stem Conv2d (in_channels=3) with a new Conv2d
    (in_channels=1), initialized from the SUM of pretrained RGB weights.

    Behavior justification: for an RGB-pretrained stem fed with R=G=B=v, the
    output equals (W_R + W_G + W_B) * v. Setting the 1-channel filter weight
    to W_sum = W_R + W_G + W_B therefore preserves that exact behavior on
    grayscale inputs, so this is the mathematically equivalent way to run a
    true 1-channel classifier without losing the pretrained stem's learned
    spatial structure. A small Gaussian perturbation breaks the rigid
    color-opponent-cancelling initialization so fine-tuning has a gradient
    signal to diverge from the RGB prior.
    """
    first_conv_name = None
    first_conv = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            first_conv_name = name
            first_conv = module
            break
    if first_conv is None:
        raise RuntimeError("No Conv2d found in model; cannot apply 1-channel stem surgery")
    if first_conv.in_channels != 3:
        raise RuntimeError(
            f"Expected stem Conv2d in_channels=3, found {first_conv.in_channels}"
        )
    new_conv = torch.nn.Conv2d(
        in_channels=1,
        out_channels=first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        dilation=first_conv.dilation,
        groups=first_conv.groups,
        bias=(first_conv.bias is not None),
        padding_mode=first_conv.padding_mode,
    ).to(first_conv.weight.device)
    with torch.no_grad():
        w = first_conv.weight.data  # (out, 3, k, k)
        w_sum = w.sum(dim=1, keepdim=True)  # (out, 1, k, k)
        w_sigma = float(w.std().item())
        perturb_std = float(perturb_std_ratio) * max(w_sigma, 1e-8)
        gen = torch.Generator(device=w.device).manual_seed(int(seed))
        noise = torch.empty_like(w_sum).normal_(mean=0.0, std=perturb_std, generator=gen)
        new_conv.weight.data.copy_(w_sum + noise)
        if first_conv.bias is not None:
            new_conv.bias.data.copy_(first_conv.bias.data)
    # Swap the conv in its parent module.
    parent_name, _, attr_name = first_conv_name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, attr_name, new_conv)


def _apply_stem_surgery_for_3channel(model, perturb_std_ratio: float, seed: int) -> None:
    """Re-initialize the first stem conv so each input channel starts from
    the average of the pretrained RGB weights plus a small Gaussian
    perturbation. Keeps in_channels=3 and out_channels=96 unchanged.

    Rationale: replicating a grayscale input to R=G=B collapses the
    pretrained stem to a 1-channel equivalent filter (W_R+W_G+W_B). When
    the three channels carry distinct physical quantities (log-mag, IF,
    GD), starting each channel from the same balanced prior gives the
    fine-tuning run a symmetric initialization that can then differentiate
    per-channel filters during training.
    """
    stem_conv = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            stem_conv = module
            break
    if stem_conv is None:
        raise RuntimeError("No Conv2d found in model; cannot apply stem surgery")
    if stem_conv.in_channels != 3:
        raise RuntimeError(
            f"Expected stem Conv2d in_channels=3, found {stem_conv.in_channels}"
        )
    with torch.no_grad():
        w = stem_conv.weight.data
        avg = w.mean(dim=1, keepdim=True)
        w_sigma = float(w.std().item())
        perturb_std = float(perturb_std_ratio) * max(w_sigma, 1e-8)
        gen = torch.Generator(device=w.device).manual_seed(int(seed))
        noise = torch.empty_like(w).normal_(mean=0.0, std=perturb_std, generator=gen)
        new_w = avg.repeat(1, 3, 1, 1) + noise
        stem_conv.weight.data.copy_(new_w)


def transform_is_mag_1ch_native(transform) -> bool:
    """Return True if the given transform (possibly a torchvision Compose)
    consumes a raw 2D ndarray directly, i.e. is / contains a
    ``MagOneChannelTransform``. Needed because train-time augmentation wraps
    the transform in a Compose, so a bare isinstance check on the outer
    object would miss the wrapped case.
    """
    if isinstance(transform, MagOneChannelTransform):
        return True
    inner = getattr(transform, "transforms", None)
    if inner:
        for t in inner:
            if isinstance(t, MagOneChannelTransform):
                return True
    return False
