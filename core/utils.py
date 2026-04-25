# Generic numeric/logging helpers shared across core modules.
#
# These are not PatchCore-specific; they live here so consumers can pull them
# without importing the full model module.

import time

import numpy as np
import torch


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def signed_log1p_np(array: np.ndarray) -> np.ndarray:
    return np.sign(array) * np.log1p(np.abs(array))


def compute_finite_mean_std(
    array: np.ndarray, chunk_size: int = 64, use_log_transform: bool = False
) -> tuple[float, float]:
    """Streaming mean/std over an (N, ...) array, ignoring non-finite values.

    If `use_log_transform=True`, applies `signed_log1p_np` to each chunk before
    accumulation (used for heavy-tailed inputs).
    """
    total_sum = 0.0
    total_sq_sum = 0.0
    total_count = 0
    num_samples = int(array.shape[0])
    for start in range(0, num_samples, chunk_size):
        end = min(start + chunk_size, num_samples)
        chunk = np.asarray(array[start:end], dtype=np.float32)
        finite_mask = np.isfinite(chunk)
        finite_values = chunk[finite_mask]
        if finite_values.size == 0:
            continue
        if use_log_transform:
            finite_values = signed_log1p_np(finite_values)
        total_sum += float(finite_values.sum(dtype=np.float64))
        total_sq_sum += float(np.square(finite_values, dtype=np.float64).sum(dtype=np.float64))
        total_count += int(finite_values.size)

    if total_count == 0:
        raise ValueError("No finite values found in dataset")

    mean = total_sum / total_count
    variance = max(total_sq_sum / total_count - mean * mean, 1e-12)
    return float(mean), float(np.sqrt(variance))
