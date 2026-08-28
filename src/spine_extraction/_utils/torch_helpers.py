"""PyTorch device management and utility helpers."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def get_device(device_str: str = "auto") -> torch.device:
    """Resolve device string to a torch.device.

    Args:
        device_str: One of 'cpu', 'cuda', or 'auto'.
            'auto' selects CUDA if available, otherwise CPU.
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            name = torch.cuda.get_device_name(0)
            free, total = torch.cuda.mem_get_info()
            logger.info(
                "Using CUDA device: %s (%.1f / %.1f GB VRAM free)",
                name, free / 1024**3, total / 1024**3,
            )
        else:
            device = torch.device("cpu")
            logger.info("CUDA not available, using CPU")
    else:
        device = torch.device(device_str)
        logger.info("Using device: %s", device)
    return device


def to_torch(arr, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert numpy array to torch tensor on the specified device."""
    import numpy as np

    if isinstance(arr, torch.Tensor):
        return arr.to(device=device, dtype=dtype)
    return torch.from_numpy(np.ascontiguousarray(arr)).to(device=device, dtype=dtype)


def to_numpy(tensor: torch.Tensor):
    """Convert torch tensor to numpy array on CPU."""
    return tensor.detach().cpu().numpy()
