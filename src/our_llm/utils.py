"""Small shared helpers: device selection, seeding, parameter counting."""

from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn


def get_device() -> torch.device:
    """CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int = 123) -> None:
    """Seed Python, NumPy and torch (both CPU and CUDA generators)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    return sum(
        p.numel() for p in module.parameters() if p.requires_grad or not trainable_only
    )


def human_count(value: int | float) -> str:
    """`13370000` -> `13.37M`."""
    for unit, size in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(value) >= size:
            return f"{value / size:.2f}{unit}"
    return f"{value:.0f}"
