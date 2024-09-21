"""Shared test utilities."""

from __future__ import annotations

from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def fake_batch(n: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """Random images in [-1, 1] and random labels."""
    return torch.rand(n, 1, 28, 28) * 2 - 1, torch.randint(0, 10, (n,))
