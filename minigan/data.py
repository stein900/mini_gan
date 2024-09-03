"""MNIST loading as in-memory tensors and a lightweight mini-batch iterator.

MNIST is small (60k x 28 x 28), so the whole split is kept on the target device and
batches are drawn by indexing. This avoids DataLoader worker overhead entirely.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import torch
from torchvision import datasets


def load_mnist(
    root: str | Path, train: bool, device: torch.device | str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Download (once) and return MNIST images scaled to [-1, 1] and integer labels.

    Returns:
        ``images`` of shape ``(N, 1, 28, 28)`` (float32) and ``labels`` of shape ``(N,)`` (int64).
    """
    dataset = datasets.MNIST(root=str(root), train=train, download=True)
    images = dataset.data.unsqueeze(1).float().div_(127.5).sub_(1.0)
    return images.to(device), dataset.targets.to(device)


def iterate_minibatches(
    images: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    *,
    shuffle: bool = True,
    drop_last: bool = True,
    generator: torch.Generator | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield ``(images, labels)`` mini-batches; the permutation is drawn on the CPU from ``generator``."""
    n = images.shape[0]
    order = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
    order = order.to(images.device)
    stop = n - n % batch_size if drop_last else n
    for start in range(0, stop, batch_size):
        idx = order[start : start + batch_size]
        yield images[idx], labels[idx]
