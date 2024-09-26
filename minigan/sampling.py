"""Deterministic latent sampling, batched generation and latent-space interpolation."""

from __future__ import annotations

import torch

from minigan.models import Generator


def latent_noise(n: int, latent_dim: int, seed: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """Standard-normal latents drawn on the CPU from a dedicated seeded generator.

    Drawing on the CPU makes the latents identical whatever the target device.
    """
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(n, latent_dim, generator=gen).to(device)


def balanced_labels(n: int, n_classes: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """Labels ``0, 1, ..., n_classes-1, 0, 1, ...`` (exactly balanced when ``n % n_classes == 0``)."""
    return (torch.arange(n) % n_classes).to(device)


@torch.no_grad()
def decode(
    generator: Generator, z: torch.Tensor, labels: torch.Tensor | None = None, batch_size: int = 1000
) -> torch.Tensor:
    """Map latents (and labels, if conditional) to images in eval mode (BatchNorm running stats).

    For a conditional generator, ``labels`` defaults to :func:`balanced_labels`.
    Returns a tensor of shape ``(len(z), 1, 28, 28)`` on the generator's device.
    """
    device = next(generator.parameters()).device
    z = z.to(device)
    if generator.config.conditional and labels is None:
        labels = balanced_labels(len(z), generator.config.n_classes)
    was_training = generator.training
    generator.eval()
    chunks = []
    for start in range(0, len(z), batch_size):
        y = None if labels is None else labels[start : start + batch_size].to(device)
        chunks.append(generator(z[start : start + batch_size], y))
    generator.train(was_training)
    return torch.cat(chunks)


def generate(
    generator: Generator, n: int, *, seed: int, labels: torch.Tensor | None = None, batch_size: int = 1000
) -> torch.Tensor:
    """Decode ``n`` latents drawn with :func:`latent_noise` from ``seed`` (see :func:`decode`)."""
    return decode(generator, latent_noise(n, generator.latent_dim, seed), labels, batch_size)


