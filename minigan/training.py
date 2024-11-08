"""Adversarial training: one optimisation step and the epoch loop.

Discriminator step (maximise the GAN value function)::

    L_D = BCE(D(x), 1 - s) + BCE(D(G(z)), 0)          # s = one-sided label smoothing

Generator step, non-saturating loss (Goodfellow et al., 2014)::

    L_G = BCE(D(G(z)), 1) = -log sigmoid(D(G(z)))
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from minigan.checkpoint import save_training_state
from minigan.data import iterate_minibatches
from minigan.metrics import Evaluator
from minigan.models import Discriminator, Generator
from minigan.sampling import generate
from minigan.viz import save_grid

STEP_KEYS = ("loss_d", "loss_g", "d_real", "d_fake")
FRAME_SEED = 2026  # fixed noise used for the per-epoch sample grid
FRAME_SIZE = 100  # 10 x 10 grid; for a cGAN, column j shows digit j


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 50
    batch_size: int = 128
    lr: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    label_smoothing: float = 0.1
    seed: int = 0


@dataclass
class History:
    """Per-step logs (``steps``) and per-epoch summaries incl. metrics (``epochs``)."""

    steps: dict[str, list[float]] = field(default_factory=lambda: {k: [] for k in STEP_KEYS})
    epochs: list[dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> History:
        return cls(steps={k: list(v) for k, v in data["steps"].items()}, epochs=list(data["epochs"]))


def make_optimizers(
    generator: Generator, discriminator: Discriminator, config: TrainConfig
) -> tuple[torch.optim.Adam, torch.optim.Adam]:
    """Adam for both players with the DCGAN settings (lr=2e-4, betas=(0.5, 0.999) by default)."""
    betas = (config.beta1, config.beta2)
    return (
        torch.optim.Adam(generator.parameters(), lr=config.lr, betas=betas),
        torch.optim.Adam(discriminator.parameters(), lr=config.lr, betas=betas),
    )


def train_step(
    generator: Generator,
    discriminator: Discriminator,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    real: torch.Tensor,
    labels: torch.Tensor,
    label_smoothing: float = 0.0,
) -> dict[str, float]:
    """Run one discriminator update then one generator update on a real mini-batch.

    Returns the two losses, ``d_real`` = mean D(x) and ``d_fake`` = mean D(G(z)) (probabilities,
    measured during the discriminator step).
    """
    n, device = real.shape[0], real.device
    conditional = generator.config.conditional
    z = torch.randn(n, generator.latent_dim, device=device)
    fake_labels = torch.randint(0, generator.config.n_classes, (n,), device=device) if conditional else None
    real_labels = labels if conditional else None
    fake = generator(z, fake_labels)

    # Discriminator: real -> 1 - s (one-sided smoothing), fake -> 0. G is frozen via detach().
    real_logits = discriminator(real, real_labels)
    fake_logits = discriminator(fake.detach(), fake_labels)
    loss_d = F.binary_cross_entropy_with_logits(
        real_logits, torch.full_like(real_logits, 1.0 - label_smoothing)
    ) + F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
    opt_d.zero_grad(set_to_none=True)
    loss_d.backward()
    opt_d.step()

    # Generator: non-saturating loss -log D(G(z)), gradients flow through the updated D.
    gen_logits = discriminator(fake, fake_labels)
    loss_g = F.binary_cross_entropy_with_logits(gen_logits, torch.ones_like(gen_logits))
    opt_g.zero_grad(set_to_none=True)
    loss_g.backward()
    opt_g.step()

    stats = torch.stack(
        [loss_d.detach(), loss_g.detach(), torch.sigmoid(real_logits).mean(), torch.sigmoid(fake_logits).mean()]
    ).tolist()  # a single host-device sync per step
    return dict(zip(STEP_KEYS, stats))


def _end_of_epoch(
    epoch: int,
    generator: Generator,
    out_dir: Path,
    evaluator: Evaluator | None,
    eval_samples: int,
    eval_seed: int,
) -> dict[str, float]:
    save_grid(generate(generator, FRAME_SIZE, seed=FRAME_SEED), out_dir / "frames" / f"epoch_{epoch:03d}.png", nrow=10)
    if evaluator is None or eval_samples <= 0:
        return {}
    return evaluator.score_generator(generator, eval_samples, seed=eval_seed)


