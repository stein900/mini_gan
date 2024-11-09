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


def fit(
    generator: Generator,
    discriminator: Discriminator,
    images: torch.Tensor,
    labels: torch.Tensor,
    config: TrainConfig,
    *,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    out_dir: Path,
    evaluator: Evaluator | None = None,
    eval_samples: int = 10_000,
    eval_seed: int = 1234,
    history: History | None = None,
    start_epoch: int = 0,
    log_every: int = 100,
) -> History:
    """Train for epochs ``start_epoch+1 .. config.epochs``.

    After every epoch (and once before training when ``start_epoch == 0``) it writes the
    fixed-noise grid ``out_dir/frames/epoch_XXX.png``, scores the generator if an evaluator is
    given, and saves ``out_dir/checkpoint.pt`` and ``out_dir/history.json``.
    """
    history = history or History()
    out_dir.mkdir(parents=True, exist_ok=True)
    if start_epoch == 0:
        history.epochs.append({"epoch": 0, **_end_of_epoch(0, generator, out_dir, evaluator, eval_samples, eval_seed)})

    for epoch in range(start_epoch + 1, config.epochs + 1):
        generator.train()
        discriminator.train()
        shuffle = torch.Generator().manual_seed(config.seed * 1000 + epoch)
        epoch_stats: dict[str, list[float]] = {k: [] for k in STEP_KEYS}
        t0 = time.perf_counter()
        for step, (x, y) in enumerate(iterate_minibatches(images, labels, config.batch_size, generator=shuffle), 1):
            stats = train_step(generator, discriminator, opt_g, opt_d, x, y, config.label_smoothing)
            for key, value in stats.items():
                epoch_stats[key].append(value)
                history.steps[key].append(value)
            if log_every and step % log_every == 0:
                print(
                    f"epoch {epoch:3d} step {step:4d} | L_D {stats['loss_d']:.3f}  L_G {stats['loss_g']:.3f} "
                    f"| D(x) {stats['d_real']:.3f}  D(G(z)) {stats['d_fake']:.3f}"
                )
        train_time = time.perf_counter() - t0

        summary: dict[str, float] = {"epoch": epoch, "train_time_s": round(train_time, 2)}
        summary.update({k: sum(v) / len(v) for k, v in epoch_stats.items()})
        summary.update(_end_of_epoch(epoch, generator, out_dir, evaluator, eval_samples, eval_seed))
        history.epochs.append(summary)
        metrics = "".join(f"  {k} {summary[k]:.3f}" for k in ("mnist_fid", "mnist_is", "label_accuracy") if k in summary)
        print(f"== epoch {epoch}/{config.epochs} ({train_time:.1f}s){metrics}")

        save_training_state(
            out_dir / "checkpoint.pt",
            generator=generator,
            discriminator=discriminator,
            opt_g=opt_g,
            opt_d=opt_d,
            epoch=epoch,
            extra={"train_config": asdict(config)},
        )
        (out_dir / "history.json").write_text(json.dumps(history.to_dict()), encoding="utf-8")
    return history
