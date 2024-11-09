from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from helpers import fake_batch
from minigan.classifier import MNISTClassifier
from minigan.metrics import Evaluator
from minigan.models import Discriminator, Generator, ModelConfig
from minigan.training import STEP_KEYS, TrainConfig, fit, make_optimizers, train_step


def _snapshot(module: torch.nn.Module) -> list[torch.Tensor]:
    return [p.detach().clone() for p in module.parameters()]


def _changed(before: list[torch.Tensor], module: torch.nn.Module) -> bool:
    return any(not torch.equal(b, p) for b, p in zip(before, module.parameters()))


def test_train_step_updates_both_networks_on_cpu(tiny_config: ModelConfig) -> None:
    generator, discriminator = Generator(tiny_config), Discriminator(tiny_config)
    opt_g, opt_d = make_optimizers(generator, discriminator, TrainConfig())
    g_before, d_before = _snapshot(generator), _snapshot(discriminator)
    real, labels = fake_batch()

    stats = train_step(generator, discriminator, opt_g, opt_d, real, labels, label_smoothing=0.1)

    assert set(stats) == set(STEP_KEYS)
    assert all(math.isfinite(v) for v in stats.values())
    assert 0.0 <= stats["d_real"] <= 1.0 and 0.0 <= stats["d_fake"] <= 1.0
    assert _changed(g_before, generator)
    assert _changed(d_before, discriminator)
    assert next(generator.parameters()).device.type == "cpu"


def test_optimizers_use_dcgan_settings() -> None:
    opt_g, opt_d = make_optimizers(Generator(), Discriminator(), TrainConfig())
    for opt in (opt_g, opt_d):
        assert opt.defaults["lr"] == 2e-4
        assert opt.defaults["betas"] == (0.5, 0.999)


def test_discriminator_loss_starts_near_two_ln2() -> None:
    # Freshly initialised D outputs logits ~ 0, i.e. D ~ 1/2, so L_D ~ 2 ln 2 and L_G ~ ln 2.
    config = ModelConfig(latent_dim=16, g_width=8, d_width=8)
    generator, discriminator = Generator(config), Discriminator(config)
    opt_g, opt_d = make_optimizers(generator, discriminator, TrainConfig())
    stats = train_step(generator, discriminator, opt_g, opt_d, *fake_batch(32))
    assert abs(stats["loss_d"] - 2 * math.log(2)) < 0.3
    assert abs(stats["loss_g"] - math.log(2)) < 0.3


def test_fit_runs_and_writes_artifacts(tmp_path: Path, tiny_config: ModelConfig) -> None:
    images, labels = fake_batch(64)
    generator, discriminator = Generator(tiny_config), Discriminator(tiny_config)
    config = TrainConfig(epochs=2, batch_size=16)
    opt_g, opt_d = make_optimizers(generator, discriminator, config)
    evaluator = Evaluator(MNISTClassifier(), images)

    history = fit(
        generator, discriminator, images, labels, config,
        opt_g=opt_g, opt_d=opt_d, out_dir=tmp_path, evaluator=evaluator, eval_samples=50, log_every=0,
    )

    assert len(history.steps["loss_d"]) == 2 * (64 // 16)
    assert [e["epoch"] for e in history.epochs] == [0, 1, 2]
    assert all("mnist_fid" in e and "mnist_is" in e for e in history.epochs)
    assert ("label_accuracy" in history.epochs[-1]) == tiny_config.conditional
    for name in ("frames/epoch_000.png", "frames/epoch_002.png", "checkpoint.pt", "history.json"):
        assert (tmp_path / name).exists()
    assert json.loads((tmp_path / "history.json").read_text())["epochs"][-1]["epoch"] == 2
