from __future__ import annotations

from pathlib import Path

import pytest
import torch

from helpers import ROOT, fake_batch
from minigan.checkpoint import (
    export_generator,
    load_classifier,
    load_generator,
    load_training_state,
    read_metadata,
    save_classifier,
    save_training_state,
)
from minigan.classifier import MNISTClassifier
from minigan.models import Discriminator, Generator, ModelConfig
from minigan.sampling import generate
from minigan.training import TrainConfig, make_optimizers, train_step


def test_generator_roundtrip_float32_is_exact(tmp_path: Path, tiny_config: ModelConfig) -> None:
    generator = Generator(tiny_config)
    export_generator(tmp_path / "g.pt", generator, half=False, metadata={"epochs": 3})
    restored = load_generator(tmp_path / "g.pt")
    assert restored.config == tiny_config
    assert torch.equal(generate(generator, 8, seed=1), generate(restored, 8, seed=1))
    assert read_metadata(tmp_path / "g.pt") == {"epochs": 3}


def test_generator_roundtrip_float16_is_close_and_smaller(tmp_path: Path) -> None:
    generator = Generator()
    export_generator(tmp_path / "g32.pt", generator, half=False)
    export_generator(tmp_path / "g16.pt", generator, half=True)
    restored = load_generator(tmp_path / "g16.pt")
    assert next(restored.parameters()).dtype == torch.float32
    assert (generate(generator, 16, seed=0) - generate(restored, 16, seed=0)).abs().max() < 1e-2
    assert (tmp_path / "g16.pt").stat().st_size < 0.6 * (tmp_path / "g32.pt").stat().st_size


def test_training_state_roundtrip(tmp_path: Path, tiny_config: ModelConfig) -> None:
    generator, discriminator = Generator(tiny_config), Discriminator(tiny_config)
    opt_g, opt_d = make_optimizers(generator, discriminator, TrainConfig())
    train_step(generator, discriminator, opt_g, opt_d, *fake_batch())
    save_training_state(
        tmp_path / "ckpt.pt", generator=generator, discriminator=discriminator, opt_g=opt_g, opt_d=opt_d, epoch=5
    )

    g2, d2 = Generator(tiny_config), Discriminator(tiny_config)
    o_g2, o_d2 = make_optimizers(g2, d2, TrainConfig())
    payload = load_training_state(tmp_path / "ckpt.pt", generator=g2, discriminator=d2, opt_g=o_g2, opt_d=o_d2)

    assert payload["epoch"] == 5
    for a, b in ((generator, g2), (discriminator, d2)):
        for (key, t1), t2 in zip(a.state_dict().items(), b.state_dict().values()):
            assert torch.equal(t1, t2), key
    first = next(iter(opt_g.state.values()))["exp_avg"]
    assert torch.equal(first, next(iter(o_g2.state.values()))["exp_avg"])


def test_training_state_rejects_other_architecture(tmp_path: Path) -> None:
    small = ModelConfig(latent_dim=16, g_width=8, d_width=8)
    generator, discriminator = Generator(small), Discriminator(small)
    opt_g, opt_d = make_optimizers(generator, discriminator, TrainConfig())
    save_training_state(tmp_path / "c.pt", generator=generator, discriminator=discriminator, opt_g=opt_g, opt_d=opt_d, epoch=1)
    other = ModelConfig(latent_dim=32, g_width=8, d_width=8)
    g2, d2 = Generator(other), Discriminator(other)
    o_g2, o_d2 = make_optimizers(g2, d2, TrainConfig())
    with pytest.raises(ValueError):
        load_training_state(tmp_path / "c.pt", generator=g2, discriminator=d2, opt_g=o_g2, opt_d=o_d2)


def test_classifier_roundtrip(tmp_path: Path) -> None:
    model = MNISTClassifier().eval()
    save_classifier(tmp_path / "clf.pt", model)
    x = torch.rand(3, 1, 28, 28)
    assert torch.equal(model(x), load_classifier(tmp_path / "clf.pt")(x))


@pytest.mark.parametrize("name", ["dcgan_generator.pt", "cgan_generator.pt", "mnist_classifier.pt"])
def test_committed_weights_are_small_and_loadable(name: str) -> None:
    path = ROOT / "weights" / name
    if not path.exists():
        pytest.skip(f"{path} not present")
    assert path.stat().st_size < 2 * 1024 * 1024
    model = load_classifier(path) if name.startswith("mnist") else load_generator(path)
    assert sum(p.numel() for p in model.parameters()) > 0
