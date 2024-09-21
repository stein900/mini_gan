from __future__ import annotations

import pytest
import torch

from minigan.classifier import MNISTClassifier
from minigan.models import Discriminator, Generator, ModelConfig, count_parameters


def _labels(config: ModelConfig, n: int) -> torch.Tensor | None:
    return torch.arange(n) % 10 if config.conditional else None


def test_generator_output_shape_and_range(tiny_config: ModelConfig) -> None:
    generator = Generator(tiny_config)
    images = generator(torch.randn(5, tiny_config.latent_dim), _labels(tiny_config, 5))
    assert images.shape == (5, 1, 28, 28)
    assert images.min() >= -1.0 and images.max() <= 1.0


def test_discriminator_returns_one_logit_per_image(tiny_config: ModelConfig) -> None:
    discriminator = Discriminator(tiny_config)
    logits = discriminator(torch.rand(6, 1, 28, 28) * 2 - 1, _labels(tiny_config, 6))
    assert logits.shape == (6,)


def test_generator_feeds_discriminator(tiny_config: ModelConfig) -> None:
    generator, discriminator = Generator(tiny_config), Discriminator(tiny_config)
    labels = _labels(tiny_config, 4)
    assert discriminator(generator(torch.randn(4, tiny_config.latent_dim), labels), labels).shape == (4,)


def test_conditional_networks_require_labels() -> None:
    config = ModelConfig(latent_dim=16, n_classes=10, g_width=8, d_width=8)
    with pytest.raises(ValueError):
        Generator(config)(torch.randn(2, 16))
    with pytest.raises(ValueError):
        Discriminator(config)(torch.zeros(2, 1, 28, 28))


def test_conditional_generator_depends_on_label() -> None:
    generator = Generator(ModelConfig(latent_dim=16, n_classes=10, g_width=8, d_width=8)).eval()
    z = torch.randn(1, 16)
    assert not torch.allclose(generator(z, torch.tensor([3])), generator(z, torch.tensor([7])))


def test_dcgan_weight_initialisation() -> None:
    generator = Generator()
    conv = generator.net[0]
    assert abs(conv.weight.std().item() - 0.02) < 0.002
    batch_norm = generator.net[1]
    assert abs(batch_norm.weight.mean().item() - 1.0) < 0.01
    assert torch.count_nonzero(batch_norm.bias) == 0


def test_default_generator_is_compact() -> None:
    # < 1M parameters, i.e. < 2 MB once exported in float16.
    assert count_parameters(Generator()) < 1_000_000
    assert count_parameters(Generator(ModelConfig(n_classes=10))) < 1_000_000


def test_classifier_shapes() -> None:
    model = MNISTClassifier().eval()
    x = torch.zeros(4, 1, 28, 28)
    assert model.embed(x).shape == (4, MNISTClassifier.feature_dim)
    assert model(x).shape == (4, 10)
