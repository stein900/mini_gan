from __future__ import annotations

import torch

from minigan.models import Generator, ModelConfig
from minigan.sampling import balanced_labels, generate, interpolation_path, latent_noise, slerp


def test_latent_noise_is_seeded() -> None:
    assert torch.equal(latent_noise(4, 16, seed=3), latent_noise(4, 16, seed=3))
    assert not torch.equal(latent_noise(4, 16, seed=3), latent_noise(4, 16, seed=4))


def test_generate_is_deterministic_with_fixed_seed(tiny_config: ModelConfig) -> None:
    generator = Generator(tiny_config).train()
    first = generate(generator, 12, seed=42)
    torch.manual_seed(999)  # the global RNG must not influence sampling
    second = generate(generator, 12, seed=42)
    assert torch.equal(first, second)
    assert not torch.equal(first, generate(generator, 12, seed=43))
    assert generator.training, "generate() must restore the previous train/eval mode"


def test_generate_batches_consistently(tiny_config: ModelConfig) -> None:
    generator = Generator(tiny_config)
    assert torch.allclose(generate(generator, 10, seed=0, batch_size=3), generate(generator, 10, seed=0), atol=1e-6)


def test_balanced_labels() -> None:
    assert torch.bincount(balanced_labels(100, 10)).tolist() == [10] * 10


def test_slerp_endpoints_and_norm() -> None:
    z0, z1 = latent_noise(2, 100, seed=0)
    path = slerp(z0, z1, torch.linspace(0, 1, 11))
    assert torch.allclose(path[0], z0, atol=1e-5) and torch.allclose(path[-1], z1, atol=1e-5)
    # Gaussian latents of the same norm stay on that sphere; linear interpolation would shrink mid-way.
    z1_scaled = z1 * z0.norm() / z1.norm()
    norms = slerp(z0, z1_scaled, torch.linspace(0, 1, 11)).norm(dim=1)
    assert torch.allclose(norms, torch.full_like(norms, z0.norm().item()), rtol=1e-4)


def test_interpolation_path_lengths() -> None:
    keys = latent_noise(4, 8, seed=0)
    assert interpolation_path(keys, steps_per_segment=5, closed=True).shape == (20, 8)
    open_path = interpolation_path(keys, steps_per_segment=5, closed=False)
    assert open_path.shape == (16, 8)
    assert torch.allclose(open_path[-1], keys[-1])
