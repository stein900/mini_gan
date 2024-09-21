from __future__ import annotations

import pytest
import torch

from minigan.models import ModelConfig


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(0)


@pytest.fixture(params=[0, 10], ids=["dcgan", "cgan"])
def tiny_config(request: pytest.FixtureRequest) -> ModelConfig:
    """Narrow networks (fast on CPU) in unconditional and conditional flavours."""
    return ModelConfig(latent_dim=16, n_classes=request.param, g_width=8, d_width=8)
