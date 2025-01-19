"""Checkpoint I/O.

* **Training state** (Generator + Discriminator + both optimisers + history): large, resumable,
  written under ``runs/`` and git-ignored.
* **Compact generator weights**: inference-only, optionally stored in float16 (~1.4 MB),
  committed under ``weights/`` so the demo works without training.
* **Classifier weights**: the evaluation network, also committed (~0.6 MB).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from minigan.classifier import MNISTClassifier
from minigan.models import Discriminator, Generator, ModelConfig

GENERATOR_FORMAT = "minigan-generator-v1"


def _load(path: str | Path) -> dict[str, Any]:
    return torch.load(Path(path), map_location="cpu", weights_only=True)


def save_training_state(
    path: str | Path,
    *,
    generator: Generator,
    discriminator: Discriminator,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    epoch: int,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save everything needed to resume training after ``epoch``."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_config": generator.config.to_dict(),
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "opt_g": opt_g.state_dict(),
            "opt_d": opt_d.state_dict(),
            "epoch": epoch,
            "extra": extra or {},
        },
        path,
    )


def load_training_state(
    path: str | Path,
    *,
    generator: Generator,
    discriminator: Discriminator,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
) -> dict[str, Any]:
    """Restore networks and optimisers in place; returns the payload (``epoch``, ``extra``)."""
    payload = _load(path)
    if payload["model_config"] != generator.config.to_dict():
        raise ValueError(f"Architecture mismatch: checkpoint has {payload['model_config']}.")
    generator.load_state_dict(payload["generator"])
    discriminator.load_state_dict(payload["discriminator"])
    opt_g.load_state_dict(payload["opt_g"])
    opt_d.load_state_dict(payload["opt_d"])
    return payload


def export_generator(
    path: str | Path, generator: Generator, *, half: bool = True, metadata: dict[str, Any] | None = None
) -> None:
    """Write inference-only generator weights (+ architecture) to ``path``."""
    state = {
        k: (v.half() if half and v.is_floating_point() else v).detach().cpu()
        for k, v in generator.state_dict().items()
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"format": GENERATOR_FORMAT, "config": generator.config.to_dict(), "state_dict": state, "metadata": metadata or {}},
        path,
    )


def load_generator(path: str | Path, device: torch.device | str = "cpu") -> Generator:
    """Rebuild a :class:`Generator` from :func:`export_generator` output (weights cast to float32)."""
    payload = _load(path)
    if payload.get("format") != GENERATOR_FORMAT:
        raise ValueError(f"{path} is not a Mini-GAN generator file.")
    generator = Generator(ModelConfig(**payload["config"]))
    generator.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in payload["state_dict"].items()})
    return generator.to(device).eval()


def read_metadata(path: str | Path) -> dict[str, Any]:
    """Return the metadata stored with exported generator or classifier weights."""
    return _load(path)["metadata"]


def save_classifier(path: str | Path, model: MNISTClassifier, metadata: dict[str, Any] | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata or {}}, path)


def load_classifier(path: str | Path, device: torch.device | str = "cpu") -> MNISTClassifier:
    model = MNISTClassifier()
    model.load_state_dict(_load(path)["state_dict"])
    return model.to(device).eval()
