"""DCGAN-style Generator and Discriminator for 28x28 images, optionally class-conditional.

With ``n_classes == 0`` the networks form an unconditional DCGAN. With ``n_classes > 0``
they form a conditional GAN (cGAN, Mirza & Osindero 2014):

* the Generator concatenates a learned label embedding to the latent vector ``z``;
* the Discriminator receives the label as an extra 28x28 input channel (a learned map per class).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

IMAGE_SIZE = 28


@dataclass(frozen=True)
class ModelConfig:
    """Architecture hyper-parameters shared by both networks (serialised with checkpoints)."""

    latent_dim: int = 100
    n_classes: int = 0
    label_dim: int = 10
    g_width: int = 64
    d_width: int = 64

    @property
    def conditional(self) -> bool:
        return self.n_classes > 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def init_weights(module: nn.Module) -> None:
    """DCGAN initialisation: conv weights ~ N(0, 0.02), BatchNorm gamma ~ N(1, 0.02), biases = 0."""
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)


class Generator(nn.Module):
    """Map ``z`` (and an optional class label) to a 1x28x28 image in [-1, 1].

    Spatial path: 1x1 -> 3x3 -> 7x7 -> 14x14 -> 28x28 through transposed convolutions,
    BatchNorm + ReLU in hidden layers and ``tanh`` at the output.
    """

    def __init__(self, config: ModelConfig = ModelConfig()) -> None:
        super().__init__()
        self.config = config
        w = config.g_width
        in_dim = config.latent_dim
        self.label_embedding: nn.Embedding | None = None
        if config.conditional:
            self.label_embedding = nn.Embedding(config.n_classes, config.label_dim)
            in_dim += config.label_dim
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_dim, w * 4, kernel_size=3, stride=1, padding=0, bias=False),  # 3x3
            nn.BatchNorm2d(w * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(w * 4, w * 2, kernel_size=3, stride=2, padding=0, bias=False),  # 7x7
            nn.BatchNorm2d(w * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(w * 2, w, kernel_size=4, stride=2, padding=1, bias=False),  # 14x14
            nn.BatchNorm2d(w),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(w, 1, kernel_size=4, stride=2, padding=1),  # 28x28
            nn.Tanh(),
        )
        self.apply(init_weights)

    @property
    def latent_dim(self) -> int:
        return self.config.latent_dim

    def forward(self, z: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        if self.label_embedding is not None:
            if labels is None:
                raise ValueError("A conditional Generator requires class labels.")
            z = torch.cat([z, self.label_embedding(labels)], dim=1)
        return self.net(z[:, :, None, None])


class Discriminator(nn.Module):
    """Map a 1x28x28 image (and an optional class label) to a single real/fake logit.

    Spatial path: 28 -> 14 -> 7 -> 4 -> 1 with strided convolutions, LeakyReLU(0.2) and
    BatchNorm on every hidden layer except the first (DCGAN guidelines). No sigmoid:
    the logit feeds ``BCEWithLogitsLoss`` for numerical stability.
    """

    def __init__(self, config: ModelConfig = ModelConfig()) -> None:
        super().__init__()
        self.config = config
        w = config.d_width
        in_channels = 1
        self.label_map: nn.Embedding | None = None
        if config.conditional:
            self.label_map = nn.Embedding(config.n_classes, IMAGE_SIZE * IMAGE_SIZE)
            in_channels += 1
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, w, kernel_size=4, stride=2, padding=1),  # 14x14
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(w, w * 2, kernel_size=4, stride=2, padding=1, bias=False),  # 7x7
            nn.BatchNorm2d(w * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(w * 2, w * 4, kernel_size=3, stride=2, padding=1, bias=False),  # 4x4
            nn.BatchNorm2d(w * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(w * 4, 1, kernel_size=4, stride=1, padding=0),  # 1x1 logit
        )
        self.apply(init_weights)

    def forward(self, images: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        if self.label_map is not None:
            if labels is None:
                raise ValueError("A conditional Discriminator requires class labels.")
            label_plane = self.label_map(labels).view(-1, 1, IMAGE_SIZE, IMAGE_SIZE)
            images = torch.cat([images, label_plane], dim=1)
        return self.net(images).view(-1)


def count_parameters(module: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
