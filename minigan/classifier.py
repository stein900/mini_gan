"""Small MNIST CNN used as the *feature extractor* for the evaluation metrics.

Its softmax outputs drive the classifier-based Inception Score and its 128-d penultimate
(global-average-pooled) features drive the MNIST-FID. It plays the role Inception-v3 plays
in the standard IS/FID, but trained on MNIST, so scores are only comparable within this repo.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from minigan.data import iterate_minibatches


def _conv_block(c_in: int, c_out: int) -> list[nn.Module]:
    return [nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]


class MNISTClassifier(nn.Module):
    """Conv(32)x2 -> pool -> Conv(64)x2 -> pool -> Conv(128) -> GAP -> Linear(10)."""

    feature_dim = 128
    n_classes = 10

    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            *_conv_block(1, 32),
            *_conv_block(32, 32),
            nn.MaxPool2d(2),  # 14x14
            *_conv_block(32, 64),
            *_conv_block(64, 64),
            nn.MaxPool2d(2),  # 7x7
            *_conv_block(64, self.feature_dim),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(self.feature_dim, self.n_classes))

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        """Penultimate features of shape ``(N, 128)`` (used for the Frechet distance)."""
        return self.backbone(images)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(images))


def train_classifier(
    model: MNISTClassifier,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    epochs: int = 5,
    batch_size: int = 128,
    lr: float = 1e-3,
    generator: torch.Generator | None = None,
) -> list[float]:
    """Train with Adam + cross-entropy; returns the mean training loss of each epoch."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history: list[float] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total, n_batches = torch.zeros((), device=images.device), 0
        for x, y in iterate_minibatches(images, labels, batch_size, generator=generator):
            loss = F.cross_entropy(model(x), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.detach()
            n_batches += 1
        scheduler.step()
        history.append(total.item() / n_batches)
        print(f"[classifier] epoch {epoch}/{epochs}  train loss {history[-1]:.4f}")
    return history


@torch.no_grad()
def accuracy(model: nn.Module, images: torch.Tensor, labels: torch.Tensor, batch_size: int = 1000) -> float:
    """Top-1 accuracy of ``model`` on ``(images, labels)`` in eval mode."""
    model.eval()
    correct = 0
    for start in range(0, images.shape[0], batch_size):
        logits = model(images[start : start + batch_size])
        correct += (logits.argmax(1) == labels[start : start + batch_size]).sum().item()
    return correct / images.shape[0]
