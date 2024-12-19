"""Train the small MNIST classifier that powers the evaluation metrics (MNIST-IS / MNIST-FID).

Usage:
    python train_classifier.py                 # ~5 epochs, writes weights/mnist_classifier.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from minigan.checkpoint import save_classifier
from minigan.classifier import MNISTClassifier, accuracy, train_classifier
from minigan.data import load_mnist
from minigan.utils import resolve_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="'auto', 'cpu' or 'cuda'")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("weights/mnist_classifier.pt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    train_x, train_y = load_mnist(args.data_dir, train=True, device=device)
    test_x, test_y = load_mnist(args.data_dir, train=False, device=device)

    model = MNISTClassifier().to(device)
    start = time.perf_counter()
    train_classifier(
        model,
        train_x,
        train_y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        generator=torch.Generator().manual_seed(args.seed),
    )
    elapsed = time.perf_counter() - start
    test_acc = accuracy(model, test_x, test_y)
    print(f"[classifier] test accuracy {test_acc:.4f}  ({elapsed:.1f}s on {device})")
    save_classifier(
        args.out,
        model,
        metadata={"test_accuracy": test_acc, "epochs": args.epochs, "seed": args.seed, "train_time_s": round(elapsed, 1)},
    )
    print(f"[classifier] saved {args.out}")


if __name__ == "__main__":
    main()
