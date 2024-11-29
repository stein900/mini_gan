"""Train a DCGAN (default) or a class-conditional GAN (``--conditional``) on MNIST.

Every epoch: fixed-noise sample grid, MNIST-IS / MNIST-FID on 10k samples (if the evaluation
classifier exists), resumable checkpoint. At the end: training-progression GIF and curves.

Usage:
    python train.py                              # DCGAN, 50 epochs   -> runs/dcgan/
    python train.py --conditional                # cGAN, 50 epochs    -> runs/cgan/
    python train.py --epochs 2 --device cpu      # quick CPU run
    python train.py --conditional --publish      # also refresh weights/, assets/ and results/
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib
import torch
from PIL import Image

from minigan.checkpoint import export_generator, load_classifier, load_training_state
from minigan.data import load_mnist
from minigan.metrics import Evaluator
from minigan.models import Discriminator, Generator, ModelConfig, count_parameters
from minigan.training import History, TrainConfig, fit, make_optimizers
from minigan.utils import resolve_device, seed_everything
from minigan.viz import caption_frame, plot_training_curves, save_gif

matplotlib.use("Agg")  # headless: figures are only written to disk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--conditional", action="store_true", help="train a class-conditional GAN")
    parser.add_argument("--run-name", default=None, help="defaults to 'cgan' or 'dcgan'")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="one-sided: real target = 1 - s")
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--g-width", type=int, default=64)
    parser.add_argument("--d-width", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="'auto', 'cpu' or 'cuda'")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--classifier", type=Path, default=Path("weights/mnist_classifier.pt"))
    parser.add_argument("--eval-samples", type=int, default=10_000, help="samples per epoch for metrics (0 = off)")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--resume", action="store_true", help="continue from runs/<run-name>/checkpoint.pt")
    parser.add_argument("--publish", action="store_true", help="export to weights/, assets/ and results/")
    return parser.parse_args()


def make_progress_gif(out_dir: Path) -> None:
    """GIF of the fixed-noise grids: every epoch up to 10 (fast early changes), then every 5th."""
    paths = sorted((out_dir / "frames").glob("epoch_*.png"))
    epochs = [int(p.stem.split("_")[1]) for p in paths]
    frames = [
        caption_frame(Image.open(p), f"epoch {e}")
        for p, e in zip(paths, epochs)
        if e <= 10 or e % 5 == 0 or e == epochs[-1]
    ]
    save_gif(frames, out_dir / "progress.gif", frame_ms=300)


def main() -> None:
    args = parse_args()
    run_name = args.run_name or ("cgan" if args.conditional else "dcgan")
    out_dir = args.runs_dir / run_name
    seed_everything(args.seed)
    device = resolve_device(args.device)

    model_config = ModelConfig(
        latent_dim=args.latent_dim, n_classes=10 if args.conditional else 0, g_width=args.g_width, d_width=args.d_width
    )
    train_config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
        label_smoothing=args.label_smoothing,
        seed=args.seed,
    )
    generator, discriminator = Generator(model_config).to(device), Discriminator(model_config).to(device)
    opt_g, opt_d = make_optimizers(generator, discriminator, train_config)
    print(f"[{run_name}] device={device}  G params={count_parameters(generator):,}  D params={count_parameters(discriminator):,}")

    train_x, train_y = load_mnist(args.data_dir, train=True, device=device)
    evaluator = None
    if args.eval_samples > 0 and args.classifier.exists():
        test_x, _ = load_mnist(args.data_dir, train=False, device=device)
        evaluator = Evaluator(load_classifier(args.classifier, device), test_x)
    elif args.eval_samples > 0:
        print(f"[{run_name}] {args.classifier} not found: run train_classifier.py to enable per-epoch metrics.")

    history, start_epoch = None, 0
    if args.resume:
        payload = load_training_state(
            out_dir / "checkpoint.pt", generator=generator, discriminator=discriminator, opt_g=opt_g, opt_d=opt_d
        )
        start_epoch = payload["epoch"]
        history = History.from_dict(json.loads((out_dir / "history.json").read_text(encoding="utf-8")))
        print(f"[{run_name}] resumed after epoch {start_epoch}")

    start = time.perf_counter()
    history = fit(
        generator,
        discriminator,
        train_x,
        train_y,
        train_config,
        opt_g=opt_g,
        opt_d=opt_d,
        out_dir=out_dir,
        evaluator=evaluator,
        eval_samples=args.eval_samples,
        history=history,
        start_epoch=start_epoch,
        log_every=args.log_every,
    )
    total_time = time.perf_counter() - start
    print(f"[{run_name}] finished in {total_time / 60:.1f} min")

    steps_per_epoch = train_x.shape[0] // train_config.batch_size
    title = "cGAN" if args.conditional else "DCGAN"
    plot_training_curves(history.steps, steps_per_epoch, out_dir / "training_curves.png", title)
    make_progress_gif(out_dir)
    metadata = {
        "run": run_name,
        "train_config": asdict(train_config),
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "total_time_s": round(total_time, 1),
        "train_time_s": round(sum(e.get("train_time_s", 0.0) for e in history.epochs), 1),
    }
    export_generator(out_dir / "generator.pt", generator, half=False, metadata=metadata)

    if args.publish:
        Path("assets").mkdir(exist_ok=True)
        Path("results").mkdir(exist_ok=True)
        export_generator(Path("weights") / f"{run_name}_generator.pt", generator, half=True, metadata=metadata)
        shutil.copyfile(out_dir / "progress.gif", Path("assets") / f"{run_name}_progress.gif")
        shutil.copyfile(out_dir / "training_curves.png", Path("assets") / f"{run_name}_training_curves.png")
        summary = {**metadata, "model_config": model_config.to_dict(), "epochs": history.epochs}
        (Path("results") / f"{run_name}_history.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[{run_name}] published weights/{run_name}_generator.pt, assets/ and results/")


if __name__ == "__main__":
    main()
