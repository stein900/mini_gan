"""Sample from the pretrained generators committed in ``weights/`` (no training needed).

Outputs (in ``--out-dir``, default ``assets/``) for each available model:
  <model>_samples.png          8x8 grid of random samples
  <model>_interpolation.gif    looping spherical interpolations in latent space
  <model>_interpolation.png    static interpolation strips
  cgan_class_grid.png          conditional model: one style (z) per row, digits 0-9 per column
  cgan_digit_<d>.png           conditional model, with --digit d: a grid of the requested digit

Usage:
    python demo.py                      # everything, for every model found in weights/
    python demo.py --digit 7            # also draw a grid of 7s with the conditional model
    python demo.py --model dcgan --seed 3 --out-dir samples
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from minigan.checkpoint import load_generator
from minigan.models import Generator
from minigan.sampling import decode, interpolation_path, latent_noise
from minigan.utils import resolve_device
from minigan.viz import grid_image, save_gif, save_grid

MODELS = ("dcgan", "cgan")
DIGITS = torch.arange(10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=("all", *MODELS), default="all")
    parser.add_argument("--digit", type=int, choices=range(10), default=None, help="digit for the conditional model")
    parser.add_argument("--n", type=int, default=64, help="number of samples in the sample / digit grids")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=12, help="interpolation frames between two keypoints")
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"))
    parser.add_argument("--out-dir", type=Path, default=Path("assets"))
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def interpolation_gif(generator: Generator, name: str, out_dir: Path, seed: int, steps: int) -> None:
    """DCGAN: 16 independent latent loops (4x4). cGAN: 3 latent loops, each rendered for digits 0-9."""
    conditional = generator.config.conditional
    n_paths, n_keys = (3, 5) if conditional else (16, 5)
    keys = latent_noise(n_paths * n_keys, generator.latent_dim, seed).view(n_paths, n_keys, -1)
    paths = torch.stack([interpolation_path(k, steps) for k in keys])  # (paths, T, D)
    frames = []
    for t in range(paths.shape[1]):
        z = paths[:, t]
        if conditional:
            images = decode(generator, z.repeat_interleave(10, dim=0), DIGITS.repeat(n_paths))
            frames.append(grid_image(images, nrow=10, scale=2))
        else:
            frames.append(grid_image(decode(generator, z), nrow=4, scale=3))
    save_gif(frames, out_dir / f"{name}_interpolation.gif", frame_ms=80, hold_last_ms=80)


def interpolation_strips(generator: Generator, name: str, out_dir: Path, seed: int) -> None:
    """Rows = slerp from z_a to z_b in 10 steps. cGAN: the same (z_a, z_b) for every digit row."""
    conditional = generator.config.conditional
    n_rows = 10 if conditional else 8
    n_pairs = 1 if conditional else n_rows
    ends = latent_noise(2 * n_pairs, generator.latent_dim, seed + 1).view(n_pairs, 2, -1)
    rows = [interpolation_path(pair, steps_per_segment=9, closed=False) for pair in ends]  # 10 points each
    z = torch.cat(rows * n_rows if conditional else rows)
    labels = DIGITS.repeat_interleave(10) if conditional else None
    save_grid(decode(generator, z, labels), out_dir / f"{name}_interpolation.png", nrow=10)


def class_grid(generator: Generator, out_dir: Path, seed: int, n_styles: int = 8) -> None:
    z = latent_noise(n_styles, generator.latent_dim, seed + 2).repeat_interleave(10, dim=0)
    save_grid(decode(generator, z, DIGITS.repeat(n_styles)), out_dir / "cgan_class_grid.png", nrow=10)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    names = MODELS if args.model == "all" else (args.model,)
    if args.digit is not None and "cgan" not in names:
        raise SystemExit("--digit needs the conditional model (use --model cgan or --model all).")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for name in names:
        path = args.weights_dir / f"{name}_generator.pt"
        if not path.exists():
            print(f"[demo] {path} not found, skipping {name}")
            continue
        generator = load_generator(path, device)
        z = latent_noise(args.n, generator.latent_dim, args.seed)
        save_grid(decode(generator, z), args.out_dir / f"{name}_samples.png", nrow=8)
        interpolation_gif(generator, name, args.out_dir, args.seed, args.steps)
        interpolation_strips(generator, name, args.out_dir, args.seed)
        if generator.config.conditional:
            class_grid(generator, args.out_dir, args.seed)
            if args.digit is not None:
                labels = torch.full((args.n,), args.digit)
                save_grid(decode(generator, z, labels), args.out_dir / f"cgan_digit_{args.digit}.png", nrow=8)
        print(f"[demo] {name}: wrote samples, interpolations{' and class grid' if generator.config.conditional else ''}")
    print(f"[demo] outputs in {args.out_dir}/")


if __name__ == "__main__":
    main()
