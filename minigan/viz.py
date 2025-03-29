"""Figures: image grids, GIFs and training / evaluation plots.

Plots use matplotlib's object-oriented API (``Figure``) rather than ``pyplot``, so they render
headless without any backend configuration.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from PIL import Image, ImageDraw, ImageFont
from torchvision.utils import make_grid

# Colour-blind-checked categorical slots (blue, orange, aqua) and recessive chart chrome.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
INK, INK_2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


# --------------------------------------------------------------------------- image grids & GIFs
def images_to_array(images: torch.Tensor, nrow: int, padding: int = 2) -> np.ndarray:
    """Tile images in [-1, 1] of shape ``(N, 1, H, W)`` into a uint8 array (white gutters)."""
    unit = images.detach().float().cpu().add(1.0).div(2.0).clamp(0.0, 1.0)
    grid = make_grid(unit, nrow=nrow, padding=padding, pad_value=1.0)
    return (grid[0].numpy() * 255.0).round().astype(np.uint8)


def grid_image(images: torch.Tensor, nrow: int, scale: int = 2) -> Image.Image:
    """Image grid as a grayscale PIL image, upscaled with nearest-neighbour interpolation."""
    image = Image.fromarray(images_to_array(images, nrow))
    return image.resize((image.width * scale, image.height * scale), Image.NEAREST)


def save_grid(images: torch.Tensor, path: str | Path, nrow: int, scale: int = 2) -> None:
    """Save an image grid as PNG (see :func:`grid_image`)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    grid_image(images, nrow, scale).save(path)


def caption_frame(image: Image.Image, text: str, band: int = 30) -> Image.Image:
    """Return a copy of a grayscale ``image`` with a white caption band on top."""
    out = Image.new("L", (image.width, image.height + band), 255)
    out.paste(image.convert("L"), (0, band))
    ImageDraw.Draw(out).text((8, band // 2), text, fill=11, font=ImageFont.load_default(size=17), anchor="lm")
    return out


def save_gif(frames: Sequence[Image.Image], path: str | Path, frame_ms: int = 220, hold_last_ms: int = 1800) -> None:
    """Write an infinitely looping GIF, holding the last frame longer."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    durations = [frame_ms] * (len(frames) - 1) + [hold_last_ms]
    frames[0].save(path, save_all=True, append_images=list(frames[1:]), duration=durations, loop=0, optimize=True)


# --------------------------------------------------------------------------- plot helpers
def _figure(ncols: int = 1, figsize: tuple[float, float] = (7.2, 3.4)) -> tuple[Figure, list[Axes]]:
    fig = Figure(figsize=figsize, dpi=150, facecolor="white", layout="constrained")
    axes = list(fig.subplots(1, ncols, squeeze=False)[0])
    for ax in axes:
        ax.set_facecolor("white")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=AXIS, labelcolor=INK_2, labelsize=8.5)
    return fig, axes


def _labels(ax: Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, fontweight="semibold")
    ax.set_xlabel(xlabel, fontsize=9, color=INK_2)
    ax.set_ylabel(ylabel, fontsize=9, color=INK_2)


def _legend(ax: Axes, **kwargs: object) -> None:
    legend = ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, **kwargs)
    for handle in legend.legend_handles:
        handle.set_alpha(1.0)


def _reference(ax: Axes, y: float, text: str) -> None:
    """Thin muted horizontal reference line, labelled just above its left end."""
    ax.axhline(y, color=MUTED, linewidth=1.0, zorder=1)
    ax.annotate(text, (0.01, y), xycoords=("axes fraction", "data"), xytext=(0, 3),
                textcoords="offset points", ha="left", va="bottom", fontsize=8, color=MUTED)


def moving_average(values: Sequence[float], window: int) -> np.ndarray:
    """Trailing moving average (length ``len(values) - window + 1``)."""
    values = np.asarray(values, dtype=np.float64)
    window = max(1, min(window, len(values)))
    return np.convolve(values, np.ones(window) / window, mode="valid")


def _save(fig: Figure, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")


# --------------------------------------------------------------------------- training curves
def plot_training_curves(
    steps: Mapping[str, Sequence[float]], steps_per_epoch: int, path: str | Path, title: str, window: int = 200
) -> None:
    """Two panels: D/G losses and D(x) / D(G(z)) per step (raw, faint) with a moving average.

    Muted horizontal lines mark the values reached when D outputs 1/2 everywhere (the
    theoretical equilibrium): L_D = 2 ln 2, L_G = ln 2, D(x) = D(G(z)) = 0.5.
    """
    fig, (ax_loss, ax_prob) = _figure(ncols=2, figsize=(9.0, 3.9))
    n = len(steps["loss_d"])
    x = np.arange(1, n + 1) / steps_per_epoch
    x_smooth = x[min(window, n) - 1 :]
    panels = (
        (ax_loss, (("loss_d", r"$\mathcal{L}_D$ (discriminator)"), ("loss_g", r"$\mathcal{L}_G$ (generator)"))),
        (ax_prob, (("d_real", r"$D(x)$ on real batch"), ("d_fake", r"$D(G(z))$ on fake batch"))),
    )
    for ax, series in panels:
        for color, (key, label) in zip(SERIES, series):
            ax.plot(x, steps[key], color=color, linewidth=0.4, alpha=0.10)
            ax.plot(x_smooth, moving_average(steps[key], window), color=color, linewidth=2.0, label=label)
    for y in (2 * np.log(2), np.log(2)):
        ax_loss.axhline(y, color=MUTED, linewidth=1.0, zorder=1)
    ax_loss.plot([], [], color=MUTED, linewidth=1.0, label=r"$2\ln 2$ and $\ln 2$ (value at $D\equiv\frac{1}{2}$)")
    ax_prob.axhline(0.5, color=MUTED, linewidth=1.0, zorder=1, label=r"0.5 ($D\equiv\frac{1}{2}$)")
    raw = np.concatenate([np.asarray(steps["loss_d"]), np.asarray(steps["loss_g"])])
    ax_loss.set_ylim(0.0, float(np.percentile(raw, 99.5)) * 1.1)
    ax_prob.set_ylim(0.0, 1.0)
    for ax in (ax_loss, ax_prob):
        _legend(ax, loc="upper center", ncols=2, bbox_to_anchor=(0.5, -0.2))
    _labels(ax_loss, f"{title}: losses", "epoch", "BCE loss")
    _labels(ax_prob, f"{title}: discriminator outputs", "epoch", "mean probability")
    _save(fig, path)


def plot_metric_curves(
    runs: Mapping[str, Sequence[Mapping[str, float]]], references: Mapping[str, float], path: str | Path
) -> None:
    """MNIST-FID (log scale) and MNIST-IS per epoch for several runs, with real-data references."""
    fig, (ax_fid, ax_is) = _figure(ncols=2, figsize=(9.0, 3.4))
    for color, (name, epochs) in zip(SERIES, runs.items()):
        ep = [e["epoch"] for e in epochs]
        for ax, key in ((ax_fid, "mnist_fid"), (ax_is, "mnist_is")):
            values = [e[key] for e in epochs]
            ax.plot(ep, values, color=color, linewidth=2.0, label=name, solid_capstyle="round")
            ax.plot(ep[-1], values[-1], "o", color=color, markersize=6, markeredgecolor="white", markeredgewidth=1.5)
            ax.annotate(f"{values[-1]:.2f}", (ep[-1], values[-1]), xytext=(5, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK_2,
                        bbox={"boxstyle": "square,pad=0.15", "facecolor": "white", "edgecolor": "none"})
    ax_fid.set_yscale("log")
    _reference(ax_fid, references["mnist_fid"], f"real train vs test: {references['mnist_fid']:.2f}")
    _reference(ax_is, references["mnist_is"], f"real test images: {references['mnist_is']:.2f}")
    _labels(ax_fid, "MNIST-FID (lower is better)", "epoch", "Frechet distance (log scale)")
    _labels(ax_is, "MNIST-IS (higher is better, max 10)", "epoch", "classifier Inception Score")
    _legend(ax_fid, loc="upper right")
    _legend(ax_is, loc="lower right")
    _save(fig, path)


# --------------------------------------------------------------------------- evaluation figures
def plot_nearest_neighbours(
    queries: torch.Tensor, neighbours: torch.Tensor, distances: torch.Tensor, path: str | Path
) -> None:
    """Columns = generated samples; first row = sample, next rows = its k nearest training images."""
    n, k = distances.shape
    fig = Figure(figsize=(1.0 * n + 0.9, 1.05 * (k + 1)), dpi=150, facecolor="white", layout="constrained")
    axes = fig.subplots(k + 1, n, squeeze=False)
    for col in range(n):
        tiles = [queries[col]] + [neighbours[col, j] for j in range(k)]
        for row, tile in enumerate(tiles):
            ax = axes[row, col]
            ax.imshow(tile.squeeze().cpu().numpy(), cmap="gray", vmin=-1.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row > 0:
                ax.set_title(f"d = {distances[col, row - 1]:.1f}", fontsize=7.5, color=INK_2, pad=2)
            if col == 0:
                ax.set_ylabel("generated" if row == 0 else f"train NN {row}", fontsize=8, color=INK_2)
    _save(fig, path)


def plot_distance_histograms(samples: Mapping[str, np.ndarray], path: str | Path) -> None:
    """Overlaid histograms of nearest-training-neighbour distances (memorisation check)."""
    fig, (ax,) = _figure(figsize=(7.2, 3.4))
    stacked = np.concatenate(list(samples.values()))
    bins = np.linspace(0.0, np.percentile(stacked, 99.5), 60)
    for color, (name, values) in zip(SERIES, samples.items()):
        ax.hist(values, bins=bins, density=True, histtype="stepfilled", color=color, alpha=0.10)
        ax.hist(values, bins=bins, density=True, histtype="step", color=color, linewidth=2.0,
                label=f"{name} (median {np.median(values):.2f})")
    _labels(ax, "Distance to the nearest training image (pixel L2, images in [-1, 1])", "L2 distance", "density")
    _legend(ax, loc="upper center", ncols=len(samples), bbox_to_anchor=(0.5, -0.22))
    _save(fig, path)


def plot_class_frequencies(frequencies: Mapping[str, Sequence[float]], path: str | Path) -> None:
    """Grouped columns: share of samples assigned to each digit by the classifier."""
    fig, (ax,) = _figure(figsize=(7.2, 3.0))
    digits = np.arange(10)
    width = 0.8 / len(frequencies)
    for i, (color, (name, freq)) in enumerate(zip(SERIES, frequencies.items())):
        offset = (i - (len(frequencies) - 1) / 2) * width
        ax.bar(digits + offset, np.asarray(freq) * 100.0, width=width * 0.85, color=color, label=name)
    ax.axhline(10.0, color=MUTED, linewidth=1.0, zorder=1, label="uniform (10%)")
    ax.set_xticks(digits)
    ax.grid(False, axis="x")
    _labels(ax, "Predicted class distribution of 10,000 samples", "digit (classifier prediction)", "share of samples (%)")
    _legend(ax, loc="upper center", ncols=len(frequencies) + 1, bbox_to_anchor=(0.5, -0.22))
    _save(fig, path)
