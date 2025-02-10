"""Final evaluation of the exported generators in ``weights/``.

Computes, for each generator, on 10,000 samples:
  * MNIST-IS (mean +- std over 10 splits) and MNIST-FID against the 10,000 MNIST test images;
  * label accuracy for the conditional model (does G draw the digit it was asked for?);
  * the predicted class distribution (mode coverage);
  * nearest-training-neighbour distances (memorisation check), compared with test -> train;
  * the spread of MNIST-FID over independent sample sets (evaluation noise, ``--resamples``).
Real-data references (train subset vs test) give the floor of each metric.

Optionally (``--seed-study``), the same protocol is applied to several training runs
(``runs/<run>/generator.pt``) to report the mean and standard deviation across training seeds.

Usage:
    python evaluate.py                     # writes results/metrics.json and figures in assets/
    python evaluate.py --seed-study "DCGAN=runs/dcgan,runs/dcgan_seed1,runs/dcgan_seed2"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch

from minigan.checkpoint import load_classifier, load_generator, read_metadata
from minigan.classifier import accuracy
from minigan.data import load_mnist
from minigan.metrics import Evaluator, classifier_inception_score, nearest_neighbours
from minigan.sampling import balanced_labels, generate
from minigan.utils import resolve_device, seed_everything
from minigan.viz import plot_class_frequencies, plot_distance_histograms, plot_metric_curves, plot_nearest_neighbours

matplotlib.use("Agg")  # headless: figures are only written to disk

DISPLAY_NAMES = {"dcgan": "DCGAN", "cgan": "cGAN"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=["dcgan", "cgan"])
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"))
    parser.add_argument("--classifier", type=Path, default=Path("weights/mnist_classifier.pt"))
    parser.add_argument("--n-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=7, help="sampling seed (distinct from the training eval seed)")
    parser.add_argument("--nn-queries", type=int, default=8, help="samples shown in the nearest-neighbour figure")
    parser.add_argument("--resamples", type=int, default=5, help="sample sets used to measure MNIST-FID noise")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--assets-dir", type=Path, default=Path("assets"))
    parser.add_argument(
        "--seed-study",
        action="append",
        default=[],
        metavar="GROUP=RUN_DIR[,RUN_DIR...]",
        help="evaluate runs/<run>/generator.pt of several training seeds as one group (repeatable)",
    )
    return parser.parse_args()


def _round(scores: dict[str, float], digits: int = 4) -> dict[str, float]:
    return {k: round(v, digits) for k, v in scores.items()}


def seed_study(evaluator: Evaluator, spec: str, n: int, seed: int, device: torch.device) -> tuple[str, dict[str, Any]]:
    """Score every run of a ``GROUP=dir1,dir2,...`` spec and aggregate mean / std per metric."""
    group, _, dirs = spec.partition("=")
    per_run = {}
    for run_dir in (Path(d) for d in dirs.split(",")):
        generator = load_generator(run_dir / "generator.pt", device)
        per_run[run_dir.name] = evaluator.score_generator(generator, n, seed=seed)
    summary: dict[str, Any] = {"runs": list(per_run)}
    for key in ("mnist_fid", "mnist_is", "label_accuracy"):
        values = [scores[key] for scores in per_run.values() if key in scores]
        if values:
            summary[key] = {
                "values": [round(v, 4) for v in values],
                "mean": round(float(np.mean(values)), 4),
                "std": round(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0, 4),
            }
    print(f"[evaluate] seed study {group}: " + "  ".join(f"{k}={v['mean']}+-{v['std']}" for k, v in summary.items() if k != "runs"))
    return group, summary


def memorisation_stats(gen_dist: np.ndarray, test_dist: np.ndarray) -> dict[str, float]:
    """Compare nearest-train distances of generated samples with those of held-out test images."""
    threshold = float(np.percentile(test_dist, 1))
    return {
        "nn_l2_median": float(np.median(gen_dist)),
        "nn_l2_min": float(gen_dist.min()),
        "share_closer_than_test_p1": float((gen_dist < threshold).mean()),
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    train_x, _ = load_mnist(args.data_dir, train=True, device=device)
    test_x, test_y = load_mnist(args.data_dir, train=False, device=device)
    classifier = load_classifier(args.classifier, device)
    evaluator = Evaluator(classifier, test_x)
    n = args.n_samples

    # ---- real-data references (best achievable values for this feature extractor and n)
    subset = torch.randperm(train_x.shape[0], generator=torch.Generator().manual_seed(args.seed))[:n]
    real_train = evaluator.score_images(train_x[subset.to(device)])
    test_is, test_is_std = classifier_inception_score(evaluator.embed(test_x)[1])
    test_nn = nearest_neighbours(test_x, train_x)[0][:, 0].numpy()
    test_freq = (torch.bincount(test_y, minlength=10).float() / test_y.numel()).tolist()
    results: dict[str, Any] = {
        "protocol": {
            "n_generated": n,
            "reference_set": "MNIST test (10,000 images)",
            "feature_extractor": "repo MNIST classifier, 128-d penultimate features (NOT Inception-v3)",
            "is_splits": 10,
            "sampling_seed": args.seed,
            "conditional_labels": "balanced (n/10 per digit)",
            "nn_distance": "Euclidean, pixel space, images scaled to [-1, 1] (784-d)",
        },
        "classifier": {**read_metadata(args.classifier), "test_accuracy": round(accuracy(classifier, test_x, test_y), 4)},
        "real_data": {
            "train_subset_vs_test": _round(real_train),
            "test_mnist_is": round(test_is, 4),
            "test_mnist_is_std": round(test_is_std, 4),
            "test_to_train_nn_l2_median": round(float(np.median(test_nn)), 4),
            "test_to_train_nn_l2_p1": round(float(np.percentile(test_nn, 1)), 4),
            "test_class_frequencies": [round(f, 4) for f in test_freq],
        },
        "models": {},
    }
    frequencies = {"real test set": test_freq}
    distances = {"real test → train": test_nn}
    curves: dict[str, list[dict[str, float]]] = {}

    # ---- generators
    for name in args.models:
        path = args.weights_dir / f"{name}_generator.pt"
        if not path.exists():
            print(f"[evaluate] skipping {name}: {path} not found")
            continue
        label = DISPLAY_NAMES.get(name, name)
        generator = load_generator(path, device)
        samples = generate(generator, n, seed=args.seed)
        labels = balanced_labels(n, generator.config.n_classes) if generator.config.conditional else None
        scores = evaluator.score_images(samples, labels)
        predicted = torch.from_numpy(evaluator.embed(samples)[1].argmax(axis=1))
        gen_nn, gen_nn_idx = nearest_neighbours(samples, train_x, k=3)
        fids = [scores["mnist_fid"]] + [
            evaluator.score_generator(generator, n, seed=args.seed + k)["mnist_fid"] for k in range(1, args.resamples)
        ]
        entry: dict[str, Any] = {
            **_round(scores),
            "mnist_fid_resampled": {
                "sampling_seeds": [args.seed + k for k in range(args.resamples)],
                "values": [round(v, 4) for v in fids],
                "mean": round(float(np.mean(fids)), 4),
                "std": round(float(np.std(fids, ddof=1)) if len(fids) > 1 else 0.0, 4),
            },
            "class_frequencies": [round(f, 4) for f in (torch.bincount(predicted, minlength=10).float() / n).tolist()],
            **_round(memorisation_stats(gen_nn[:, 0].numpy(), test_nn)),
            "weights_file": path.as_posix(),
            "weights_size_bytes": path.stat().st_size,
            "training": read_metadata(path),
        }
        history_path = args.results_dir / f"{name}_history.json"
        if history_path.exists():
            epochs = json.loads(history_path.read_text(encoding="utf-8"))["epochs"]
            curves[label] = [e for e in epochs if "mnist_fid" in e]
            entry["per_epoch"] = [
                {k: round(e[k], 4) for k in ("epoch", "mnist_fid", "mnist_is", "label_accuracy") if k in e}
                for e in curves[label]
            ]
        results["models"][name] = entry
        frequencies[label] = entry["class_frequencies"]
        distances[f"{label} → train"] = gen_nn[:, 0].numpy()

        q = args.nn_queries
        plot_nearest_neighbours(
            samples[:q].cpu(), train_x[gen_nn_idx[:q].to(device)].cpu(), gen_nn[:q], args.assets_dir / f"{name}_nearest_neighbours.png"
        )
        print(f"[evaluate] {label}: " + "  ".join(f"{k}={v}" for k, v in _round(scores, 3).items()))

    if args.seed_study:
        results["seed_study"] = dict(seed_study(evaluator, spec, n, args.seed, device) for spec in args.seed_study)

    # ---- figures and JSON
    if curves:
        references = {"mnist_fid": real_train["mnist_fid"], "mnist_is": test_is}
        plot_metric_curves(curves, references, args.assets_dir / "metrics_vs_epoch.png")
    plot_class_frequencies(frequencies, args.assets_dir / "class_frequencies.png")
    plot_distance_histograms(distances, args.assets_dir / "nn_distances.png")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    out = args.results_dir / "metrics.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[evaluate] real train-vs-test MNIST-FID={real_train['mnist_fid']:.3f}  test MNIST-IS={test_is:.3f}")
    print(f"[evaluate] wrote {out}")


if __name__ == "__main__":
    main()
