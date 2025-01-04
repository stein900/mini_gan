"""Classifier-based GAN metrics on MNIST: an Inception-Score-like score and an "MNIST-FID".

Both metrics replace Inception-v3 (ImageNet) by the small MNIST classifier of
:mod:`minigan.classifier`. They follow the IS / FID formulas exactly but are **not** the
standard Inception-v3 FID/IS and must not be compared with numbers reported in papers.
"""

from __future__ import annotations

import numpy as np
import torch

from minigan.classifier import MNISTClassifier
from minigan.models import Generator
from minigan.sampling import balanced_labels, generate


def feature_statistics(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean vector and (unbiased) covariance matrix of row-wise features, in float64."""
    features = np.asarray(features, dtype=np.float64)
    return features.mean(axis=0), np.cov(features, rowvar=False)


def _sqrt_psd(matrix: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(matrix)
    return (eigvecs * np.sqrt(np.clip(eigvals, 0.0, None))) @ eigvecs.T


def frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray) -> float:
    r"""Squared 2-Wasserstein (Frechet) distance between N(mu1, sigma1) and N(mu2, sigma2).

    .. math:: \lVert\mu_1-\mu_2\rVert^2 + \mathrm{Tr}\big(\Sigma_1 + \Sigma_2 - 2(\Sigma_1\Sigma_2)^{1/2}\big)

    ``Tr((S1 S2)^{1/2})`` is computed as ``sum(sqrt(eig(S1^{1/2} S2 S1^{1/2})))``: the matrix is
    symmetric PSD, so only symmetric eigendecompositions are needed (no complex round-off).
    """
    sqrt_s1 = _sqrt_psd(sigma1)
    middle = sqrt_s1 @ sigma2 @ sqrt_s1
    tr_covmean = np.sqrt(np.clip(np.linalg.eigvalsh((middle + middle.T) / 2.0), 0.0, None)).sum()
    diff = mu1 - mu2
    return float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2.0 * tr_covmean)


def classifier_inception_score(probs: np.ndarray, splits: int = 10, eps: float = 1e-12) -> tuple[float, float]:
    r"""``exp(E_x KL(p(y|x) || p(y)))`` computed on ``splits`` folds; returns (mean, std).

    It is maximal (= number of classes) when every sample is classified confidently and
    the classes are used uniformly, and equals 1 when all samples share the same prediction.
    """
    scores = []
    for part in np.array_split(np.asarray(probs, dtype=np.float64), splits):
        marginal = part.mean(axis=0, keepdims=True)
        kl = (part * (np.log(part + eps) - np.log(marginal + eps))).sum(axis=1)
        scores.append(np.exp(kl.mean()))
    return float(np.mean(scores)), float(np.std(scores))


class Evaluator:
    """Caches the real-data feature statistics and scores image sets or generators against them."""

    def __init__(self, classifier: MNISTClassifier, reference_images: torch.Tensor, batch_size: int = 1000) -> None:
        self.classifier = classifier.eval()
        self.batch_size = batch_size
        features, _ = self.embed(reference_images)
        self.ref_mu, self.ref_sigma = feature_statistics(features)

    @torch.no_grad()
    def embed(self, images: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Return classifier features ``(N, 128)`` and softmax probabilities ``(N, 10)``."""
        device = next(self.classifier.parameters()).device
        feats, probs = [], []
        for start in range(0, images.shape[0], self.batch_size):
            batch = images[start : start + self.batch_size].to(device)
            f = self.classifier.embed(batch)
            feats.append(f.cpu())
            probs.append(torch.softmax(self.classifier.head(f), dim=1).cpu())
        return torch.cat(feats).double().numpy(), torch.cat(probs).double().numpy()

    def score_images(self, images: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, float]:
        """MNIST-IS (mean/std over 10 splits), MNIST-FID vs the reference set and, if ``labels``
        are given, the classifier accuracy w.r.t. those labels (label fidelity of a cGAN)."""
        features, probs = self.embed(images)
        is_mean, is_std = classifier_inception_score(probs)
        mu, sigma = feature_statistics(features)
        scores = {
            "mnist_is": is_mean,
            "mnist_is_std": is_std,
            "mnist_fid": frechet_distance(mu, sigma, self.ref_mu, self.ref_sigma),
        }
        if labels is not None:
            scores["label_accuracy"] = float((probs.argmax(axis=1) == labels.cpu().numpy()).mean())
        return scores

    def score_generator(self, generator: Generator, n: int, seed: int) -> dict[str, float]:
        """Generate ``n`` samples (balanced labels if conditional) and score them."""
        images = generate(generator, n, seed=seed)
        labels = balanced_labels(n, generator.config.n_classes) if generator.config.conditional else None
        return self.score_images(images, labels)


