from __future__ import annotations

import numpy as np
import pytest
import torch

from minigan.classifier import MNISTClassifier
from minigan.metrics import (
    Evaluator,
    classifier_inception_score,
    feature_statistics,
    frechet_distance,
    nearest_neighbours,
)


def _random_spd(rng: np.random.Generator, d: int) -> np.ndarray:
    a = rng.normal(size=(d, d))
    return a @ a.T + 0.1 * np.eye(d)


def test_frechet_distance_of_identical_distributions_is_zero() -> None:
    features = np.random.default_rng(0).normal(size=(2000, 32))
    mu, sigma = feature_statistics(features)
    assert frechet_distance(mu, sigma, mu, sigma) == pytest.approx(0.0, abs=1e-8)


def test_frechet_distance_pure_mean_shift() -> None:
    rng = np.random.default_rng(1)
    sigma = _random_spd(rng, 8)
    shift = rng.normal(size=8)
    assert frechet_distance(np.zeros(8), sigma, shift, sigma) == pytest.approx(shift @ shift, rel=1e-8)


def test_frechet_distance_diagonal_closed_form() -> None:
    a, b = np.array([1.0, 4.0, 9.0]), np.array([4.0, 1.0, 9.0])
    expected = np.sum((np.sqrt(a) - np.sqrt(b)) ** 2)  # = 1 + 1 + 0
    assert frechet_distance(np.zeros(3), np.diag(a), np.zeros(3), np.diag(b)) == pytest.approx(expected)


def test_frechet_distance_matches_scipy_sqrtm_and_is_symmetric() -> None:
    scipy_linalg = pytest.importorskip("scipy.linalg")  # reference implementation (optional dependency)
    rng = np.random.default_rng(2)
    mu1, mu2 = rng.normal(size=6), rng.normal(size=6)
    s1, s2 = _random_spd(rng, 6), _random_spd(rng, 6)
    covmean = scipy_linalg.sqrtm(s1 @ s2).real
    reference = (mu1 - mu2) @ (mu1 - mu2) + np.trace(s1 + s2 - 2 * covmean)
    assert frechet_distance(mu1, s1, mu2, s2) == pytest.approx(reference, rel=1e-6)
    assert frechet_distance(mu1, s1, mu2, s2) == pytest.approx(frechet_distance(mu2, s2, mu1, s1), rel=1e-6)


def test_frechet_distance_grows_with_distribution_gap() -> None:
    rng = np.random.default_rng(3)
    ref = feature_statistics(rng.normal(size=(5000, 4)))
    same = feature_statistics(rng.normal(size=(5000, 4)))
    shifted = feature_statistics(rng.normal(loc=0.5, size=(5000, 4)))
    assert frechet_distance(*same, *ref) < 0.02 < frechet_distance(*shifted, *ref)


def test_inception_score_extremes() -> None:
    confident_uniform = np.eye(10)[np.arange(1000) % 10]
    assert classifier_inception_score(confident_uniform)[0] == pytest.approx(10.0, rel=1e-6)
    collapsed = np.tile(np.eye(10)[3], (1000, 1))
    assert classifier_inception_score(collapsed)[0] == pytest.approx(1.0, rel=1e-6)
    uncertain = np.full((1000, 10), 0.1)
    assert classifier_inception_score(uncertain)[0] == pytest.approx(1.0, rel=1e-6)


def test_evaluator_scores_reference_set_as_perfect_match() -> None:
    images = torch.rand(200, 1, 28, 28) * 2 - 1
    evaluator = Evaluator(MNISTClassifier(), images, batch_size=64)
    scores = evaluator.score_images(images, labels=torch.zeros(200, dtype=torch.long))
    assert scores["mnist_fid"] == pytest.approx(0.0, abs=1e-6)
    assert 1.0 <= scores["mnist_is"] <= 10.0
    assert 0.0 <= scores["label_accuracy"] <= 1.0


def test_nearest_neighbours_finds_exact_copies() -> None:
    bank = torch.rand(50, 1, 28, 28)
    distances, indices = nearest_neighbours(bank[[4, 17]], bank, k=2, chunk=1)
    assert indices[:, 0].tolist() == [4, 17]
    assert torch.allclose(distances[:, 0], torch.zeros(2), atol=1e-3)
    assert (distances[:, 1] > 1.0).all()
