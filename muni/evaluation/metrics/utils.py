"""Numerical kernels shared by every metric implementation.

This module contains no dataset loading, decoding, logging, or benchmark
protocol decisions.  Online validation and paper benchmarks must call these
same functions so a metric's mathematical definition has one owner.
"""

from __future__ import annotations

import numpy as np
import torch


def frechet_distance(
    mean_a: np.ndarray,
    covariance_a: np.ndarray,
    mean_b: np.ndarray,
    covariance_b: np.ndarray,
) -> float:
    """Return the Fréchet distance between two Gaussian distributions."""
    from scipy import linalg

    difference = mean_a - mean_b
    covariance_mean, _ = linalg.sqrtm(covariance_a @ covariance_b, disp=False)
    if np.iscomplexobj(covariance_mean):
        covariance_mean = covariance_mean.real
    return float(
        difference @ difference
        + np.trace(covariance_a + covariance_b - 2.0 * covariance_mean)
    )


def cosine_similarity_scores(
    features_a: torch.Tensor,
    features_b: torch.Tensor,
    *,
    scale: float = 100.0,
) -> torch.Tensor:
    """Return one scaled cosine score per aligned feature pair."""
    return (features_a * features_b).sum(dim=-1) * float(scale)


def relative_ranking_scores(
    generated_features: torch.Tensor,
    conditioned_features: torch.Tensor,
    reference_features: torch.Tensor,
    *,
    scale: float = 100.0,
) -> torch.Tensor:
    """Return per-sample relative-rank scores against a reference pool."""

    conditioned_similarity = (generated_features * conditioned_features).sum(dim=-1)
    reference_similarities = generated_features @ reference_features.T
    return (
        (reference_similarities < conditioned_similarity.unsqueeze(1))
        .float()
        .mean(dim=-1)
        * float(scale)
    )


def per_sample_class_correct(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Return one float 0/1 classification result per sample on CPU."""
    return (predictions.reshape(-1) == targets.reshape(-1)).float().cpu()


def squared_error_sum_and_count(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[float, int]:
    """Return the float32 squared-error sum and compared element count."""
    difference = (
        predictions.detach().float() - targets.detach().float()
    ).reshape(-1)
    return float((difference * difference).sum().item()), int(difference.numel())

