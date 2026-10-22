"""Common posterior and abstract module interfaces."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


def estimate_standard_normal_kl(
    posterior: "DiagonalGaussianPosterior",
) -> torch.Tensor:
    """Return per-sample ``KL(q(z) || N(0, I))`` in nats.

    0.5 * sum( mu**2 + exp(logvar) - 1 - logvar )
    """
    batch = int(posterior.latent_mean.shape[0])
    n = posterior.latent_numel()
    if n == 0:
        return torch.zeros(
            batch,
            device=posterior.latent_mean.device,
            dtype=posterior.latent_mean.dtype,
        )
    axes = tuple(range(1, posterior.latent_mean.ndim))
    terms = (
        posterior.latent_mean.pow(2)
        + posterior.latent_logvar.exp()
        - 1.0
        - posterior.latent_logvar
    )
    return 0.5 * (terms.sum(dim=axes) if axes else terms)


@dataclass
class DiagonalGaussianPosterior:
    """Single diagonal Gaussian ``N(latent_mean, diag(exp(latent_logvar)))``.

    ``latent_mean`` and ``latent_logvar`` share shape ``(B, *latent_shape)``.
    Empty latents (``prod(latent_shape) == 0``) are allowed and used by the
    trivial/unconditional adapter.
    """

    latent_mean: torch.Tensor
    latent_logvar: torch.Tensor

    def sample(self) -> torch.Tensor:
        if self.latent_mean.numel() == 0:
            return self.latent_mean
        std = (0.5 * self.latent_logvar).exp()
        return self.latent_mean + std * torch.randn_like(self.latent_mean)

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        batch = int(self.latent_mean.shape[0])
        n = self.latent_numel()
        if n == 0:
            return torch.zeros(
                batch, device=self.latent_mean.device, dtype=self.latent_mean.dtype
            )
        axes = tuple(range(1, self.latent_mean.ndim))
        diff = z - self.latent_mean
        inv_var = (-self.latent_logvar).exp()
        logvar_sum = self.latent_logvar.sum(dim=axes) if axes else self.latent_logvar
        squared_error = diff.pow(2) * inv_var
        error_sum = squared_error.sum(dim=axes) if axes else squared_error
        # log N(z; mu, diag(exp(logvar)))
        return -0.5 * n * math.log(2.0 * math.pi) - 0.5 * logvar_sum - 0.5 * error_sum

    def entropy_nats(self) -> torch.Tensor:
        """Return analytical entropy ``H(q)`` in nats, shape ``(B,)``."""
        batch = int(self.latent_mean.shape[0])
        n = self.latent_numel()
        if n == 0:
            return torch.zeros(
                batch, device=self.latent_mean.device, dtype=self.latent_mean.dtype
            )
        axes = tuple(range(1, self.latent_mean.ndim))
        const = 0.5 * n * (1.0 + math.log(2.0 * math.pi))
        return const + 0.5 * (
            self.latent_logvar.sum(dim=axes) if axes else self.latent_logvar
        )

    def latent_numel(self) -> int:
        if self.latent_mean.ndim == 0 or self.latent_mean.shape[0] == 0:
            return 0
        return int(self.latent_mean[0].numel())

    @property
    def latent_shape(self) -> tuple[int, ...]:
        return tuple(self.latent_mean.shape[1:])


def split_posterior_features(
    features: torch.Tensor,
    logvar_init_bias: float = -2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split the final axis into mean and bounded, biased log-variance heads."""
    if features.shape[-1] % 2 != 0:
        raise ValueError(
            f"Backbone posterior-feature width must be even (2 * latent_dim); "
            f"got last-dim = {features.shape[-1]}.  Configure the backbone's "
            f"output width via ``${{x2:${{model.latent_dim}}}}`` or equivalent."
        )
    mean, raw_logvar = features.chunk(2, dim=-1)
    logvar = (raw_logvar + float(logvar_init_bias)).clamp(-15.0, 5.0)
    return mean, logvar


class BaseEncoder(nn.Module, ABC):
    """Unified encoder returning a diagonal Gaussian posterior."""

    @abstractmethod
    def encode_dist(
        self, x: torch.Tensor, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mean, logvar)`` each of shape ``(B, *latent_shape)``."""
        raise NotImplementedError

    def posterior(
        self,
        x: torch.Tensor,
        **kwargs,
    ) -> DiagonalGaussianPosterior:
        mean, logvar = self.encode_dist(x, **kwargs)
        return DiagonalGaussianPosterior(latent_mean=mean, latent_logvar=logvar)


class BasePrior(nn.Module, ABC):
    """Unified prior modeled via continuous-time flow / diffusion."""

    @abstractmethod
    def compute_loss(
        self,
        posterior: DiagonalGaussianPosterior,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Return ``(loss, metrics)``."""
        raise NotImplementedError

    @abstractmethod
    def sample(self, batch_size: int = 1, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def compute_denoising_cost(
        self,
        posterior: DiagonalGaussianPosterior,
        num_u_steps: int = 16,
    ) -> torch.Tensor:
        """Per-sample finite-cutoff weighted denoising integral, in nats."""
        raise NotImplementedError

    def compute_denoising_cost_minus_entropy(
        self,
        posterior: DiagonalGaussianPosterior,
        num_u_steps: int = 16,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Finite-cutoff denoising cost minus posterior entropy, and per dimension.

        This diagnostic can be negative; it is not an NLL, KL, or MI bound.
        """
        cost = self.compute_denoising_cost(posterior, num_u_steps=num_u_steps)
        entropy = posterior.entropy_nats().to(dtype=cost.dtype)
        cost_minus_entropy = cost - entropy
        dims = posterior.latent_numel()
        per_dim = (
            cost_minus_entropy / float(dims)
            if dims > 0
            else torch.zeros_like(cost_minus_entropy)
        )
        return cost_minus_entropy, per_dim


class BaseAggregator(nn.Module, ABC):
    """Merge per-modality posteriors into one diagonal Gaussian."""

    @abstractmethod
    def forward(
        self,
        posteriors: dict[str, DiagonalGaussianPosterior],
        modality_order: list[str],
    ) -> DiagonalGaussianPosterior:
        raise NotImplementedError


class BaseDecoder(nn.Module, ABC):
    """Unified decoder (flow, AR, classifier, etc.)."""

    @abstractmethod
    def compute_loss(
        self, target: torch.Tensor, z_cond: torch.Tensor, **kwargs
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def sample(self, z_cond: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError
