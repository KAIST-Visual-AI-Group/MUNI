"""Per-modality private-latent prior used by MMVAE+."""

from __future__ import annotations

import torch
import torch.nn as nn

from .core import DiagonalGaussianPosterior


class ModalityPrivatePrior(nn.Module):
    """Zero-mean diagonal Gaussian with an optionally learned log-scale."""

    def __init__(
        self,
        dim: int,
        init_log_sigma: float = 0.0,
        learnable: bool = False,
    ):
        super().__init__()
        if int(dim) <= 0:
            raise ValueError(f"ModalityPrivatePrior dim must be positive, got {dim}.")
        self.dim = int(dim)
        self.mu = nn.Parameter(torch.zeros(self.dim), requires_grad=False)
        self.log_sigma = nn.Parameter(
            torch.full((self.dim,), float(init_log_sigma)),
            requires_grad=bool(learnable),
        )

    def posterior(
        self, batch_size: int, *latent_extra: int, dtype: torch.dtype | None = None
    ) -> DiagonalGaussianPosterior:
        """Broadcast ``r_m`` to shape ``(B, *latent_extra, D_z)``."""
        target_shape = (int(batch_size), *tuple(int(x) for x in latent_extra), self.dim)
        mean = self.mu if dtype is None else self.mu.to(dtype=dtype)
        logvar = 2.0 * self.log_sigma
        if dtype is not None:
            logvar = logvar.to(dtype=dtype)
        return DiagonalGaussianPosterior(
            latent_mean=mean.expand(target_shape),
            latent_logvar=logvar.expand(target_shape),
        )

    def sample(
        self, batch_size: int, *latent_extra: int, dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        return self.posterior(batch_size, *latent_extra, dtype=dtype).sample()

    def kl(self, posterior: DiagonalGaussianPosterior) -> torch.Tensor:
        """Return per-sample KL from the encoded private posterior to this prior."""
        mean = self.mu.to(dtype=posterior.latent_mean.dtype)
        logvar = (2.0 * self.log_sigma).to(dtype=posterior.latent_logvar.dtype)
        precision = (-logvar).exp()
        terms = (
            (posterior.latent_mean - mean).square() * precision
            + posterior.latent_logvar.exp() * precision
            - 1.0
            - posterior.latent_logvar
            + logvar
        )
        axes = tuple(range(1, terms.ndim))
        return 0.5 * (terms.sum(dim=axes) if axes else terms)
