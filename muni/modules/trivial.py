"""No-op encoder and prior adapters for unconditional (encoder-free) setups."""

import torch
import torch.nn as nn

from .core import BaseEncoder, BasePrior, DiagonalGaussianPosterior


class TrivialEncoderAdapter(BaseEncoder):
    """Returns an empty ``(B, 0, D)`` posterior -- disables the encoder pathway."""

    def __init__(self, _backbone: nn.Module, dim: int = 512):
        super().__init__()
        self.dim = int(dim)

    def encode_dist(
        self, x: torch.Tensor, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = x.shape[0]
        empty = torch.empty(B, 0, self.dim, device=x.device, dtype=x.dtype)
        return empty, empty


class TrivialPriorAdapter(BasePrior):
    """Zero loss, empty samples -- disables the prior pathway."""

    def __init__(self, _backbone: nn.Module, dim: int = 512):
        super().__init__()
        self.dim = int(dim)
        self.dummy_param = nn.Parameter(torch.zeros(1))

    def compute_loss(self, posterior: DiagonalGaussianPosterior):
        # Touch both the dummy param and the posterior to keep the autograd
        # graph connected even when the prior path is inactive.
        loss = 0.0 * self.dummy_param.sum()
        if posterior.latent_numel() > 0:
            loss = loss + 0.0 * posterior.sample().sum()
        metrics = {"prior_loss": loss.detach()}
        return loss, metrics

    @torch.no_grad()
    def sample(
        self, batch_size: int = 1, device: torch.device | None = None
    ) -> torch.Tensor:
        if device is None:
            device = self.dummy_param.device
        return torch.empty(batch_size, 0, self.dim, device=device)

    @torch.no_grad()
    def compute_denoising_cost(
        self,
        posterior: DiagonalGaussianPosterior,
        num_u_steps: int = 16,
    ):
        if posterior.latent_numel() > 0:
            raise NotImplementedError(
                "TrivialPriorAdapter has no denoiser for non-empty latents, "
                "so a denoising cost is unavailable."
            )
        sample = posterior.sample()
        batch = int(sample.shape[0])
        return torch.zeros(batch, device=sample.device, dtype=sample.dtype)
