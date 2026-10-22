"""Adapters for pre-extracted modality features."""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import BaseEncoder, BaseDecoder, split_posterior_features
from .ops import scale_grad
from muni.util.misc import unwrap_modality_feature


class LatentFeatureEncoderAdapter(BaseEncoder):
    """Split a feature backbone's output into latent mean and log-variance."""

    def __init__(self, backbone: nn.Module, logvar_init_bias: float = -2.0):
        super().__init__()
        self.backbone = backbone
        self.logvar_init_bias = float(logvar_init_bias)

    def encode_dist(self, x, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        x = unwrap_modality_feature(x)
        feat = self.backbone(x)
        return split_posterior_features(feat, self.logvar_init_bias)


class LatentFeatureMMSEDecoderAdapter(BaseDecoder):
    """Regress a modality feature directly from latent context."""

    def __init__(
        self,
        backbone: nn.Module,
        encoder_grad_scale: float = 1.0,
        loss_type: str = "mse",
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.encoder_grad_scale = float(encoder_grad_scale)
        if loss_type not in ("mse", "cosine"):
            raise ValueError(f"loss_type must be 'mse' or 'cosine'; got {loss_type!r}.")
        self.loss_type = loss_type

    def compute_loss(
        self,
        target: torch.Tensor,
        z_cond: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        z = scale_grad(z_cond, self.encoder_grad_scale)
        pred = self.backbone(z)

        if self.loss_type == "mse":
            loss = F.mse_loss(pred, target)
        else:
            loss = 1.0 - F.cosine_similarity(pred, target, dim=-1).mean()
        return loss, {}

    @torch.no_grad()
    def sample(
        self,
        z_cond: torch.Tensor,
        **kwargs,  # noqa: ARG002  (accepted for adapter API parity)
    ) -> torch.Tensor:
        return self.backbone(z_cond)

