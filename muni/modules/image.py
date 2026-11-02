"""Image encoder adapters mapping ConvNet features to latent-posterior sequences.

Each adapter emits ``2 * latent_dim`` channels per token and splits them into
``(mean, logvar)`` heads; see ``muni/modules/core.py::split_posterior_features``.
"""

import torch
import torch.nn as nn
from einops import rearrange

from .core import BaseEncoder, split_posterior_features
from muni.models.components.resampler import PerceiverResampler


class ImageGridEncoderAdapter(BaseEncoder):
    """Flatten a ConvNet feature map into a spatial token sequence posterior.

    The backbone's feature-map channel width must equal ``2 * latent_dim``.
    """

    def __init__(self, backbone: nn.Module, logvar_init_bias: float = -2.0):
        super().__init__()
        self.backbone = backbone
        self.logvar_init_bias = float(logvar_init_bias)

    def encode_dist(
        self, x: torch.Tensor, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone(x)  # (B, 2D, h, w)
        tokens = rearrange(feat, "B D h w -> B (h w) D")  # (B, L, 2D)
        return split_posterior_features(tokens, self.logvar_init_bias)


class ImageResamplerEncoderAdapter(BaseEncoder):
    """Perceiver-compress spatial features into a fixed-length latent posterior.

    ``output_dim`` is the width of the resampler's final projection; it must
    equal ``2 * latent_dim`` so the adapter can split into parallel
    ``(mean, logvar)`` heads.  Configure via ``${x2:${model.latent_dim}}``.
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        num_latents: int,
        output_dim: int,
        hidden_size: int | None = None,
        num_heads: int = 4,
        depth: int = 2,
        logvar_init_bias: float = -2.0,
        modality_idx: int | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.logvar_init_bias = float(logvar_init_bias)
        hidden_size = int(feature_dim) if hidden_size is None else int(hidden_size)
        self.resampler = PerceiverResampler(
            input_dim=int(feature_dim),
            hidden_size=hidden_size,
            output_dim=int(output_dim),
            num_queries=int(num_latents),
            depth=int(depth),
            num_heads=int(num_heads),
        )
        self.modality_idx = None if modality_idx is None else int(modality_idx)

    def encode_dist(
        self, x: torch.Tensor, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.modality_idx is None:
            feat = self.backbone(x)
        else:
            feat = self.backbone(x, modality_idx=self.modality_idx)
        tokens = rearrange(feat, "B D h w -> B (h w) D")
        posterior_feat = self.resampler(tokens)
        return split_posterior_features(posterior_feat, self.logvar_init_bias)
