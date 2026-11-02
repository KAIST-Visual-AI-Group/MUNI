"""MLP encoders for pre-extracted features."""

import torch
import torch.nn as nn

from muni.models.components.transformer import zero_module


class _ResBlockNoTime(nn.Module):
    """FlowBind-compatible residual block initialized as the identity."""

    def __init__(self, channels: int, mid_channels: int, dropout: float = 0.1):
        super().__init__()
        self.in_layers = nn.Sequential(
            nn.LayerNorm(channels),
            nn.SiLU(),
            nn.Linear(channels, mid_channels, bias=True),
        )
        out_linear = zero_module(nn.Linear(mid_channels, channels, bias=True))
        self.out_layers = nn.Sequential(
            nn.LayerNorm(mid_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            out_linear,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.out_layers(self.in_layers(x))


class ResidualMLPFlatEncoder(nn.Module):
    """FlowBind-compatible residual feature encoder."""

    def __init__(
        self,
        feat_dim: int = 768,
        latent_dim: int = 1536,
        hidden_dim: int = 2048,
        num_blocks: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.input_proj = nn.Linear(feat_dim, hidden_dim)
        self.blocks = nn.Sequential(
            *[_ResBlockNoTime(hidden_dim, hidden_dim, dropout=dropout) for _ in range(int(num_blocks))]
        )
        self.out_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.blocks(h)
        h = self.out_proj(h)
        return h.unsqueeze(1)
