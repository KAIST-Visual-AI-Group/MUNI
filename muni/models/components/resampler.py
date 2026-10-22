"""Reusable Perceiver-style latent resampler blocks."""

from __future__ import annotations

import torch
import torch.nn as nn

from muni.models.components.transformer import CrossAttention, RMSNorm, SwiGLUFFN


class PerceiverResamplerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.norm_slots_cross = RMSNorm(hidden_size, eps=1e-6)
        self.norm_context = RMSNorm(hidden_size, eps=1e-6)
        self.cross_attn = CrossAttention(
            hidden_size,
            context_dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=True,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.norm_mlp = RMSNorm(hidden_size, eps=1e-6)
        self.mlp = SwiGLUFFN(hidden_size, int(hidden_size * mlp_ratio), drop=proj_drop)

    def forward(
        self,
        slots: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        slots = slots + self.cross_attn(
            self.norm_slots_cross(slots),
            self.norm_context(context),
        )
        slots = slots + self.mlp(self.norm_mlp(slots))
        return slots


class PerceiverResampler(nn.Module):
    """Pool variable-length `(B, N, D)` features into learned latent slots."""

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_queries: int,
        *,
        output_dim: int | None = None,
        depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_size = int(hidden_size)
        self.num_queries = int(num_queries)
        self.output_dim = self.hidden_size if output_dim is None else int(output_dim)

        self.input_proj = nn.Linear(self.input_dim, self.hidden_size, bias=True)
        self.query_slots = nn.Parameter(torch.randn(1, self.num_queries, self.hidden_size) * 0.02)

        self.blocks = nn.ModuleList(
            [
                PerceiverResamplerBlock(
                    self.hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                )
                for _ in range(depth)
            ]
        )
        self.norm = RMSNorm(self.hidden_size, eps=1e-6)
        self.out_proj = (
            nn.Identity()
            if self.output_dim == self.hidden_size
            else nn.Linear(self.hidden_size, self.output_dim, bias=True)
        )

        self._init_weights()

    def _init_weights(self) -> None:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

    def forward(self, context: torch.Tensor | None) -> torch.Tensor | None:
        if context is None or context.shape[1] == 0:
            return None

        context = self.input_proj(context)
        slots = self.query_slots.expand(context.shape[0], -1, -1)
        for block in self.blocks:
            slots = block(slots, context)
        return self.out_proj(self.norm(slots))
