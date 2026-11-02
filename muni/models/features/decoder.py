"""Context-conditional MLP regressor for deterministic feature decoding."""

from __future__ import annotations

import torch
import torch.nn as nn

from muni.models.flows.mlp import MLPFlowBlock, MLPFlowFinalLayer


class MLPFeatureDecoder(nn.Module):
    """Regress one feature vector from an aggregated latent sequence."""

    def __init__(
        self,
        feat_dim: int,
        context_dim: int,
        context_memory_slots: int = 1,
        hidden_size: int = 768,
        depth: int = 10,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.context_dim = int(context_dim)
        self.context_memory_slots = int(context_memory_slots)
        self.hidden_size = int(hidden_size)

        ctx_input_dim = self.context_dim * self.context_memory_slots
        self.context_proj = nn.Sequential(
            nn.Linear(ctx_input_dim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

        self.query_token = nn.Parameter(torch.empty(1, 1, hidden_size))

        self.blocks = nn.ModuleList([MLPFlowBlock(hidden_size, mlp_ratio, dropout) for _ in range(depth)])
        self.final_layer = MLPFlowFinalLayer(hidden_size, self.feat_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        # Only context and residual MLP matrices use Xavier initialization.
        xavier_linears = [self.context_proj[0], self.context_proj[2]]
        for block in self.blocks:
            xavier_linears.extend((block.mlp.fc1, block.mlp.fc2))
        for linear in xavier_linears:
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)

        nn.init.normal_(self.query_token, std=0.02)

        # Override only the output weight; constructor-owned gates/bias stay zero.
        # A small nonzero output projection breaks feature-dimension symmetry.
        nn.init.normal_(self.final_layer.linear.weight, std=0.02)

    def forward(self, z_cond: torch.Tensor) -> torch.Tensor:
        B = z_cond.shape[0]
        cond = self.context_proj(z_cond.reshape(B, -1))
        x = self.query_token.expand(B, -1, -1)
        for block in self.blocks:
            x = block(x, cond)
        out = self.final_layer(x, cond)
        return out.squeeze(1)
