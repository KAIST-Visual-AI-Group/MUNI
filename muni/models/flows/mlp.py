"""Attention-free adaLN-Zero flow backbones for short latent sequences."""

import torch
import torch.nn as nn
from timm.models.vision_transformer import Mlp

from muni.models.components.transformer import LabelEmbedder, TimestepEmbedder, modulate, zero_module


class MLPFlowBlock(nn.Module):
    """Pre-norm residual MLP block with adaLN-Zero gating."""

    def __init__(self, hidden_size: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)

        def approx_gelu():
            return nn.GELU(approximate="tanh")

        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden,
            act_layer=approx_gelu,
            drop=dropout,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            zero_module(nn.Linear(hidden_size, 3 * hidden_size, bias=True)),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale, gate = self.adaLN_modulation(c).chunk(3, dim=-1)
        h = self.mlp(modulate(self.norm(x), shift.unsqueeze(1), scale.unsqueeze(1)))
        return x + gate.unsqueeze(1) * h


class MLPFlowFinalLayer(nn.Module):
    """adaLN-conditioned output projection (zero-initialized)."""

    def __init__(self, hidden_size: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        zero_module(self.adaLN_modulation[-1])
        zero_module(self.linear)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        return self.linear(modulate(self.norm(x), shift.unsqueeze(1), scale.unsqueeze(1)))


class MLPFlowDenoiser(nn.Module):
    """Denoise tokens with context folded into the adaLN conditioning vector."""

    def _init_weights(self) -> None:
        # Gates and output projections retain their constructors' zero policy.
        xavier_linears = [self.input_proj]
        if self.has_context:
            xavier_linears.extend((self.context_proj[0], self.context_proj[2]))
        for block in self.blocks:
            xavier_linears.extend((block.mlp.fc1, block.mlp.fc2))
        for linear in xavier_linears:
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)

        for linear in (self.t_embedder.mlp[0], self.t_embedder.mlp[2]):
            nn.init.normal_(linear.weight, std=0.02)
            nn.init.zeros_(linear.bias)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

    def _init_conditioning(self, hidden_size, num_classes, label_drop_prob, context_dim, context_memory_slots):
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, label_drop_prob)
        self.has_context = context_dim is not None
        if self.has_context:
            slots = context_memory_slots if context_memory_slots is not None else 1
            self.context_proj = nn.Sequential(
                nn.Linear(int(context_dim) * int(slots), hidden_size, bias=True),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size, bias=True),
            )

    def _context_condition(self, context):
        ctx_cond = self.context_proj(context.reshape(context.shape[0], -1))
        if self.training and self.context_drop_prob > 0.0:
            keep = (torch.rand(ctx_cond.shape[0], 1, device=ctx_cond.device) >= self.context_drop_prob).to(
                ctx_cond.dtype
            )
            ctx_cond = ctx_cond * keep
        return ctx_cond


    def __init__(
        self,
        token_dim: int,
        hidden_size: int = 768,
        depth: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        num_classes: int = 1,
        label_drop_prob: float = 0.0,
        context_dim: int | None = None,
        context_memory_slots: int | None = None,
        context_drop_prob: float = 0.0,
    ):
        super().__init__()
        self.token_dim = int(token_dim)
        self.num_classes = int(num_classes)
        self.hidden_size = int(hidden_size)
        self.context_drop_prob = float(context_drop_prob)

        self.input_proj = nn.Linear(self.token_dim, hidden_size, bias=True)
        self.final_layer = MLPFlowFinalLayer(hidden_size, self.token_dim)

        self._init_conditioning(hidden_size, num_classes, label_drop_prob, context_dim, context_memory_slots)

        self.blocks = nn.ModuleList([MLPFlowBlock(hidden_size, mlp_ratio, dropout) for _ in range(depth)])

        self._init_weights()


    def forward(
        self,
        raw: torch.Tensor,
        grid_pos: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.input_proj(raw)
        cond = self.t_embedder(t) + self.y_embedder(y)

        if context is not None and self.has_context:
            cond = cond + self._context_condition(context)

        for block in self.blocks:
            x = block(x, cond)

        return self.final_layer(x, cond)
