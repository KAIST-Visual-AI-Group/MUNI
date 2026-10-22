"""MLP encoder and flow backbone for low-dimensional synthetic experiments."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from muni.models.components.transformer import zero_module


class _ResBlock(nn.Module):
    """Identity-initialized pre-RMSNorm residual block for the encoder."""

    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim)
        self.fc1 = nn.Linear(dim, dim, bias=False)
        self.norm2 = nn.RMSNorm(dim)
        self.fc2 = nn.Linear(dim, dim, bias=False)
        zero_module(self.fc2)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.fc1(self.norm1(x)))
        h = F.silu(self.norm2(h))
        h = self.fc2(h)
        return x + self.drop(h)


class _SinTimeEmbed(nn.Module):
    """Sinusoidal timestep embedding + 2-layer MLP (DiT / MAE style)."""

    def __init__(self, dim: int, freq: int = 256):
        super().__init__()
        self.dim = int(dim)
        self.freq = float(freq)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim, bias=True),
            nn.SiLU(),
            nn.Linear(dim, dim, bias=True),
        )
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.normal_(self.mlp[2].weight, std=0.02)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.flatten().to(torch.float32)
        half = self.dim // 2
        freqs = torch.exp(-math.log(self.freq) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([args.cos(), args.sin()], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


class _LabelEmbed(nn.Module):
    """Class label embedder with optional CFG dropout."""

    def __init__(self, num_classes: int, dim: int, drop_prob: float = 0.0):
        super().__init__()
        self.table = nn.Embedding(num_classes + 1, dim)
        nn.init.normal_(self.table.weight, std=0.02)
        self.num_classes = int(num_classes)
        self.drop_prob = float(drop_prob)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        if self.training and self.drop_prob > 0.0:
            mask = torch.rand(y.shape[0], device=y.device) < self.drop_prob
            y = torch.where(mask, torch.full_like(y, self.num_classes), y)
        return self.table(y)


class _AdaLNCatBlock(nn.Module):
    """Identity-initialized adaLN block whose residual branch can learn."""

    def __init__(
        self,
        dim: int,
        ctx_dim: int | None = None,
        mlp_ratio: float = 2.6667,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm = nn.RMSNorm(dim, elementwise_affine=False)
        self.has_ctx = ctx_dim is not None
        in_dim = dim + (int(ctx_dim) if self.has_ctx else 0)

        hidden = int(dim * mlp_ratio)
        self.w_in = nn.Linear(in_dim, hidden, bias=False)
        self.w_gate = nn.Linear(in_dim, hidden, bias=False)
        self.w_out = nn.Linear(hidden, dim, bias=False)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.adaln = nn.Sequential(
            nn.SiLU(),
            zero_module(nn.Linear(dim, 3 * dim, bias=True)),
        )

    def forward(
        self,
        x: torch.Tensor,  # (B, L, dim)
        c: torch.Tensor,  # (B, dim)
        ctx: torch.Tensor | None,  # (B, ctx_dim) or None
    ) -> torch.Tensor:
        shift, scale, gate = self.adaln(c).chunk(3, dim=-1)
        h = self.norm(x) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        if self.has_ctx and ctx is not None:
            ctx_exp = ctx.unsqueeze(1).expand(-1, h.shape[1], -1)
            h = torch.cat([h, ctx_exp], dim=-1)
        h = self.w_out(F.silu(self.w_gate(h)) * self.w_in(h))
        return x + gate.unsqueeze(1) * self.drop(h)


class ToyEncoder(nn.Module):
    """Encode flat features into one posterior token."""

    def __init__(
        self,
        feat_dim: int = 50,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        depth: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_proj = nn.Linear(feat_dim, hidden_dim, bias=True)
        self.blocks = nn.ModuleList([_ResBlock(hidden_dim, dropout=dropout) for _ in range(int(depth))])
        self.out_norm = nn.RMSNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, latent_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        for blk in self.blocks:
            h = blk(h)
        h = self.out_norm(h)
        return self.out_proj(h).unsqueeze(1)


class ToyFlowMLP(nn.Module):
    """Use the same flow backbone for an unconditional prior or decoder."""

    def __init__(
        self,
        token_dim: int,
        hidden_size: int = 256,
        depth: int = 4,
        mlp_ratio: float = 2.6667,
        dropout: float = 0.0,
        num_classes: int = 1,
        label_drop_prob: float = 0.0,
        context_dim: int | None = None,
        context_memory_slots: int | None = None,
        context_drop_prob: float = 0.0,
    ):
        super().__init__()
        self.token_dim = int(token_dim)
        self.hidden_size = int(hidden_size)
        self.num_classes = int(num_classes)
        self.context_drop_prob = float(context_drop_prob)
        self.has_context = context_dim is not None

        self.in_proj = nn.Linear(self.token_dim, self.hidden_size, bias=True)
        self.t_embed = _SinTimeEmbed(self.hidden_size)
        self.y_embed = _LabelEmbed(self.num_classes, self.hidden_size, label_drop_prob)

        if self.has_context:
            slots = int(context_memory_slots) if context_memory_slots is not None else 1
            ctx_input_dim = int(context_dim) * slots
            self.ctx_proj = nn.Sequential(
                nn.Linear(ctx_input_dim, self.hidden_size, bias=True),
                nn.SiLU(),
                nn.Linear(self.hidden_size, self.hidden_size, bias=True),
            )
        else:
            self.ctx_proj = None

        block_ctx_dim = self.hidden_size if self.has_context else None
        self.blocks = nn.ModuleList(
            [
                _AdaLNCatBlock(
                    self.hidden_size,
                    ctx_dim=block_ctx_dim,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(int(depth))
            ]
        )

        # Zero output initializes the flow velocity field at zero.
        self.final_norm = nn.RMSNorm(self.hidden_size, elementwise_affine=False)
        self.final_adaln = nn.Sequential(
            nn.SiLU(),
            zero_module(nn.Linear(self.hidden_size, 2 * self.hidden_size, bias=True)),
        )
        self.final_proj = zero_module(nn.Linear(self.hidden_size, self.token_dim, bias=True))

    def forward(
        self,
        raw: torch.Tensor,  # (B, L, token_dim)
        grid_pos: torch.Tensor,  # ignored (flat seq_len=1)
        t: torch.Tensor,  # (B,) timesteps in [0, 1]
        y: torch.Tensor,  # (B,) class labels
        context: torch.Tensor | None = None,  # (B, L_ctx, ctx_dim) or None
    ) -> torch.Tensor:
        del grid_pos
        x = self.in_proj(raw)
        c = self.t_embed(t) + self.y_embed(y)

        ctx_vec: torch.Tensor | None = None
        if self.has_context and context is not None:
            ctx_flat = context.reshape(context.shape[0], -1)
            ctx_vec = self.ctx_proj(ctx_flat)
            if self.training and self.context_drop_prob > 0.0:
                keep = (torch.rand(ctx_vec.shape[0], 1, device=ctx_vec.device) >= self.context_drop_prob).to(
                    ctx_vec.dtype
                )
                ctx_vec = ctx_vec * keep
            c = c + ctx_vec
        elif self.has_context:
            ctx_vec = torch.zeros(x.shape[0], self.hidden_size, device=x.device, dtype=x.dtype)

        for blk in self.blocks:
            x = blk(x, c, ctx_vec)

        shift, scale = self.final_adaln(c).chunk(2, dim=-1)
        h = self.final_norm(x) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.final_proj(h)
