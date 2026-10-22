"""Shared transformer, conditioning, and positional-encoding primitives."""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


def zero_module(module: nn.Module) -> nn.Module:
    """Zero every parameter in ``module`` in place and return the module."""
    for parameter in module.parameters():
        parameter.detach().zero_()
    return module


class GemmaePrefixProjector(nn.Module):
    """Project GemmaE features to the prefix layout stored in ``gemmae_3.pt``."""

    def __init__(self, in_dim: int = 768, out_dim: int = 1152, plen: int = 3) -> None:
        super().__init__()
        hidden_dim = max(in_dim, out_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, plen * out_dim),
        )
        self.ln = nn.LayerNorm(out_dim)
        self.prefix_len = int(plen)
        self.out_dim = int(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = rearrange(self.net(x), "b (p d) -> b p d", p=self.prefix_len)
        return self.ln(projected)


class TimestepEmbedder(nn.Module):
    """Embed scalar timesteps into a vector of size ``hidden_size``."""

    def __init__(self, hidden_size: int, freq: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(freq, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.freq = freq

    @staticmethod
    def _sincos(t: torch.Tensor, dim: int) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -np.log(10_000) * torch.arange(half, device=t.device, dtype=t.dtype) / half
        )
        args = t[..., None] * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self._sincos(t, self.freq))


# Positional encoding primitives


def make_base_frequencies(num_freqs: int, theta: float = 10000.0) -> torch.Tensor:
    """Core frequency bands: 1 / (theta ** (k / num_freqs)) for k in [0, num_freqs)."""
    omega = torch.arange(num_freqs, dtype=torch.float64)
    return 1.0 / (theta ** (omega / num_freqs))


def compute_pos_2d(
    pos_h: torch.Tensor, pos_w: torch.Tensor, freqs: torch.Tensor
) -> torch.Tensor:
    """Return ``[cos_h, cos_w, sin_h, sin_w]`` positional features."""
    freqs = freqs.to(device=pos_h.device)
    angles_h = torch.einsum("l, f -> l f", pos_h.to(torch.float64), freqs)
    angles_w = torch.einsum("l, f -> l f", pos_w.to(torch.float64), freqs)
    return torch.cat(
        [
            angles_h.cos(),
            angles_w.cos(),
            angles_h.sin(),
            angles_w.sin(),
        ],
        dim=-1,
    ).float()


def compute_pos_1d(pos: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    freqs = freqs.to(device=pos.device)
    angles = torch.einsum("l, f -> l f", pos.to(torch.float64), freqs)
    return torch.cat([angles.cos(), angles.sin()], dim=-1).float()


def _apply_rope_single(x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    rope_cos, rope_sin = rope.chunk(2, dim=-1)
    rope_cos = repeat(rope_cos, "... f -> ... (f r)", r=2)
    rope_sin = repeat(rope_sin, "... f -> ... (f r)", r=2)

    def _rotate_half(y):
        y = rearrange(y, "... (d r) -> ... d r", r=2)
        y1, y2 = y.unbind(dim=-1)
        return rearrange(torch.stack((-y2, y1), dim=-1), "... d r -> ... (d r)")

    return x * rope_cos + _rotate_half(x) * rope_sin


def apply_rope(q, k, rope):
    """Apply shared broadcastable rotary embeddings to queries and keys."""
    return _apply_rope_single(q, rope), _apply_rope_single(k, rope)


class LabelEmbedder(nn.Module):
    """Embeds class labels with optional dropout for CFG-like conditioning."""

    def __init__(self, num_classes, hidden_size, dropout_prob=0.0):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels):
        drop = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        return torch.where(drop, torch.full_like(labels, self.num_classes), labels)

    def forward(self, labels):
        if self.training and self.dropout_prob > 0:
            labels = self.token_drop(labels)
        return self.embedding_table(labels)


class SwiGLUFFN(nn.Module):
    def __init__(self, in_features, hidden_features, drop=0.0):
        super().__init__()
        hidden_features = int(hidden_features * 2 / 3)
        self.fc1 = nn.Linear(in_features, hidden_features * 2)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x, gate = self.fc1(x).chunk(2, dim=-1)
        return self.drop(self.fc2(x * F.silu(gate)))


class CrossAttention(nn.Module):
    """Cross-attention over a separate key/value sequence."""

    def __init__(
        self,
        dim,
        context_dim=None,
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        kv_dim = dim if context_dim is None else int(context_dim)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(kv_dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context):
        q = rearrange(self.q(x), "B N (H D) -> B H N D", H=self.num_heads)
        kv = rearrange(
            self.kv(context), "B N (two H D) -> two B H N D", two=2, H=self.num_heads
        )
        k, v = kv.unbind(0)

        q = self.q_norm(q)
        k = self.k_norm(k)

        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        x = rearrange(x, "B H N D -> B N (H D)")
        return self.proj_drop(self.proj(x))
