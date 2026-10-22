"""Categorical encoder/decoder adapters for discrete class labels."""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import BaseEncoder, BaseDecoder, split_posterior_features


class ClassifierEncoderAdapter(BaseEncoder):
    """Wrap a categorical embedding backbone into a Gaussian posterior.

    ``backbone(x)`` must return ``(B, L, 2 * latent_dim)``; adapter splits into
    ``(mean, logvar)``.
    """

    def __init__(self, backbone: nn.Module, logvar_init_bias: float = -2.0):
        super().__init__()
        self.backbone = backbone
        self.logvar_init_bias = float(logvar_init_bias)

    def encode_dist(
        self, x: torch.Tensor, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone(x)
        return split_posterior_features(feat, self.logvar_init_bias)


class ClassifierDecoderAdapter(BaseDecoder):
    """Wraps categorical decoders to emit standard compute_loss / sample APIs."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def compute_loss(
        self, target: torch.Tensor, z_cond: torch.Tensor, **kwargs
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        logits = self.backbone(z_cond)

        target_norm = target.squeeze(-1).long()
        loss = F.cross_entropy(logits, target_norm)

        # Per-step adapter metrics intentionally empty — single canonical
        # decoder loss = ``train/{tgt}/loss`` (auto-added by engine).
        # Accuracy / per-class breakdown still computed during eval.
        return loss, {}

    @torch.no_grad()
    def sample(
        self, z_cond: torch.Tensor, *, deterministic: bool = False, **kwargs
    ) -> torch.Tensor:
        """Sample a class, or return the argmax when deterministic."""
        logits = self.backbone(z_cond)
        if deterministic:
            pred = logits.argmax(dim=-1)
        else:
            probs = F.softmax(logits, dim=-1)
            pred = torch.multinomial(probs, 1).squeeze(-1)
        return pred.unsqueeze(-1)
