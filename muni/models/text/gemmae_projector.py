"""Project latent context through the pretrained GemmaE text-prefix manifold.

The trainable head predicts a normalized text feature. The GemmaE statistics
and prefix projector then reproduce the feature-to-prefix path used at eval.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from muni.models.components.transformer import GemmaePrefixProjector

from muni.models.features.decoder import MLPFeatureDecoder


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_IMAGE_TEXT_AUDIO_STATS_DIR = _PROJECT_ROOT / "assets" / "image_text_audio" / "stats"


class GemmaeReuseProjector(nn.Module):
    """Map context to a text feature and then to Gemma prefix embeddings."""

    def __init__(
        self,
        context_dim: int,
        context_memory_slots: int = 1,
        feat_dim: int = 768,
        prefix_len: int = 3,
        prefix_out_dim: int = 1152,
        gemmae_proj_ckpt: str | None = None,
        hidden_size: int = 768,
        depth: int = 10,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        freeze_prefix_proj: bool = True,
    ) -> None:
        super().__init__()
        self.prefix_len = int(prefix_len)
        self.feat_dim = int(feat_dim)
        # Compatibility marker consumed by GemmaARTextDecoderAdapter.
        self.align_head = True

        self.feature_head = MLPFeatureDecoder(
            feat_dim=feat_dim,
            context_dim=context_dim,
            context_memory_slots=context_memory_slots,
            hidden_size=hidden_size,
            depth=depth,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        self.prefix_proj = GemmaePrefixProjector(
            in_dim=feat_dim,
            out_dim=prefix_out_dim,
            plen=prefix_len,
        )
        ckpt_path = Path(gemmae_proj_ckpt)
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = payload["prefix_proj"]
        self.prefix_proj.load_state_dict(state, strict=True)
        # Unfreezing is intended only for protocols whose CE modality gate
        # prevents unsupported routes from adapting the pretrained mapping.
        self.freeze_prefix_proj = bool(freeze_prefix_proj)
        for p in self.prefix_proj.parameters():
            p.requires_grad_(not self.freeze_prefix_proj)
        if self.freeze_prefix_proj:
            self.prefix_proj.eval()

        mean = torch.load(
            _IMAGE_TEXT_AUDIO_STATS_DIR / "gemmae_text_mean.pt", map_location="cpu"
        ).float()
        std = torch.load(
            _IMAGE_TEXT_AUDIO_STATS_DIR / "gemmae_text_std.pt", map_location="cpu"
        ).float()
        if mean.shape[-1] != feat_dim or std.shape[-1] != feat_dim:
            raise ValueError(
                f"text mean/std last dim {mean.shape[-1]} does not match feat_dim={feat_dim}"
            )
        self.register_buffer("text_mean", mean)
        self.register_buffer("text_std", std)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_prefix_proj:
            self.prefix_proj.eval()
        return self

    def forward(self, z_cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.feature_head(z_cond)
        feat_dn = feat * self.text_std + self.text_mean
        # Match the fp32 GemmaE evaluation path.
        with torch.autocast(device_type=feat.device.type, enabled=False):
            prefix = self.prefix_proj(feat_dn.float())
        return prefix, feat
