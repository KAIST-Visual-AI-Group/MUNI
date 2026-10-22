"""SegFormer-based digit and quadrant verifier."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import SegformerConfig, SegformerModel


class UnifiedVerifier(nn.Module):
    """Shared MiT encoder with classification heads."""

    def __init__(
        self,
        backbone: str = "nvidia/mit-b0",
        pretrained: bool = True,
        cls_heads: dict[str, int] | None = None,
    ):
        super().__init__()
        self.cls_specs = {str(k): int(v) for k, v in (cls_heads or {}).items()}

        if pretrained:
            self.backbone = SegformerModel.from_pretrained(backbone)
        else:
            self.backbone = SegformerModel(SegformerConfig.from_pretrained(backbone))
        hidden_sizes = list(self.backbone.config.hidden_sizes)

        self.cls_heads = nn.ModuleDict(
            {
                name: nn.Linear(hidden_sizes[-1], classes)
                for name, classes in self.cls_specs.items()
            }
        )

    @property
    def heads(self) -> dict[str, int]:
        return dict(self.cls_specs)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = list(
            self.backbone(
                pixel_values=x,
                output_hidden_states=True,
                return_dict=True,
            ).hidden_states
        )
        outputs = {
            name: head(features[-1].mean(dim=(-2, -1)))
            for name, head in self.cls_heads.items()
        }
        return outputs


def build_verifier_model(cfg: dict) -> UnifiedVerifier:
    model_cfg = cfg.get("model", {})
    return UnifiedVerifier(
        backbone=str(model_cfg.get("backbone", "nvidia/mit-b0")),
        pretrained=bool(model_cfg.get("pretrained", True)),
        cls_heads=model_cfg.get("cls_heads"),
    )
