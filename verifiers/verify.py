"""Checkpoint-backed inference wrapper for visual verifiers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .model import UnifiedVerifier, build_verifier_model


def _checkpoint_config(checkpoint: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(checkpoint["config"])
    model_cfg = dict(cfg.get("model", {}))
    model_cfg["pretrained"] = False
    cfg["model"] = model_cfg
    return cfg


class FrozenVerifier:
    """Frozen verifier using EMA checkpoint weights when available."""

    def __init__(self, model: UnifiedVerifier, device: torch.device):
        self.model = model.to(device).eval()
        self.model.requires_grad_(False)
        self.device = device
        self._cache_input: torch.Tensor | None = None
        self._cache_out: dict[str, torch.Tensor] | None = None

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        device: str | torch.device = "cuda",
        *,
        use_ema: bool = True,
    ) -> FrozenVerifier:
        checkpoint = torch.load(path, map_location="cpu")
        model = build_verifier_model(_checkpoint_config(checkpoint))
        model.load_state_dict(checkpoint["model"])

        if use_ema and "ema" in checkpoint:
            tensors = {**dict(model.named_parameters()), **dict(model.named_buffers())}
            for name, value in checkpoint["ema"].items():
                target = tensors.get(name)
                if target is not None and target.is_floating_point():
                    target.data.copy_(value)
        return cls(model, torch.device(device))

    @torch.no_grad()
    def predict(self, rgb: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the verifier without changing the caller's RGB value range."""
        if rgb is not self._cache_input:
            dtype = next(self.model.parameters()).dtype
            self._cache_out = self.model(
                rgb.to(self.device, dtype=dtype, non_blocking=True)
            )
            self._cache_input = rgb
        return self._cache_out

    @torch.no_grad()
    def predict_classes(
        self, rgb: torch.Tensor, *, heads: tuple[str, ...] | None = None
    ) -> dict[str, torch.Tensor]:
        """Postprocess all heads by default, or only explicitly requested heads."""
        outputs = self.predict(rgb)
        selected = tuple(self.model.cls_specs) if heads is None else heads
        return {name: outputs[name].argmax(dim=-1) for name in selected}
