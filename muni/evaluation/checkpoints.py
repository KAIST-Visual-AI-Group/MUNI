"""Checkpoint selection and model loading for evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from muni.checkpoint_utils import install_model_state
from muni.framework.model import MultimodalWrapper, build_model


def build_muni_model_from_checkpoint(
    cfg: Any,
    checkpoint_path: str | Path,
    device: torch.device,
    *,
    checkpoint_key: str | None = None,
    checkpoint: dict[str, Any] | None = None,
) -> MultimodalWrapper:
    """Build a MUNI model and install its selected evaluation weights."""
    model = build_model(cfg).to(device)
    payload = checkpoint
    if payload is None:
        payload = torch.load(checkpoint_path, map_location="cpu")
    load_ema_checkpoint(model, device, payload, checkpoint_key=checkpoint_key)
    model.eval()
    return model


def select_checkpoint_state_dict(
    ckpt: dict[str, Any],
    checkpoint_key: str | None = None,
) -> tuple[str, dict[str, torch.Tensor]]:
    """Select the requested state dict, preferring EMA weights by default."""
    state_key = checkpoint_key or ("model_ema1" if "model_ema1" in ckpt else "model")
    return state_key, ckpt[state_key]


def load_ema_checkpoint(
    model: MultimodalWrapper,
    device: torch.device,
    ckpt: dict[str, Any],
    checkpoint_key: str | None = None,
) -> tuple[str, dict[str, object]]:
    """Load the selected checkpoint state into ``model`` using its EMA contract."""
    state_key, state_dict = select_checkpoint_state_dict(ckpt, checkpoint_key)
    live_state = ckpt["model"] if state_key == "model_ema1" else state_dict
    missing, unexpected, load_info = install_model_state(
        model, live_state, allow_missing_frozen=True
    )
    if missing or unexpected:
        raise RuntimeError(
            f"Incomplete checkpoint '{state_key}': missing={missing}, unexpected={unexpected}"
        )

    if state_key == "model_ema1":
        # EMA is deduplicated by named parameter, not a full state_dict. Load
        # live buffers/aliases first, then write EMA once per physical tensor.
        normalized_ema = model.ema.normalize(state_dict, model)
        model.to(device)
        model.ema.load(model, normalized_ema, device)
        model.ema.install(model)
        return state_key, load_info

    model.to(device)
    return state_key, load_info
