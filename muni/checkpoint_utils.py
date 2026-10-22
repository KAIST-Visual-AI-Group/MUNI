"""Utilities for compile-safe checkpoint loading."""

from __future__ import annotations

from collections.abc import Mapping

import torch


def normalize_state_dict_key(key: str) -> str:
    """Normalize keys across compile-on/off checkpoint saves."""
    if key.startswith("_orig_mod."):
        key = key[len("_orig_mod.") :]
    return key.replace("._orig_mod.", ".")


def adapt_state_dict_keys(
    state_dict: Mapping[str, torch.Tensor],
    target_keys,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    """Match target keys by native name first, then an unambiguous compiled alias."""
    model_keys = list(target_keys)
    ckpt_keys = list(state_dict.keys())

    if set(model_keys) == set(ckpt_keys):
        return dict(state_dict), {"translated_keys": 0, "missing": [], "unused": []}

    normalized_ckpt: dict[str, list[str]] = {}
    for key in ckpt_keys:
        normalized_ckpt.setdefault(normalize_state_dict_key(key), []).append(key)

    adapted: dict[str, torch.Tensor] = {}
    used_ckpt_keys: set[str] = set()
    translated = 0
    missing: list[str] = []

    for model_key in model_keys:
        source_key = None
        if model_key in state_dict:
            source_key = model_key
        else:
            candidates = normalized_ckpt.get(normalize_state_dict_key(model_key), [])
            if len(candidates) == 1:
                source_key = candidates[0]
            elif len(candidates) > 1:
                raise RuntimeError(f"Checkpoint key mapping is ambiguous for '{model_key}': {candidates}")

        if source_key is None:
            missing.append(model_key)
            continue

        adapted[model_key] = state_dict[source_key]
        used_ckpt_keys.add(source_key)
        if source_key != model_key:
            translated += 1

    unused = [key for key in ckpt_keys if key not in used_ckpt_keys]
    return adapted, {
        "translated_keys": translated,
        "missing": missing,
        "unused": unused,
    }


def install_model_state(
    model: torch.nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    allow_missing_frozen: bool = False,
) -> tuple[list[str], list[str], dict[str, object]]:
    """Install adapted weights, leaving acceptance and errors to the caller."""
    adapted_state, load_info = adapt_state_dict_keys(state_dict, model.state_dict())
    missing, unexpected = model.load_state_dict(adapted_state, strict=False)
    if missing and allow_missing_frozen:
        trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}
        missing = [name for name in missing if name in trainable_names]
    return missing, unexpected, load_info
