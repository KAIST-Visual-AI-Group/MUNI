"""Compile-safe checkpoint key matching."""

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

    if set(model_keys) == set(state_dict):
        return dict(state_dict), {"translated_keys": 0, "missing": [], "unused": []}

    normalized_ckpt: dict[str, list[str]] = {}
    for key in state_dict:
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

    unused = [key for key in state_dict if key not in used_ckpt_keys]
    return adapted, {
        "translated_keys": translated,
        "missing": missing,
        "unused": unused,
    }
