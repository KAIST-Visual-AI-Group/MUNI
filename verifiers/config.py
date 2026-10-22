"""Verifier YAML configuration loading."""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
import re
from typing import Any

import yaml


_ENV_PATTERN = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")


def _expand_environment(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if not isinstance(value, str):
        return value

    names = set(_ENV_PATTERN.findall(value))
    missing = sorted(name for name in names if not os.environ.get(name, "").strip())
    if missing:
        raise RuntimeError(
            f"Unset environment variable(s): {', '.join(missing)}"
        )
    return _ENV_PATTERN.sub(
        lambda match: str(Path(os.environ[match.group(1)].strip()).expanduser()),
        value,
    )


def load_verifier_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    return _expand_environment(cfg)
