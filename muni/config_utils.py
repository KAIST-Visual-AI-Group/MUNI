"""OmegaConf loading with recursive ``base`` inheritance and strict CLI overrides.

OmegaConf replaces lists wholesale. A variant that overrides a list must
restate every inherited item it intends to keep.
"""

from collections.abc import Sequence
from math import comb
import importlib
import os
from pathlib import Path
import re

from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.errors import InterpolationResolutionError


# Posterior adapters split backbone outputs into mean and log-variance halves.
if not OmegaConf.has_resolver("x2"):
    OmegaConf.register_new_resolver("x2", lambda x: 2 * int(x))
if not OmegaConf.has_resolver("mul"):
    OmegaConf.register_new_resolver("mul", lambda a, b: int(a) * int(b))
if not OmegaConf.has_resolver("add"):
    OmegaConf.register_new_resolver("add", lambda a, b: int(a) + int(b))


def _gmm_modality_dim(n_modalities, region_dim, region_dim_by_size) -> int:
    """Sum widths of Venn regions containing one modality."""
    n = int(n_modalities)
    fallback = int(region_dim)
    widths = {int(k): int(v) for k, v in dict(region_dim_by_size or {}).items()}
    if n < 2 or fallback < 1:
        raise ValueError("GMM n_modalities must be >= 2 and region_dim must be positive")
    if any(widths.get(size, fallback) < 1 for size in range(1, n + 1)):
        raise ValueError("all GMM region widths must be >= 1")
    return sum(comb(n - 1, size - 1) * widths.get(size, fallback) for size in range(1, n + 1))


if not OmegaConf.has_resolver("gmm_modality_dim"):
    OmegaConf.register_new_resolver("gmm_modality_dim", _gmm_modality_dim)


class ConfigOverrideError(ValueError):
    """Raised when a CLI config override is malformed or ambiguous."""


class MissingConfigEnvironmentError(RuntimeError):
    """Raised when a config references an unset required environment root."""


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise MissingConfigEnvironmentError(
            f"Required environment variable {name} is not set. "
            f"Set {name} to a non-empty local root, or replace its environment "
            "reference with an explicit path in the config."
        )
    return str(Path(value.strip()).expanduser())


if not OmegaConf.has_resolver("required_env"):
    OmegaConf.register_new_resolver("required_env", _required_env, use_cache=False)


_OVERRIDE_KEY_PATTERN = re.compile(
    r"[A-Za-z_][A-Za-z0-9_-]*(?:(?:\.(?:[A-Za-z_][A-Za-z0-9_-]*|[0-9]+))|(?:\[[0-9]+\]))*"
)


def _resolve_base_path(config_path: Path, base_ref: str) -> Path:
    """Find the base config file relative to the referencing config or cwd."""
    base_path = Path(base_ref)
    for candidate in (config_path.parent / base_path, Path.cwd() / base_path):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Base config {base_ref!r} referenced by {config_path} was not found")


def _load_and_merge(config_path: Path, _stack: tuple[Path, ...] = ()) -> DictConfig:
    """Load an unresolved inheritance chain so variant interpolation can propagate."""
    canonical_path = config_path.expanduser().resolve()
    if canonical_path in _stack:
        cycle_start = _stack.index(canonical_path)
        cycle = (*_stack[cycle_start:], canonical_path)
        raise ValueError(
            "Circular config inheritance: " + " -> ".join(str(path) for path in cycle)
        )

    cfg = OmegaConf.load(config_path)
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"Config root must be a mapping: {config_path}")

    base_ref = cfg.get("base")
    if base_ref is not None:
        if not isinstance(base_ref, str):
            raise TypeError(f"Config 'base' must be a path string: {config_path}")
        base_cfg = _load_and_merge(
            _resolve_base_path(config_path, base_ref),
            (*_stack, canonical_path),
        )
        del cfg["base"]
        cfg = OmegaConf.merge(base_cfg, cfg)

    return cfg


def get_obj_from_str(string: str):
    module, cls = string.rsplit(".", 1)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", {}))


def _config_path_exists(cfg: DictConfig, key: str) -> bool:
    """Check key existence without resolving an interpolated value."""
    node = cfg
    for part in re.sub(r"\[([0-9]+)\]", r".\1", key).split("."):
        if isinstance(node, DictConfig):
            if part not in node:
                return False
            node = node._get_node(part)
        elif isinstance(node, ListConfig):
            if not part.isdigit() or int(part) >= len(node):
                return False
            node = node._get_node(int(part))
        else:
            return False
    return True


def _apply_cli_overrides(cfg: DictConfig, cli_overrides: Sequence[str]) -> DictConfig:
    """Apply strict CLI overrides in order.

    Use ``key=value`` to override an existing key and ``+key=value`` to add a
    new key. Using ``+`` for an existing key is rejected.
    """
    for token in cli_overrides:
        if not isinstance(token, str) or "=" not in token:
            raise ConfigOverrideError(
                f"Malformed config override {token!r}. Expected key=value for an existing key "
                "or +key=value for a new key."
            )
        explicit_add = token.startswith("+")
        assignment = token[1:] if explicit_add else token
        key, _, raw_value = assignment.partition("=")
        if not _OVERRIDE_KEY_PATTERN.fullmatch(key):
            hint = (
                " Do not prefix config overrides with '--'."
                if key.startswith("-")
                else ""
            )
            raise ConfigOverrideError(
                f"Malformed config override key {key!r} in {token!r}. Use a dotted key such as train.epochs=100.{hint}"
            )

        exists = _config_path_exists(cfg, key)
        if explicit_add and exists:
            raise ConfigOverrideError(
                f"Config key {key!r} already exists; override it with {key}=value (without '+')."
            )
        if not explicit_add and not exists:
            raise ConfigOverrideError(
                f"Unknown config key {key!r}. Check for a typo, or use +{key}=value to add it intentionally."
            )

        try:
            value = OmegaConf.from_dotlist([f"value={raw_value}"]).value
            OmegaConf.update(cfg, key, value, merge=True, force_add=explicit_add)
        except Exception as exc:
            raise ConfigOverrideError(
                f"Could not apply config override {token!r}: {exc}"
            ) from exc
    return cfg


def load_config(
    config_path: str | Path,
    cli_overrides: Sequence[str] | None = None,
    *,
    resolve: bool = True,
) -> DictConfig:
    """Load and override a config, optionally deferring value resolution."""
    cfg = _load_and_merge(Path(config_path))

    cfg = _apply_cli_overrides(cfg, cli_overrides or ())
    if not resolve:
        return cfg

    try:
        OmegaConf.resolve(cfg)
    except InterpolationResolutionError as exc:
        if isinstance(exc.__context__, MissingConfigEnvironmentError):
            raise exc.__context__ from exc
        raise

    return cfg


def print_config(cfg: DictConfig, title: str = "Configuration") -> None:
    from muni.util.misc import print_on_main as print

    print(f"\n{'=' * 60}")
    print(f"{title:^60}")
    print(f"{'=' * 60}")
    print(OmegaConf.to_yaml(cfg))
    print(f"{'=' * 60}\n")
