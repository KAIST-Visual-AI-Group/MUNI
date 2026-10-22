"""Stochastic routing policy and metric namespacing for training."""

from collections.abc import Mapping
from dataclasses import dataclass
import math
import random

import torch

import muni.util.misc as misc

# Source kinds choose all, one, any non-empty, or a near-full subset. Target
# kinds choose prior, any available decoder, self, cross, or their weighted
# union. all_cross is the sole structurally impossible grid entry.
SOURCE_KINDS: tuple[str, ...] = ("all", "one", "any", "almost")
TARGET_KINDS: tuple[str, ...] = ("any", "decoder", "prior", "self", "cross")

TRAIN_PATH_TYPES: tuple[str, ...] = tuple(f"{s}_{t}" for s in SOURCE_KINDS for t in TARGET_KINDS)
STOP_GRAD_MODES: tuple[str, ...] = ("pairs", "none", "target", "all")


def _split_path_name(name: str) -> tuple[str, str]:
    source, target = name.split("_", 1)
    return source, target


def _path_feasible(source_kind: str, target_kind: str, A: int, M: int) -> bool:
    if A == 0:
        return False
    if source_kind == "almost" and A < M - 1:
        return False
    if target_kind == "cross":
        if source_kind == "all":
            return False
        if A < 2:
            return False
        if source_kind == "almost" and A < M:
            return False
    return True


def _weighted_choice(rng: random.Random, items: list[str], weights: dict[str, float]) -> str:
    w = [float(weights.get(m, 1.0)) for m in items]
    if sum(w) <= 0:
        return rng.choice(items)
    return rng.choices(items, weights=w, k=1)[0]


def _resolve_stop_grad_pairs(
    routing_cfg,
    modalities: list[str],
) -> set[tuple[str, str]]:
    """Normalize fallback ``(encoded source, target)`` stop-gradient rules."""
    if routing_cfg is None:
        return set()

    spec = routing_cfg.get("stop_grad_pairs", [])
    if spec is None:
        return set()
    if isinstance(spec, str):
        if spec.strip().lower() in {"", "none"}:
            return set()
        raise ValueError(f"model.routing.stop_grad_pairs must be a list of pairs, not a single string {spec!r}.")

    known = set(modalities)
    valid_targets = known | {"prior"}
    pairs: set[tuple[str, str]] = set()
    for idx, entry in enumerate(spec):
        if isinstance(entry, str):
            src, tgt = (part.strip() for part in entry.split("->", 1))
        elif isinstance(entry, Mapping):
            src = entry.get("src")
            tgt = entry.get("tgt", entry.get("target"))
        else:
            src, tgt = entry

        src = str(src)
        tgt = str(tgt)
        if src not in known:
            raise ValueError(
                f"Unknown stop-grad source modality {src!r} in pair {idx}. Known modalities: {modalities}."
            )
        if tgt not in valid_targets:
            raise ValueError(f"Unknown stop-grad target {tgt!r} in pair {idx}. Known targets: {sorted(valid_targets)}.")
        pairs.add((src, tgt))
    return pairs


def _get_path_weight(train_paths_cfg, path_type: str) -> float:
    path_cfg = train_paths_cfg.get(path_type, {})
    if path_cfg is None:
        return 0.0
    if isinstance(path_cfg, (float, int)):
        return float(path_cfg)
    return float(path_cfg.get("weight", 0.0))


def _get_path_stop_grad_mode(train_paths_cfg, path_type: str) -> str:
    path_cfg = train_paths_cfg.get(path_type, {})
    if not isinstance(path_cfg, Mapping):
        return "pairs"
    return str(path_cfg.get("stop_grad", "pairs"))


def _sample_source(
    source_kind: str,
    available: list[str],
    M: int,
    target_kind: str,
    encoder_weights: dict[str, float],
    rng: random.Random,
) -> list[str]:
    """Sample an encoded subset uniformly from the feasible subsets."""
    A = len(available)
    if source_kind == "all":
        return list(available)
    if source_kind == "one":
        return [_weighted_choice(rng, available, encoder_weights)]
    if source_kind == "any":
        # any_cross excludes the full subset so a target always remains.
        hi = (1 << A) - 1 if target_kind == "cross" else (1 << A)
        mask = rng.randrange(1, hi)
        return [available[i] for i in range(A) if mask & (1 << i)]
    if source_kind == "almost":
        if A == M:
            if target_kind == "cross":
                choice = rng.randrange(M)
                return [available[i] for i in range(A) if i != choice]
            choice = rng.randrange(M + 1)
            if choice == M:
                return list(available)
            return [available[i] for i in range(A) if i != choice]
        # At A=M-1, available is already a drop-one subset.
        return list(available)


def _sample_target(
    target_kind: str,
    encoded: list[str],
    available: list[str],
    decoder_weights: dict[str, float],
    prior_weight: float,
    rng: random.Random,
) -> str:
    if target_kind == "prior":
        return "prior"
    if target_kind == "decoder":
        return _weighted_choice(rng, list(available), decoder_weights)
    if target_kind == "self":
        return _weighted_choice(rng, encoded, decoder_weights)
    if target_kind == "cross":
        remaining = [m for m in available if m not in encoded]
        return _weighted_choice(rng, remaining, decoder_weights)
    if target_kind == "any":
        # Baseline paths draw among the prior and available decoders.
        targets = ["prior"] + list(available)
        weights = {
            "prior": float(prior_weight),
            **{m: float(decoder_weights.get(m, 1.0)) for m in available},
        }
        return _weighted_choice(rng, targets, weights)


@dataclass
class RoutingPolicy:
    """Validated routing configuration with an epoch-local deterministic RNG."""

    modalities: list[str]
    path_weights: dict[str, float]
    stop_grad_pairs: set[tuple[str, str]]
    path_stop_grad_modes: dict[str, str]
    prior_weight: float
    encoder_weights: dict[str, float]
    decoder_weights: dict[str, float]
    rng: random.Random

    @classmethod
    def from_config(cls, routing_cfg, *, seed: int) -> "RoutingPolicy":
        modalities = list(routing_cfg.modalities)
        if not modalities:
            raise ValueError("cfg.model.routing.modalities must not be empty.")
        if len(set(modalities)) != len(modalities):
            raise ValueError(f"cfg.model.routing.modalities contains duplicates: {modalities}.")
        train_paths_cfg = routing_cfg.get("train_paths", {}) or {}
        path_weights = {path: _get_path_weight(train_paths_cfg, path) for path in TRAIN_PATH_TYPES}
        path_stop_grad_modes = {
            path: _get_path_stop_grad_mode(train_paths_cfg, path) for path in TRAIN_PATH_TYPES
        }

        unknown_paths = [key for key in train_paths_cfg if key not in TRAIN_PATH_TYPES]
        if unknown_paths:
            raise ValueError(
                "Unknown training path name(s) in cfg.model.routing.train_paths: "
                f"{sorted(unknown_paths)}. Valid names: {sorted(TRAIN_PATH_TYPES)}."
            )
        if path_weights.get("all_cross", 0.0) > 0.0:
            raise ValueError("all_cross has no unencoded target; its weight must be zero.")
        invalid_path_weights = {
            name: weight for name, weight in path_weights.items() if not math.isfinite(weight) or weight < 0.0
        }
        if invalid_path_weights:
            raise ValueError(f"Training path weights must be finite and non-negative: {invalid_path_weights}.")
        invalid_stop_grad_modes = {
            path: mode for path, mode in path_stop_grad_modes.items() if mode not in STOP_GRAD_MODES
        }
        if invalid_stop_grad_modes:
            raise ValueError(
                f"Path stop-gradient modes must be one of {STOP_GRAD_MODES}: {invalid_stop_grad_modes}."
            )

        prior_weight = float(routing_cfg.get("prior", {}).get("weight", 1.0))
        if not math.isfinite(prior_weight) or prior_weight < 0.0:
            raise ValueError(f"Prior routing weight must be finite and non-negative, got {prior_weight}.")
        path_weights["almost_prior"] *= prior_weight
        if not any(weight > 0.0 for weight in path_weights.values()):
            raise ValueError("At least one training path must have a positive effective weight.")
        encoders_cfg = routing_cfg.get("encoders", {})
        encoder_weights = {
            modality: float(encoders_cfg[modality].get("weight", 1.0))
            for modality in encoders_cfg
            if encoders_cfg[modality] is not None and "weight" in encoders_cfg[modality]
        }
        decoders_cfg = routing_cfg.get("decoders", {})
        decoder_weights = {
            modality: float(decoders_cfg[modality].get("weight", 1.0))
            for modality in decoders_cfg
            if decoders_cfg[modality] is not None and "weight" in decoders_cfg[modality]
        }
        invalid_modality_weights = {
            **{
                f"encoder:{name}": weight
                for name, weight in encoder_weights.items()
                if not math.isfinite(weight) or weight < 0.0
            },
            **{
                f"decoder:{name}": weight
                for name, weight in decoder_weights.items()
                if not math.isfinite(weight) or weight < 0.0
            },
        }
        if invalid_modality_weights:
            raise ValueError(f"Modality routing weights must be finite and non-negative: {invalid_modality_weights}.")
        return cls(
            modalities=modalities,
            path_weights=path_weights,
            stop_grad_pairs=_resolve_stop_grad_pairs(routing_cfg, modalities),
            path_stop_grad_modes=path_stop_grad_modes,
            prior_weight=prior_weight,
            encoder_weights=encoder_weights,
            decoder_weights=decoder_weights,
            rng=random.Random(seed),
        )

    def sample(self, available: list[str], device: torch.device) -> tuple[str, list[str], str, list[str]]:
        """Sample one route on rank 0 and broadcast its compact representation."""
        num_modalities = len(self.modalities)
        if misc.is_dist_avail_and_initialized():
            buffer = torch.zeros(num_modalities + 3, device=device, dtype=torch.long)
            if misc.get_rank() == 0:
                path_type, encoded, target = self._sample_local(available)
                buffer[0] = TRAIN_PATH_TYPES.index(path_type)
                buffer[1] = len(encoded)
                for index, modality in enumerate(encoded):
                    buffer[2 + index] = self.modalities.index(modality)
                buffer[2 + len(encoded)] = num_modalities if target == "prior" else self.modalities.index(target)
            torch.distributed.broadcast(buffer, 0)
            path_type = TRAIN_PATH_TYPES[int(buffer[0].item())]
            num_encoded = int(buffer[1].item())
            encoded = [self.modalities[int(buffer[2 + index].item())] for index in range(num_encoded)]
            target_index = int(buffer[2 + num_encoded].item())
            target = "prior" if target_index == num_modalities else self.modalities[target_index]
        else:
            path_type, encoded, target = self._sample_local(available)

        stop_grad_mode = self.path_stop_grad_modes[path_type]
        if stop_grad_mode == "all":
            stop_grad = list(encoded)
        elif stop_grad_mode == "target":
            stop_grad = [target] if target in encoded else []
        elif stop_grad_mode == "none":
            stop_grad = []
        else:
            stop_grad = [modality for modality in encoded if (modality, target) in self.stop_grad_pairs]
        return path_type, encoded, target, stop_grad

    def _sample_local(self, available: list[str]) -> tuple[str, list[str], str]:
        M = len(self.modalities)
        A = len(available)
        feasible = {
            path: weight
            for path, weight in self.path_weights.items()
            if weight > 0 and _path_feasible(*_split_path_name(path), A, M)
        }

        names = list(feasible)
        weights = list(feasible.values())
        path_type = self.rng.choices(names, weights=weights, k=1)[0]
        src_kind, tgt_kind = _split_path_name(path_type)
        encoded = _sample_source(src_kind, available, M, tgt_kind, self.encoder_weights, self.rng)
        target = _sample_target(tgt_kind, encoded, available, self.decoder_weights, self.prior_weight, self.rng)
        return path_type, encoded, target


def _prefix_training_metrics(
    metrics: Mapping[str, float | int],
    *,
    tgt_modality: str,
    loss_value: float | None = None,
) -> dict[str, float]:
    """Namespace target metrics while keeping ``_global_`` series unscoped."""
    scope = "prior" if tgt_modality == "prior" else tgt_modality
    prefixed: dict[str, float] = {}
    for name, value in metrics.items():
        if name.startswith("_global_"):
            prefixed[f"train/{name[len('_global_') :]}"] = float(value)
        else:
            prefixed[f"train/{scope}/{name}"] = float(value)
    if loss_value is not None:
        prefixed.setdefault(f"train/{scope}/loss", float(loss_value))
    return prefixed
