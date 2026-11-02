"""Compile routing configuration and draw synchronized training routes."""

from collections.abc import Mapping
from dataclasses import dataclass
import math
import random
from types import MappingProxyType

import torch

import muni.util.misc as misc

# Source kinds choose all, one, any non-empty, or a near-full subset. Target
# kinds choose prior, any available decoder, self, cross, or their weighted
# union. all_cross is the sole structurally impossible grid entry.
SOURCE_KINDS: tuple[str, ...] = ("all", "one", "any", "almost")
TARGET_KINDS: tuple[str, ...] = ("any", "decoder", "prior", "self", "cross")

TRAIN_PATH_TYPES: tuple[str, ...] = tuple(f"{s}_{t}" for s in SOURCE_KINDS for t in TARGET_KINDS)
STOP_GRAD_MODES: tuple[str, ...] = ("pairs", "none", "target", "all")


def _path_feasible(source_kind: str, target_kind: str, available_count: int, modality_count: int) -> bool:
    if available_count == 0:
        return False
    if source_kind == "almost" and available_count < modality_count - 1:
        return False
    if target_kind == "cross":
        if source_kind == "all":
            return False
        if available_count < 2:
            return False
        if source_kind == "almost" and available_count < modality_count:
            return False
    return True


def _weighted_choice(rng: random.Random, items: list[str], weights: Mapping[str, float]) -> str:
    """Draw by configured weights, using uniform choice when all weights are zero."""
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


def _sample_source(
    source_kind: str,
    available: list[str],
    modality_count: int,
    target_kind: str,
    encoder_weights: Mapping[str, float],
    rng: random.Random,
) -> list[str]:
    """Choose sources: weighted for ``one``, uniform over subsets for ``any/almost``."""
    available_count = len(available)
    if source_kind == "all":
        return list(available)
    if source_kind == "one":
        return [_weighted_choice(rng, available, encoder_weights)]
    if source_kind == "any":
        # any_cross excludes the full subset so a target always remains.
        hi = (1 << available_count) - 1 if target_kind == "cross" else (1 << available_count)
        mask = rng.randrange(1, hi)
        return [available[i] for i in range(available_count) if mask & (1 << i)]
    if source_kind == "almost":
        if available_count == modality_count:
            if target_kind == "cross":
                choice = rng.randrange(modality_count)
                return [available[i] for i in range(available_count) if i != choice]
            choice = rng.randrange(modality_count + 1)
            if choice == modality_count:
                return list(available)
            return [available[i] for i in range(available_count) if i != choice]
        # With one modality missing, the available set is the only eligible subset.
        return list(available)


def _sample_target(
    target_kind: str,
    encoded: list[str],
    available: list[str],
    decoder_weights: Mapping[str, float],
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


@dataclass(frozen=True)
class _Route:
    wire_id: int
    name: str
    source_kind: str
    target_kind: str
    weight: float
    stop_grad_mode: str


@dataclass(frozen=True)
class _RouteDistribution:
    routes: tuple[_Route, ...]
    weights: tuple[float, ...]


@dataclass(frozen=True)
class RoutingPolicy:
    """Compiled routing distributions with an epoch-local deterministic RNG."""

    modalities: tuple[str, ...]
    routes: tuple[_Route, ...]
    distributions: tuple[_RouteDistribution, ...]
    stop_grad_pairs: frozenset[tuple[str, str]]
    prior_weight: float
    encoder_weights: Mapping[str, float]
    decoder_weights: Mapping[str, float]
    rng: random.Random

    @classmethod
    def from_config(cls, routing_cfg, *, seed: int) -> "RoutingPolicy":
        modalities = list(routing_cfg.modalities)
        if not modalities:
            raise ValueError("cfg.model.routing.modalities must not be empty.")
        if len(set(modalities)) != len(modalities):
            raise ValueError(f"cfg.model.routing.modalities contains duplicates: {modalities}.")
        train_paths_cfg = routing_cfg.get("train_paths", {}) or {}
        routes = []
        for wire_id, path in enumerate(TRAIN_PATH_TYPES):
            path_cfg = train_paths_cfg.get(path, {})
            if path_cfg is None:
                weight = 0.0
            elif isinstance(path_cfg, (float, int)):
                weight = float(path_cfg)
            else:
                weight = float(path_cfg.get("weight", 0.0))
            stop_grad_mode = (
                str(path_cfg.get("stop_grad", "pairs"))
                if isinstance(path_cfg, Mapping) else "pairs"
            )
            source_kind, target_kind = path.split("_", 1)
            routes.append(_Route(wire_id, path, source_kind, target_kind, weight, stop_grad_mode))

        unknown_paths = [key for key in train_paths_cfg if key not in TRAIN_PATH_TYPES]
        if unknown_paths:
            raise ValueError(
                "Unknown training path name(s) in cfg.model.routing.train_paths: "
                f"{sorted(unknown_paths)}. Valid names: {sorted(TRAIN_PATH_TYPES)}."
            )
        if any(route.name == "all_cross" and route.weight > 0.0 for route in routes):
            raise ValueError("all_cross has no unencoded target; its weight must be zero.")
        invalid_path_weights = {
            route.name: route.weight for route in routes if not math.isfinite(route.weight) or route.weight < 0.0
        }
        if invalid_path_weights:
            raise ValueError(f"Training path weights must be finite and non-negative: {invalid_path_weights}.")
        invalid_stop_grad_modes = {
            route.name: route.stop_grad_mode for route in routes if route.stop_grad_mode not in STOP_GRAD_MODES
        }
        if invalid_stop_grad_modes:
            raise ValueError(
                f"Path stop-gradient modes must be one of {STOP_GRAD_MODES}: {invalid_stop_grad_modes}."
            )

        prior_weight = float(routing_cfg.get("prior", {}).get("weight", 1.0))
        if not math.isfinite(prior_weight) or prior_weight < 0.0:
            raise ValueError(f"Prior routing weight must be finite and non-negative, got {prior_weight}.")
        # almost_prior uses prior.weight. Other *_prior paths keep their own weights.
        routes = tuple(
            _Route(
                route.wire_id, route.name, route.source_kind, route.target_kind,
                route.weight * prior_weight, route.stop_grad_mode,
            )
            if route.name == "almost_prior" else route
            for route in routes
        )
        if not any(route.weight > 0.0 for route in routes):
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
        distributions = []
        for available_count in range(len(modalities) + 1):
            feasible = tuple(
                route for route in routes
                if route.weight > 0
                and _path_feasible(route.source_kind, route.target_kind, available_count, len(modalities))
            )
            distributions.append(_RouteDistribution(feasible, tuple(route.weight for route in feasible)))
        return cls(
            modalities=tuple(modalities),
            routes=routes,
            distributions=tuple(distributions),
            stop_grad_pairs=frozenset(_resolve_stop_grad_pairs(routing_cfg, modalities)),
            prior_weight=prior_weight,
            encoder_weights=MappingProxyType(encoder_weights),
            decoder_weights=MappingProxyType(decoder_weights),
            rng=random.Random(seed),
        )

    def sample(self, available: list[str], device: torch.device) -> tuple[str, list[str], str, list[str]]:
        """Sample one route on rank 0 and broadcast its compact representation."""
        num_modalities = len(self.modalities)
        if misc.is_dist_avail_and_initialized():
            buffer = torch.zeros(num_modalities + 3, device=device, dtype=torch.long)
            if misc.get_rank() == 0:
                route, encoded, target = self._sample_local(available)
                buffer[0] = route.wire_id
                buffer[1] = len(encoded)
                for index, modality in enumerate(encoded):
                    buffer[2 + index] = self.modalities.index(modality)
                buffer[2 + len(encoded)] = num_modalities if target == "prior" else self.modalities.index(target)
            torch.distributed.broadcast(buffer, 0)
            route = self.routes[int(buffer[0].item())]
            num_encoded = int(buffer[1].item())
            encoded = [self.modalities[int(buffer[2 + index].item())] for index in range(num_encoded)]
            target_index = int(buffer[2 + num_encoded].item())
            target = "prior" if target_index == num_modalities else self.modalities[target_index]
        else:
            route, encoded, target = self._sample_local(available)

        stop_grad_mode = route.stop_grad_mode
        if stop_grad_mode == "all":
            stop_grad = list(encoded)
        elif stop_grad_mode == "target":
            stop_grad = [target] if target in encoded else []
        elif stop_grad_mode == "none":
            stop_grad = []
        else:
            stop_grad = [modality for modality in encoded if (modality, target) in self.stop_grad_pairs]
        return route.name, encoded, target, stop_grad

    def _sample_local(self, available: list[str]) -> tuple[_Route, list[str], str]:
        distribution = self.distributions[len(available)]
        route = self.rng.choices(distribution.routes, weights=distribution.weights, k=1)[0]
        encoded = _sample_source(
            route.source_kind, available, len(self.modalities), route.target_kind, self.encoder_weights, self.rng,
        )
        target = _sample_target(
            route.target_kind, encoded, available, self.decoder_weights, self.prior_weight, self.rng,
        )
        return route, encoded, target


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
