"""Sampling options and shared prior-batch execution for evaluation protocols."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


def _mapping(value: Mapping[str, Any] | None, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping, got {type(value).__name__}.")
    return dict(value)


def merge_sample_kwargs(*values: Mapping[str, Any] | None) -> dict[str, Any]:
    """Merge sampling mappings left-to-right without interpreting their keys."""
    merged: dict[str, Any] = {}
    for index, value in enumerate(values):
        merged.update(_mapping(value, f"sample kwargs #{index + 1}"))
    return merged


def resolve_sample_kwargs(
    cfg,
    *,
    prior_overrides: Mapping[str, Any] | None = None,
    decoder_overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Resolve ``inference`` defaults plus runtime overrides.

    ``decoder_kwargs.default`` applies to every decoder, while modality entries
    refine it. The contained option names remain adapter-owned.
    """
    inference = cfg.get("inference", {}) or {}
    prior = merge_sample_kwargs(inference.get("prior_kwargs"), prior_overrides)

    configured = _mapping(inference.get("decoder_kwargs"), "inference.decoder_kwargs")
    overrides = _mapping(decoder_overrides, "decoder overrides")
    modalities = (set(configured) | set(overrides)) - {"default"}
    configured_default = configured.get("default")
    override_default = overrides.get("default")
    default = merge_sample_kwargs(configured_default, override_default)
    decoders = {
        modality: merge_sample_kwargs(
            configured_default,
            configured.get(modality),
            override_default,
            overrides.get(modality),
        )
        for modality in modalities
    }
    if default:
        decoders["default"] = default
    return prior, decoders


def decoder_sample_kwargs(
    options: Mapping[str, Mapping[str, Any]] | None,
    modality: str,
    route_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Select generic defaults and one modality's adapter-owned options."""
    options = options or {}
    return merge_sample_kwargs(options.get(modality, options.get("default")), route_overrides)


def collect_decoder_outputs(outputs, *, empty_device=None):
    """Collect native batches; callers own per-batch dtype and placement."""
    outputs = iter(outputs)
    try:
        first = next(outputs)
    except StopIteration:
        if empty_device is None:
            return torch.empty((0,))
        return torch.empty((0,), dtype=torch.float32, device=empty_device)
    if isinstance(first, torch.Tensor):
        return torch.cat([first, *outputs], dim=0)
    if not isinstance(first, list):
        raise TypeError("Decoder batches must be tensors or lists")
    result = list(first)
    for output in outputs:
        if not isinstance(output, list):
            raise TypeError("Decoder batches must have a consistent output type")
        result.extend(output)
    return result


@torch.no_grad()
def iter_prior_decodes(model, modality, z_prior, batch_size, *, sample_kwargs=None):
    """Decode one fixed latent stream without changing modality-major RNG order."""
    for start in range(0, len(z_prior), batch_size):
        z = z_prior[start : start + batch_size]
        with torch.autocast(
            device_type=z.device.type, dtype=torch.bfloat16,
            enabled=z.device.type == "cuda",
        ):
            output = model.decode_from_prior(modality, z, sample_kwargs=sample_kwargs)
        yield output.detach() if isinstance(output, torch.Tensor) else output


def iter_prior_batches(
    model,
    modalities,
    num_samples: int,
    batch_size: int,
    device: torch.device,
    *,
    prior_sample_kwargs=None,
    decoder_options=None,
    autocast: bool = True,
):
    """Yield on-device outputs coupled by one shared prior draw per batch.

    Protocols own seeding, populations, scoring, and output placement. GMM
    deliberately disables autocast; image benchmarks use CUDA bfloat16.
    """
    for start in range(0, num_samples, batch_size):
        size = min(batch_size, num_samples - start)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=autocast and device.type == "cuda",
        ):
            z = model.sample_prior(size, device, sample_kwargs=prior_sample_kwargs)
            outputs = {}
            for modality in modalities:
                output = model.decode_from_prior(
                    modality, z,
                    sample_kwargs=decoder_sample_kwargs(decoder_options, modality),
                )
                outputs[modality] = output.detach() if isinstance(output, torch.Tensor) else output
        yield outputs


def iter_conditional_batches(
    model, features, sources, target, num_samples: int, batch_size: int, device,
    *, mode: str, sample_kwargs=None, deterministic_encode: bool = False,
):
    """Yield cohort slices and on-device predictions; callers own cohort policy."""
    for start in range(0, num_samples, batch_size):
        cohort_slice = slice(start, min(start + batch_size, num_samples))
        batch = {
            source: features[source][cohort_slice].to(device)
            for source in sources
        }
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            prediction = model.conditional_sample(
                batch, sources, target, mode=mode,
                decoder_sample_kwargs=sample_kwargs,
                deterministic_encode=deterministic_encode,
            )
        yield cohort_slice, prediction
