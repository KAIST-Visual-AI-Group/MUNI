"""Independent phases used by the public evaluation-suite facade."""

from dataclasses import dataclass
from typing import Any
import random

import numpy as np

import torch
import torch.distributed as dist

from muni.config_utils import instantiate_from_config
from muni.evaluation.sampling import decoder_sample_kwargs
import muni.util.misc as misc
from muni.util.misc import print_on_main as print
from muni.evaluation.data import (
    EvaluationFeatureCache,
    iter_eval_batch_matches,
    load_aligned_eval_batch,
    load_raw_gt,
    required_eval_samples,
    resolve_eval_dataset,
)


def _instantiate_nodes(nodes):
    return [instantiate_from_config(node) for node in nodes]


def _format_metrics(metrics: dict[str, float]) -> str:
    return ", ".join(
        f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}"
        for key, value in metrics.items()
    )


def _shard_num_samples(num_samples: int, rank: int, world_size: int) -> tuple[int, int]:
    samples_per_rank = -(-num_samples // world_size)
    start = rank * samples_per_rank
    return start, max(0, min(samples_per_rank, num_samples - start))


def _seed_eval_rng(eval_seed: int, rank: int, device: torch.device) -> None:
    seed = eval_seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.manual_seed(seed)


def _required_generation_samples(modality_cfg) -> int:
    metric_cfgs = modality_cfg.get("metrics", []) or []
    logger_cfgs = modality_cfg.get("loggers", []) or []
    vis_limit = max(
        [int(node.get("params", {}).get("num_vis_samples", 0)) for node in logger_cfgs]
        + [0]
    )
    metric_limits = [
        int(node.get("params", {}).get("num_samples", 256)) for node in metric_cfgs
    ]
    return max(metric_limits + [int(vis_limit), 0])


def _generation_batch_size(metric_cfgs, num_samples: int) -> int:
    if metric_cfgs:
        return max(
            1,
            max(
                int(node.get("params", {}).get("batch_size", 128))
                for node in metric_cfgs
            ),
        )
    return max(1, min(128, num_samples)) if num_samples > 0 else 0


def _normalize_modalities_key(required_modalities: list[str]) -> tuple[str, ...]:
    return tuple(sorted(set(required_modalities)))


def _report_metrics(values, log_writer, step, prefix, label, *, numeric_only=False):
    if values:
        print(f"{label} {_format_metrics(values)}")
        if log_writer is not None:
            for key, value in values.items():
                if not numeric_only or isinstance(value, (int, float)):
                    log_writer.add_scalar(f"{prefix}/{key}", value, step)


def _run_metrics_and_loggers(ctx, node, phase, name, preds, targets, **logger_kwargs):
    metrics = _instantiate_nodes(node.get("metrics", []))
    loggers = _instantiate_nodes(node.get("loggers", []))
    task_name = f"{phase}/{name}"
    values: dict[str, float] = {}
    for metric in metrics:
        metric.update(preds, targets, **logger_kwargs)
        values.update(metric.compute())
    _report_metrics(
        values, ctx.log_writer, ctx.step, f"metrics/{task_name}",
        f"[Eval: {phase.capitalize()}: {name}]",
    )
    if ctx.log_writer is not None:
        for logger in loggers:
            logger.log(
                ctx.log_writer, ctx.step, preds, targets,
                f"viz/{task_name}", **logger_kwargs,
            )
    return values


@dataclass(frozen=True)
class EvaluationContext:
    model: Any
    data_loader: Any
    eval_cfg: Any
    eval_datasets_cfg: Any
    log_writer: Any
    step: int
    device: torch.device
    eval_seed: int
    rank: int
    world_size: int
    is_main: bool
    eval_batch_cache: dict[tuple[str, ...], dict]
    prior_outputs: dict[str, Any]
    decoder_sample_kwargs: dict[str, dict[str, Any]]
    feature_cache: EvaluationFeatureCache


def run_generation_phase(ctx: EvaluationContext):
    if "generation" not in ctx.eval_cfg or not ctx.is_main:
        return None

    results = {}
    for modality, modality_cfg in ctx.eval_cfg.generation.items():
        if modality_cfg is None:
            continue
        metric_cfgs = modality_cfg.get("metrics", []) or []
        logger_cfgs = modality_cfg.get("loggers", []) or []
        num_samples = _required_generation_samples(modality_cfg)
        predictions = ctx.prior_outputs.get(modality)
        is_empty = (
            predictions is None
            or (isinstance(predictions, torch.Tensor) and predictions.numel() == 0)
            or (isinstance(predictions, list) and not predictions)
        )
        if num_samples <= 0 or is_empty:
            results[modality] = {}
            continue

        values: dict[str, float] = {}
        metrics = _instantiate_nodes(metric_cfgs)
        loggers = _instantiate_nodes(logger_cfgs)
        for metric_cfg, metric in zip(metric_cfgs, metrics):
            metric_num_samples = int(metric_cfg.get("params", {}).get("num_samples", 0))
            metric_predictions = predictions
            if metric_num_samples > 0:
                if isinstance(predictions, torch.Tensor):
                    metric_predictions = predictions[
                        : min(metric_num_samples, int(predictions.shape[0]))
                    ]
                elif isinstance(predictions, list):
                    metric_predictions = predictions[:metric_num_samples]
            metric.update(metric_predictions)
            values.update(metric.compute())

        _report_metrics(
            values, ctx.log_writer, ctx.step, f"metrics/generation/{modality}",
            f"[Eval: Generation] {modality}:", numeric_only=True,
        )
        results[modality] = values
        if ctx.log_writer is not None:
            has_predictions = (
                isinstance(predictions, torch.Tensor) and predictions.numel() > 0
            ) or (isinstance(predictions, list) and bool(predictions))
            if has_predictions:
                for logger in loggers:
                    logger.log(
                        ctx.log_writer,
                        ctx.step,
                        predictions,
                        None,
                        f"viz/generation/{modality}",
                    )
    return results


def _resolve_phase_batch(ctx: EvaluationContext, node, required):
    """Resolve aligned features, then the cached or freshly scanned loader batch."""
    dataset = resolve_eval_dataset(ctx.eval_datasets_cfg, *required)
    batch, keys = None, None
    if dataset is not None:
        batch, keys = load_aligned_eval_batch(
            dataset, required, required_eval_samples(node), ctx.device,
            cache=ctx.feature_cache,
        )
    if batch is None:
        batch = ctx.eval_batch_cache.get(_normalize_modalities_key(required))
    if batch is None:
        batch = next(iter_eval_batch_matches(
            ctx.data_loader, ctx.device, [_normalize_modalities_key(required)]
        ), (None, None))[1]
    return batch, keys, dataset


def _run_paired_phase(ctx: EvaluationContext, phase: str):
    if phase not in ctx.eval_cfg or not ctx.is_main:
        return None
    reconstruction = phase == "reconstruction"
    nodes = (
        ctx.eval_cfg[phase].items() if reconstruction
        else ((None, node) for node in ctx.eval_cfg[phase])
    )
    results = {}
    with torch.random.fork_rng(
        devices=[ctx.device] if ctx.device.type == "cuda" else []
    ):
        _seed_eval_rng(ctx.eval_seed, ctx.rank, ctx.device)
        for modality, node in nodes:
            if reconstruction:
                if node is None:
                    continue
                target, sources, name = modality, [modality], modality
            else:
                target = node["tgt"]
                sources = list(node["srcs"]) if "srcs" in node else [node["src"]]
                name = f"{'-'.join(sources)}_to_{target}"
            batch, keys, dataset = _resolve_phase_batch(
                ctx, node, sources if reconstruction else sources + [target]
            )
            raw_source, raw_target = None, None
            if batch is not None and keys is not None:
                if not reconstruction:
                    raw_source = {
                        source: load_raw_gt(dataset, source, keys) for source in sources
                    }
                    if len(sources) == 1:
                        raw_source = raw_source[sources[0]]
                raw_target = load_raw_gt(dataset, target, keys)
            if batch is None:
                print(f"[Eval: {phase.capitalize()}] {name}: skipped (no batch with required modalities)")
                continue
            predictions = ctx.model.conditional_sample(
                batch, sources, target,
                mode="reconstruct" if reconstruction else (
                    "direct" if len(sources) == 1 else "aggregate"
                ),
                decoder_sample_kwargs=decoder_sample_kwargs(
                    ctx.decoder_sample_kwargs, target, node.get("sample_kwargs")
                ),
            )
            kwargs = {"raw_target_gt": raw_target}
            if not reconstruction:
                kwargs = {
                    "sources": batch[sources[0]] if len(sources) == 1 else {
                        source: batch[source] for source in sources
                    },
                    "src_modalities": sources,
                    "tgt_modality": target,
                    "raw_source_gt": raw_source,
                    "raw_target_gt": raw_target,
                }
            results[name] = _run_metrics_and_loggers(
                ctx, node, phase, name, predictions, batch[target], **kwargs
            )
    return results


def run_coherence_phase(ctx: EvaluationContext):
    if "coherence" not in ctx.eval_cfg or not ctx.is_main:
        return None

    coherence_cfg = ctx.eval_cfg.coherence
    heads_cfg = coherence_cfg.get("heads", {}) or {}
    head_modalities = [
        modality for modality, value in heads_cfg.items() if value is not None
    ]
    if not head_modalities:
        return None

    pivot_modality = str(coherence_cfg["pivot_modality"])
    num_samples = int(coherence_cfg.get("num_samples", 1024))
    batch_size = int(coherence_cfg.get("batch_size", 128))
    pivot = ctx.prior_outputs.get(pivot_modality)
    source_outputs = {
        modality: ctx.prior_outputs.get(modality) for modality in head_modalities
    }
    source_tensors: dict[str, torch.Tensor] = {}
    missing: list[str] = []
    for modality, value in source_outputs.items():
        if not isinstance(value, torch.Tensor) or value.numel() == 0:
            missing.append(modality)
        else:
            source_tensors[modality] = value
    if not isinstance(pivot, torch.Tensor) or pivot.numel() == 0 or missing:
        print(
            f"[Eval: Coherence] skipped: pivot '{pivot_modality}' missing={pivot is None}, head_missing={missing}"
        )
        return None

    available = min(
        int(pivot.shape[0]), *(int(source_tensors[m].shape[0]) for m in head_modalities)
    )
    effective_count = min(num_samples, available)
    if effective_count < num_samples:
        print(
            f"[Eval: Coherence] truncating to {effective_count} samples (plan {num_samples} > available {available})"
        )

    metrics_by_head = {
        modality: _instantiate_nodes(heads_cfg[modality].get("metrics", []) or [])
        for modality in head_modalities
    }
    for start in range(0, effective_count, batch_size):
        end = min(start + batch_size, effective_count)
        batch_pivot = pivot[start:end]
        batch_sources = {
            pivot_modality: batch_pivot,
            **{
                modality: source_tensors[modality][start:end]
                for modality in head_modalities
            },
        }
        for modality in head_modalities:
            for metric in metrics_by_head[modality]:
                metric.update(batch_pivot, targets=None, sources=batch_sources)

    results = {}
    for modality in head_modalities:
        values: dict[str, float] = {}
        for metric in metrics_by_head[modality]:
            values.update(metric.compute())
        _report_metrics(
            values, ctx.log_writer, ctx.step, f"metrics/coherence/{modality}",
            f"[Eval: Coherence: {modality}]",
        )
        results[modality] = values
    return results


def run_information_phase(ctx: EvaluationContext):
    if "information" not in ctx.eval_cfg:
        return None

    results = {}
    for modality, evaluator_cfg in ctx.eval_cfg.information.items():
        if evaluator_cfg is None:
            continue
        evaluator = instantiate_from_config(evaluator_cfg)
        _, local_count = _shard_num_samples(
            evaluator.num_samples, ctx.rank, ctx.world_size
        )
        with torch.random.fork_rng(
            devices=[ctx.device] if ctx.device.type == "cuda" else []
        ):
            _seed_eval_rng(ctx.eval_seed, ctx.rank, ctx.device)
            shard_stats = evaluator.evaluate(
                ctx.model,
                ctx.data_loader,
                device=ctx.device,
                modality=modality,
                num_samples=local_count,
            )
        if ctx.world_size > 1:
            for key in ("cost_minus_entropy_per_dim_sum", "count"):
                dist.all_reduce(shard_stats[key], op=dist.ReduceOp.SUM)
        if not ctx.is_main:
            continue

        count = int(shard_stats["count"].item())
        values = {
            "DenoisingCostMinusEntropy_per_dim": float("nan")
            if count == 0
            else (shard_stats["cost_minus_entropy_per_dim_sum"] / float(count)).item()
        }
        _report_metrics(
            values, ctx.log_writer, ctx.step, f"information/{modality}",
            f"[Eval: Information] {modality}:",
        )
        results[modality] = values
    return results if ctx.is_main else None


def run_evaluation_phases(ctx: EvaluationContext) -> dict[str, dict]:
    """Run rank-appropriate phases in the stable public order."""
    results: dict[str, dict] = {}
    for name, phase in (
        ("generation", run_generation_phase),
        ("reconstruction", lambda ctx: _run_paired_phase(ctx, "reconstruction")),
        ("translation", lambda ctx: _run_paired_phase(ctx, "translation")),
        ("coherence", run_coherence_phase),
        ("information", run_information_phase),
    ):
        phase_result = phase(ctx)
        if phase_result is not None:
            results[name] = phase_result
    return results
