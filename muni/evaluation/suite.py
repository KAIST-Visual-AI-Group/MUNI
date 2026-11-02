"""Execute prepared generation, paired, coherence, and information tasks."""

from dataclasses import dataclass
from typing import Any
import random
import inspect

import numpy as np

import torch
import torch.distributed as dist

from muni.config_utils import get_obj_from_str, instantiate_from_config
from muni.evaluation.sampling import collect_decoder_outputs, decoder_sample_kwargs
import muni.util.misc as misc
from muni.util.misc import print_on_main as print
from muni.evaluation.data import (
    EvaluationFeatureCache,
    RetainedPrefix,
    load_aligned_eval_batch,
    load_raw_gt,
    required_eval_samples,
    resolve_eval_dataset,
    _explicit_eval_samples,
    _select_loader_batches,
    _slice_eval_payload,
)


def _instantiate_nodes(nodes, device):
    components = []
    for node in nodes:
        params = dict(node.get("params", {}) or {})
        constructor = get_obj_from_str(node["target"])
        parameters = inspect.signature(constructor).parameters
        # These options also drive suite planning, even for components that do
        # not consume them. Keep consumed options and reject every other typo.
        for key in ("num_samples", "batch_size", "num_vis_samples"):
            if key not in parameters:
                params.pop(key, None)
        if "device" in parameters:
            if params.get("device") is None:
                params["device"] = device
        components.append(constructor(**params))
    return components


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
    """Planner demand: metric counts default to 256; logger counts to their constructor default."""
    metric_cfgs = modality_cfg.get("metrics", []) or []
    logger_cfgs = modality_cfg.get("loggers", []) or []
    vis_limit = 0
    for node in logger_cfgs:
        params = node.get("params", {}) or {}
        if "num_vis_samples" in params:
            count = params["num_vis_samples"]
        else:
            constructor = get_obj_from_str(node["target"])
            parameter = inspect.signature(constructor).parameters.get("num_vis_samples")
            count = (
                parameter.default
                if parameter is not None and parameter.default is not inspect.Parameter.empty
                else 0
            )
        vis_limit = max(vis_limit, int(count))
    metric_limits = [
        int(node.get("params", {}).get("num_samples", 256)) for node in metric_cfgs
    ]
    return max(metric_limits + [int(vis_limit), 0])


def _generation_batch_size(metric_cfgs, num_samples: int) -> int:
    """Decode chunk: the largest metric ``batch_size`` (default 128), else at most 128."""
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


def _run_metrics_and_loggers(ctx, task, preds, targets, **logger_kwargs):
    metrics = _instantiate_nodes(task.metrics, ctx.device)
    loggers = _instantiate_nodes(task.loggers, ctx.device)
    phase, name = task.phase, task.name
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
class GenerationTask:
    modality: str
    metrics: Any
    loggers: Any
    num_samples: int


@dataclass(frozen=True)
class CoherenceTask:
    pivot: str
    heads: dict
    num_samples: int
    batch_size: int


@dataclass(frozen=True)
class PairedTask:
    phase: str
    name: str
    sources: list[str]
    target: str
    mode: str
    required: list[str]
    prefix: RetainedPrefix
    loader_cap: int
    packed_cap: int
    metrics: Any
    loggers: Any
    sample_kwargs: Any


def prepare_paired_tasks(eval_cfg):
    tasks = []
    prefixes = {}

    def retain_prefix(required, loader_cap):
        key = _normalize_modalities_key(required)
        if key not in prefixes:
            prefixes[key] = RetainedPrefix(modalities=key, num_samples=loader_cap)
        else:
            prefixes[key].num_samples = max(prefixes[key].num_samples, loader_cap)
        return prefixes[key]

    for modality, node in eval_cfg.get("reconstruction", {}).items():
        if node is None:
            continue
        required = [modality]
        loader_cap = _explicit_eval_samples(node)
        tasks.append(PairedTask(
            phase="reconstruction",
            name=modality,
            sources=required,
            target=modality,
            mode="reconstruct",
            required=required,
            prefix=retain_prefix(required, loader_cap),
            loader_cap=loader_cap,
            packed_cap=required_eval_samples(node),
            metrics=node.get("metrics", []),
            loggers=node.get("loggers", []),
            sample_kwargs=node.get("sample_kwargs"),
        ))

    for node in eval_cfg.get("translation", []):
        sources = list(node["srcs"]) if "srcs" in node else [node["src"]]
        target = node["tgt"]
        required = sources + [target]
        loader_cap = _explicit_eval_samples(node)
        tasks.append(PairedTask(
            phase="translation",
            name=f"{'-'.join(sources)}_to_{target}",
            sources=sources,
            target=target,
            mode="direct" if len(sources) == 1 else "aggregate",
            required=required,
            prefix=retain_prefix(required, loader_cap),
            loader_cap=loader_cap,
            packed_cap=required_eval_samples(node),
            metrics=node.get("metrics", []),
            loggers=node.get("loggers", []),
            sample_kwargs=node.get("sample_kwargs"),
        ))
    return tasks, list(prefixes.values())


@dataclass
class InformationJob:
    """Information evaluator for one modality.

    Over a one-shot iterator it is initialized before loader retention, which needs its
    local sample count; otherwise it is initialized inside the information phase.
    """
    modality: str
    config: Any
    source: Any
    evaluator: Any = None
    local_count: int = 0

    def initialize(self, rank, world_size):
        self.evaluator = instantiate_from_config(self.config)
        _, self.local_count = _shard_num_samples(
            self.evaluator.num_samples, rank, world_size
        )


@dataclass(frozen=True)
class EvaluationContext:
    model: Any
    enabled_phases: frozenset[str]
    eval_datasets_cfg: Any
    log_writer: Any
    step: int
    device: torch.device
    eval_seed: int
    rank: int
    world_size: int
    is_main: bool
    generation_tasks: list[GenerationTask]
    coherence_task: CoherenceTask | None
    paired_tasks: list[PairedTask]
    information_jobs: list[InformationJob]
    prior_outputs: dict[str, Any]
    decoder_sample_kwargs: dict[str, dict[str, Any]]
    feature_cache: EvaluationFeatureCache
    batch_size: int = 128


def run_generation_phase(ctx: EvaluationContext):
    if "generation" not in ctx.enabled_phases or not ctx.is_main:
        return None

    results = {}
    for task in ctx.generation_tasks:
        modality = task.modality
        metric_cfgs, logger_cfgs = task.metrics, task.loggers
        num_samples = task.num_samples
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
        metrics = _instantiate_nodes(metric_cfgs, ctx.device)
        loggers = _instantiate_nodes(logger_cfgs, ctx.device)
        for metric_cfg, metric in zip(metric_cfgs, metrics):
            # Unlike planner demand (default 256), an omitted count consumes every decoded row.
            metric_num_samples = int(metric_cfg.get("params", {}).get("num_samples", 0))
            metric_predictions = predictions
            if metric_num_samples > 0 and isinstance(predictions, (torch.Tensor, list)):
                metric_predictions = predictions[:metric_num_samples]
            metric.update(metric_predictions)
            values.update(metric.compute())

        _report_metrics(
            values, ctx.log_writer, ctx.step, f"metrics/generation/{modality}",
            f"[Eval: Generation] {modality}:", numeric_only=True,
        )
        results[modality] = values
        if ctx.log_writer is not None and isinstance(predictions, (torch.Tensor, list)):
            for logger in loggers:
                logger.log(
                    ctx.log_writer,
                    ctx.step,
                    predictions,
                    None,
                    f"viz/generation/{modality}",
                )
    return results


def _resolve_phase_batch(ctx: EvaluationContext, task):
    """Select a packed cohort or retained loader prefix without merging payloads."""
    dataset = resolve_eval_dataset(ctx.eval_datasets_cfg, *task.required)
    if dataset is not None:
        batch, keys = load_aligned_eval_batch(
            dataset, task.required, task.packed_cap, ctx.device,
            cache=ctx.feature_cache,
        )
        if batch is not None:
            return [batch], keys, dataset
    batches = _select_loader_batches(
        task.prefix.batches, task.required, task.loader_cap, ctx.device,
    )
    return batches, None, dataset


def _run_paired_phase(ctx: EvaluationContext, phase: str):
    if phase not in ctx.enabled_phases or not ctx.is_main:
        return None
    reconstruction = phase == "reconstruction"
    tasks = (task for task in ctx.paired_tasks if task.phase == phase)
    results = {}
    with torch.random.fork_rng(
        devices=[ctx.device] if ctx.device.type == "cuda" else []
    ):
        _seed_eval_rng(ctx.eval_seed, ctx.rank, ctx.device)
        for task in tasks:
            target, sources, name = task.target, task.sources, task.name
            batches, keys, dataset = _resolve_phase_batch(ctx, task)
            raw_source, raw_target = None, None
            if batches and keys is not None:
                if not reconstruction:
                    raw_source = {
                        source: load_raw_gt(dataset, source, keys) for source in sources
                    }
                    if len(sources) == 1:
                        raw_source = raw_source[sources[0]]
                raw_target = load_raw_gt(dataset, target, keys)
            if not batches:
                print(f"[Eval: {phase.capitalize()}] {name}: skipped (no batch with required modalities)")
                continue
            mode = task.mode
            sample_kwargs = decoder_sample_kwargs(
                ctx.decoder_sample_kwargs, target, task.sample_kwargs
            )

            def prediction_batches():
                for batch in batches:
                    count = len(misc.unwrap_modality_feature(batch[sources[0]]))
                    for start in range(0, count, ctx.batch_size):
                        yield ctx.model.conditional_sample(
                            _slice_eval_payload(batch, slice(start, start + ctx.batch_size)),
                            sources, target,
                            mode=mode,
                            decoder_sample_kwargs=sample_kwargs.copy(),
                        )
            predictions = collect_decoder_outputs(prediction_batches())
            # Optional token payloads remain batch-local for inference and text logging.
            batch = {
                modality: collect_decoder_outputs(
                    misc.unwrap_modality_feature(part[modality]) for part in batches
                )
                for modality in dict.fromkeys(sources + [target])
            }
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
            kwargs["target_batches"] = [part[target] for part in batches]
            results[name] = _run_metrics_and_loggers(
                ctx, task, predictions, batch[target], **kwargs
            )
    return results


def run_coherence_phase(ctx: EvaluationContext):
    if "coherence" not in ctx.enabled_phases or not ctx.is_main:
        return None

    task = ctx.coherence_task
    if task is None:
        return None
    heads_cfg = task.heads
    head_modalities = list(heads_cfg)
    pivot_modality = task.pivot
    num_samples = task.num_samples
    batch_size = task.batch_size
    pivot = ctx.prior_outputs.get(pivot_modality)
    source_tensors: dict[str, torch.Tensor] = {}
    missing: list[str] = []
    for modality in head_modalities:
        value = ctx.prior_outputs.get(modality)
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
        modality: _instantiate_nodes(heads_cfg[modality], ctx.device)
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
    if "information" not in ctx.enabled_phases:
        return None

    results = {}
    for job in ctx.information_jobs:
        modality = job.modality
        if job.evaluator is None:
            job.initialize(ctx.rank, ctx.world_size)
        evaluator = job.evaluator
        local_count = job.local_count
        data_loader = job.source
        with torch.random.fork_rng(
            devices=[ctx.device] if ctx.device.type == "cuda" else []
        ):
            _seed_eval_rng(ctx.eval_seed, ctx.rank, ctx.device)
            shard_stats = evaluator.evaluate(
                ctx.model,
                data_loader,
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
