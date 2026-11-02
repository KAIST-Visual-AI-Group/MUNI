"""Prepare evaluation tasks and their data before executing the suite."""

import math
from collections.abc import Iterator
from dataclasses import dataclass

import torch
import torch.distributed as dist
from muni.evaluation.sampling import (
    collect_decoder_outputs,
    iter_prior_decodes,
    decoder_sample_kwargs,
    merge_sample_kwargs,
    resolve_sample_kwargs,
)
from muni.evaluation.suite import (
    EvaluationContext,
    GenerationTask,
    CoherenceTask,
    InformationJob,
    prepare_paired_tasks,
    _generation_batch_size,
    _required_generation_samples,
    _seed_eval_rng,
    run_evaluation_phases,
)
import muni.util.misc as misc
from muni.util.misc import print_on_main as print
from muni.evaluation.data import (
    EvaluationFeatureCache, RetainedPrefix, _collect_eval_loader_batches,
)
from muni.training.batching import resolve_eval_batch_size


@dataclass(frozen=True)
class PriorDecode:
    planned_count: int
    batch_size: int
    sample_kwargs: dict


def _prepare_prior_decodes(eval_cfg, decoder_kwargs):
    generation_cfg = eval_cfg.get("generation", {}) or {}
    generation = [
        GenerationTask(
            modality=modality,
            metrics=node.get("metrics", []) or [],
            loggers=node.get("loggers", []) or [],
            num_samples=_required_generation_samples(node),
        )
        for modality, node in generation_cfg.items()
        if node is not None
    ]
    coherence_cfg = eval_cfg.get("coherence", {}) or {}
    heads = {
        modality: node.get("metrics", []) or []
        for modality, node in (coherence_cfg.get("heads", {}) or {}).items()
        if node is not None
    }
    coherence = CoherenceTask(
        pivot=str(coherence_cfg["pivot_modality"]),
        heads=heads,
        num_samples=int(coherence_cfg.get("num_samples", 1024)),
        batch_size=int(coherence_cfg.get("batch_size", 128)),
    ) if heads else None
    planned_counts = {
        task.modality: task.num_samples for task in generation if task.num_samples > 0
    }
    if coherence is not None and coherence.num_samples > 0:
        for modality in [coherence.pivot, *coherence.heads]:
            planned_counts[modality] = max(
                planned_counts.get(modality, 0), coherence.num_samples
            )

    schedule = {}
    for modality, planned_count in planned_counts.items():
        node = generation_cfg.get(modality)
        sample_kwargs = decoder_sample_kwargs(decoder_kwargs, modality)
        if node is not None and node.get("sample_kwargs"):
            sample_kwargs = merge_sample_kwargs(sample_kwargs, node["sample_kwargs"])
        schedule[modality] = PriorDecode(
            planned_count=planned_count,
            batch_size=(
                _generation_batch_size(node.get("metrics", []) or [], planned_count)
                if node is not None else max(1, coherence.batch_size)
            ),
            sample_kwargs=sample_kwargs,
        )
    if schedule:
        # Configured overrides are validated even for modalities with no demand.
        for modality, node in generation_cfg.items():
            if modality not in schedule and node is not None and node.get("sample_kwargs"):
                merge_sample_kwargs(
                    decoder_sample_kwargs(decoder_kwargs, modality), node["sample_kwargs"]
                )
    return schedule, generation, coherence


def _gather_modality_output(
    out_local,
    *,
    rank: int,
    world_size: int,
    plan_count: int,
):
    """Return the planned prefix in rank-contiguous order on the main rank."""
    if world_size <= 1:
        return out_local[:plan_count]

    if isinstance(out_local, torch.Tensor):
        if rank == 0:
            gather_buffer = [torch.empty_like(out_local) for _ in range(world_size)]
            dist.gather(out_local, gather_buffer, dst=0)
            full = torch.cat(gather_buffer, dim=0)
            return full[:plan_count]
        else:
            dist.gather(out_local, dst=0)
            return None

    if isinstance(out_local, list):
        if rank == 0:
            buffer: list = [None for _ in range(world_size)]
            dist.gather_object(out_local, buffer, dst=0)
            merged: list = []
            for shard in buffer:
                if shard is None:
                    continue
                merged.extend(shard)
            return merged[:plan_count]
        else:
            dist.gather_object(out_local, None, dst=0)
            return None

    raise TypeError(
        f"_gather_modality_output: unsupported local output type {type(out_local).__name__}"
    )


def _co_generate_prior_outputs(
    model,
    schedule: dict[str, PriorDecode],
    *,
    device: torch.device,
    rank: int,
    world_size: int,
    eval_seed: int,
    prior_sample_kwargs=None,
):
    """Decode all planned modalities from one rank-sharded prior stream."""
    outputs: dict[str, object] = {}
    if not schedule:
        return outputs

    global_max = max(decode.planned_count for decode in schedule.values())
    if global_max <= 0:
        return outputs

    samples_per_rank = math.ceil(global_max / max(world_size, 1))
    if rank == 0:
        print(
            f"[Eval: PriorCoGen] plan = { {mod: decode.planned_count for mod, decode in schedule.items()} }; total z = {global_max}, per-rank shard = {samples_per_rank}"
        )

    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        _seed_eval_rng(eval_seed, rank, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=(device.type == "cuda"),
        ):
            z_full = model.sample_prior(
                samples_per_rank,
                device,
                sample_kwargs=prior_sample_kwargs,
            )

        # Decode every modality from the full local shard. Rank-contiguous
        # gathering then preserves cross-modal z alignment after prefix trims.
        for mod, decode in schedule.items():
            local_out = collect_decoder_outputs(iter_prior_decodes(
                model, mod, z_full, decode.batch_size,
                sample_kwargs=decode.sample_kwargs,
            ), empty_device=z_full.device)
            gathered = _gather_modality_output(
                local_out,
                rank=rank,
                world_size=world_size,
                plan_count=decode.planned_count,
            )
            if rank == 0:
                outputs[mod] = gathered
            else:
                outputs[mod] = None
            del local_out

    return outputs


@torch.inference_mode()
def run_evaluation_suite(
    model,
    data_loader,
    cfg,
    log_writer,
    step,
    device,
    *,
    prior_sample_kwargs=None,
    decoder_sample_kwargs_by_modality=None,
):
    """Run configured evaluation without changing the caller's model mode."""
    with misc.evaluation_mode(model), misc.preserve_rng_state(device):
        eval_cfg = cfg.get("evaluators", None)
        if not eval_cfg:
            return None

        eval_seed = int(cfg.get("eval", {}).get("seed", 43))
        _seed_eval_rng(eval_seed, misc.get_rank(), device)
        prior_sample_kwargs, decoder_sample_kwargs_by_modality = resolve_sample_kwargs(
            cfg,
            prior_overrides=prior_sample_kwargs,
            decoder_overrides=decoder_sample_kwargs_by_modality,
        )
        world_size = misc.get_world_size()
        rank = misc.get_rank()
        is_main = rank == 0
        paired_tasks, prefixes = prepare_paired_tasks(eval_cfg)
        information_jobs = [
            InformationJob(modality=modality, config=node, source=data_loader)
            for modality, node in (eval_cfg.get("information", {}) or {}).items()
            if node is not None
        ]
        if isinstance(data_loader, Iterator):
            for job in information_jobs:
                job.initialize(rank, world_size)
                prefix = RetainedPrefix(
                    modalities=(job.modality,),
                    num_samples=job.local_count,
                    complete=job.local_count <= 0,
                )
                job.source = prefix.batches
                if job.local_count > 0:
                    prefixes.append(prefix)
        _collect_eval_loader_batches(data_loader, prefixes)
        print(f"--- Running Evaluation Suite at Epoch {step} ---")

        prior_schedule, generation_tasks, coherence_task = _prepare_prior_decodes(
            eval_cfg, decoder_sample_kwargs_by_modality
        )
        if prior_schedule:
            prior_outputs = _co_generate_prior_outputs(
                model,
                prior_schedule,
                device=device,
                rank=rank,
                world_size=world_size,
                eval_seed=eval_seed,
                prior_sample_kwargs=prior_sample_kwargs,
            )
            if world_size > 1:
                dist.barrier()
        else:
            prior_outputs = {}

        context = EvaluationContext(
            model=model,
            enabled_phases=frozenset(eval_cfg),
            eval_datasets_cfg=cfg.get("eval_datasets", {}) or {},
            log_writer=log_writer,
            step=step,
            device=device,
            eval_seed=eval_seed,
            rank=rank,
            world_size=world_size,
            is_main=is_main,
            generation_tasks=generation_tasks,
            coherence_task=coherence_task,
            paired_tasks=paired_tasks,
            information_jobs=information_jobs,
            prior_outputs=prior_outputs,
            decoder_sample_kwargs=decoder_sample_kwargs_by_modality,
            feature_cache=EvaluationFeatureCache(),
            batch_size=resolve_eval_batch_size(
                cfg.get("train", {}) or {}, cfg.get("eval", {}) or {}, world_size
            ),
        )
        results = run_evaluation_phases(context)
        if world_size > 1:
            dist.barrier()
        return results
