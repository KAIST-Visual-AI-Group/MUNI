import math

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
    _generation_batch_size,
    _normalize_modalities_key,
    _required_generation_samples,
    _seed_eval_rng,
    run_evaluation_phases,
)
import muni.util.misc as misc
from muni.util.misc import print_on_main as print
from muni.evaluation.data import EvaluationFeatureCache, iter_eval_batch_matches


def _collect_eval_batch_requests(eval_cfg) -> list[tuple[str, ...]]:
    """Collect modality sets needed by reconstruction and translation."""
    requests: set[tuple[str, ...]] = set()

    for mod, mod_cfg in eval_cfg.get("reconstruction", {}).items():
        if mod_cfg is None:
            continue
        requests.add(_normalize_modalities_key([mod]))

    for cfg_node in eval_cfg.get("translation", []):
        srcs = list(cfg_node["srcs"]) if "srcs" in cfg_node else [cfg_node["src"]]
        requests.add(_normalize_modalities_key(srcs + [cfg_node["tgt"]]))

    return sorted(requests)


def _collect_prior_sample_plan(eval_cfg) -> dict[str, int]:
    """Return the maximum prior sample count needed per modality."""
    plan: dict[str, int] = {}

    gen_cfg = eval_cfg.get("generation", {}) or {}
    for mod, mod_cfg in gen_cfg.items():
        if mod_cfg is None:
            continue
        need = _required_generation_samples(mod_cfg)
        if need > 0:
            plan[mod] = max(plan.get(mod, 0), need)

    coh_cfg = eval_cfg.get("coherence", {}) or {}
    if coh_cfg:
        coh_n = int(coh_cfg.get("num_samples", 1024))
        heads_cfg = coh_cfg.get("heads", {}) or {}
        head_mods = [h for h, v in heads_cfg.items() if v is not None]
        if head_mods and coh_n > 0:
            pivot_modality = str(coh_cfg["pivot_modality"])
            plan[pivot_modality] = max(plan.get(pivot_modality, 0), coh_n)
            for h in head_mods:
                plan[h] = max(plan.get(h, 0), coh_n)

    return plan


def _gather_modality_output(
    out_local,
    *,
    rank: int,
    world_size: int,
    plan_count: int,
):
    """Gather one decoded modality on rank 0 and trim it to ``plan_count``."""
    if world_size <= 1:
        return out_local

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
    plan: dict[str, int],
    *,
    device: torch.device,
    rank: int,
    world_size: int,
    eval_seed: int,
    batch_size: int,
    prior_sample_kwargs=None,
    decoder_sample_kwargs_by_modality=None,
):
    """Decode all planned modalities from one rank-sharded prior stream."""
    outputs: dict[str, object] = {}
    if not plan:
        return outputs

    global_max = max(plan.values()) if plan else 0
    if global_max <= 0:
        return outputs

    samples_per_rank = math.ceil(global_max / max(world_size, 1))
    if rank == 0:
        print(
            f"[Eval: PriorCoGen] plan = {plan}; total z = {global_max}, per-rank shard = {samples_per_rank}"
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
        for mod, plan_count in plan.items():
            local_out = collect_decoder_outputs(iter_prior_decodes(
                model, mod, z_full, batch_size,
                sample_kwargs=decoder_sample_kwargs(
                    decoder_sample_kwargs_by_modality, mod
                ),
            ), empty_device=z_full.device)
            gathered = _gather_modality_output(
                local_out,
                rank=rank,
                world_size=world_size,
                plan_count=plan_count,
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
        eval_batch_cache = dict(iter_eval_batch_matches(
            data_loader, device, _collect_eval_batch_requests(eval_cfg)
        ))
        print(f"--- Running Evaluation Suite at Epoch {step} ---")

        prior_plan = _collect_prior_sample_plan(eval_cfg)
        if prior_plan:
            batch_size = 128
            generation_cfg = eval_cfg.get("generation", {}) or {}
            for modality, planned_count in prior_plan.items():
                metric_cfgs = (generation_cfg.get(modality, {}) or {}).get(
                    "metrics", []
                ) or []
                batch_size = max(
                    batch_size, _generation_batch_size(metric_cfgs, planned_count)
                )
            coherence_cfg = eval_cfg.get("coherence", {}) or {}
            if coherence_cfg:
                batch_size = max(batch_size, int(coherence_cfg.get("batch_size", 128)))
            generation_overrides = dict(decoder_sample_kwargs_by_modality)
            for modality, modality_cfg in generation_cfg.items():
                if modality_cfg is not None and modality_cfg.get("sample_kwargs"):
                    generation_overrides[modality] = merge_sample_kwargs(
                        decoder_sample_kwargs(
                            decoder_sample_kwargs_by_modality,
                            modality,
                        ),
                        modality_cfg.get("sample_kwargs"),
                    )
            prior_outputs = _co_generate_prior_outputs(
                model,
                prior_plan,
                device=device,
                rank=rank,
                world_size=world_size,
                eval_seed=eval_seed,
                batch_size=batch_size,
                prior_sample_kwargs=prior_sample_kwargs,
                decoder_sample_kwargs_by_modality=generation_overrides,
            )
            if world_size > 1:
                dist.barrier()
        else:
            prior_outputs = {}

        context = EvaluationContext(
            model=model,
            data_loader=data_loader,
            eval_cfg=eval_cfg,
            eval_datasets_cfg=cfg.get("eval_datasets", {}) or {},
            log_writer=log_writer,
            step=step,
            device=device,
            eval_seed=eval_seed,
            rank=rank,
            world_size=world_size,
            is_main=is_main,
            eval_batch_cache=eval_batch_cache,
            prior_outputs=prior_outputs,
            decoder_sample_kwargs=decoder_sample_kwargs_by_modality,
            feature_cache=EvaluationFeatureCache(),
        )
        results = run_evaluation_phases(context)
        if world_size > 1:
            dist.barrier()
        return results
