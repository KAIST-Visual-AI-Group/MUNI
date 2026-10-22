"""Paper benchmark for Gaussian-mixture fidelity and coherence."""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Sequence
from pathlib import Path

import torch
from omegaconf import OmegaConf

from muni.config_utils import load_config
from muni.data import build_dataset, build_eval_loader
from scripts.benchmarks import runtime
from scripts.benchmarks.runtime import (
    build_benchmark_parser,
    resolve_device,
    run_cli,
    markdown_table,
    write_json,
)
from muni.evaluation.checkpoints import build_muni_model_from_checkpoint
from scripts.model_loading import resolve_sampling_steps
from muni.evaluation.sampling import (
    decoder_sample_kwargs, iter_prior_batches, resolve_sample_kwargs,
)
from muni.evaluation.metrics.gmm import (
    AlignmentMSE,
    PairwiseSharedDimsMSE,
    SlicedWassersteinMetric,
)
from muni.util.misc import extract_dict_data, move_to_device, seed_everything


# Method loading


def load_method(
    method_dir: Path,
    num_samples: int,
    device: torch.device,
    checkpoint_name: str = "checkpoint-last.pth",
    config_path: Path | None = None,
    checkpoint_path: Path | None = None,
):
    """Build the model and load its EMA checkpoint."""
    resolved_config = config_path or (method_dir / "config.yaml")
    resolved_checkpoint = checkpoint_path or (method_dir / checkpoint_name)
    cfg = load_config(str(resolved_config))
    # Expand the eval split.  ``toy.n_eval`` is the canonical knob, but a
    # number of downstream interpolations (``dataset_eval.stem_range``,
    # per-subdataset ``params.n_samples``) do not re-resolve after an
    # update, so we also write concrete values directly.
    OmegaConf.update(cfg, "toy.n_eval", int(num_samples))
    n_train = int(cfg.toy.get("n_train", 1_000_000))
    total = n_train + int(num_samples)
    if cfg.get("dataset_eval", None) is not None:
        OmegaConf.update(cfg, "dataset_eval.stem_range", [n_train, total])
        # Each subdataset's procedural pool size must cover [0, total).
        eval_subs = cfg.dataset_eval.get("subdatasets", {})
        for sub_key in list(eval_subs):
            OmegaConf.update(
                cfg,
                f"dataset_eval.subdatasets.{sub_key}.params.n_samples",
                int(total),
            )

    model = build_muni_model_from_checkpoint(
        cfg,
        str(resolved_checkpoint),
        device,
    )
    return cfg, model


# Measurements


@torch.no_grad()
def measure_unconditional(
    model,
    modalities: list[str],
    num_samples: int,
    batch_size: int,
    marginal_dir: Path,
    venn_metric_params: dict,
    device: torch.device,
    prior_sample_kwargs=None,
    decoder_options=None,
):
    """Measure marginal SWD and shared-dimension coherence."""
    swd = {
        m: SlicedWassersteinMetric(
            reference_file=str(marginal_dir / f"marginal_{m}.npz"),
            num_projections=128,
            num_samples=num_samples,
            batch_size=batch_size,
            seed=ord(m),
        )
        for m in modalities
    }
    coherence_metric = PairwiseSharedDimsMSE(**venn_metric_params)

    for gens in iter_prior_batches(
        model, modalities, num_samples, batch_size, device,
        prior_sample_kwargs=prior_sample_kwargs,
        decoder_options=decoder_options,
        autocast=False,
    ):
        for m, out in gens.items():
            swd[m].update(out)
        coherence_metric.update(sources=gens)

    fidelity = {m: float(swd[m].compute()["SWD"]) for m in modalities}
    fidelity["avg"] = sum(fidelity[m] for m in modalities) / len(modalities)

    coherence = {
        name.removeprefix("PairMSE/"): value
        for name, value in coherence_metric.compute().items()
    }

    return {"fidelity": fidelity, "coherence": coherence}


def _make_eval_loader(cfg, batch_size: int):
    dataset_cfg = cfg.get("dataset_eval", None) or cfg.dataset
    return build_eval_loader(
        build_dataset(dataset_cfg, flip_prob=0.0),
        batch_size=int(batch_size),
        num_workers=int(cfg.train.get("num_workers", 1)),
        pin_memory=bool(cfg.train.get("pin_mem", True)),
        num_replicas=1,
        rank=0,
    )


@torch.no_grad()
def _iter_eval_batches(
    model,
    cfg,
    num_samples: int,
    batch_size: int,
    device: torch.device,
):
    """Yield ``data`` dicts (modality → CUDA tensor) up to ``num_samples`` rows."""
    eval_loader = _make_eval_loader(cfg, batch_size)
    n_done = 0
    for batch in eval_loader:
        if n_done >= num_samples:
            return
        data = extract_dict_data(batch)
        data = {
            k: move_to_device(v, device, non_blocking=True) for k, v in data.items()
        }
        first = next(iter(data.values()))
        B = int(first.shape[0])
        if n_done + B > num_samples:
            B = num_samples - n_done
            data = {k: v[:B] for k, v in data.items()}
        yield data
        n_done += B


@torch.no_grad()
def measure_conditional(
    model,
    cfg,
    modalities: list[str],
    num_samples: int,
    batch_size: int,
    venn_metric_params: dict,
    device: torch.device,
    decoder_options=None,
    *,
    mode: str,
):
    """Measure ordered routes with independent source-dimension populations."""
    if mode == "direct":
        pairs = [(s, t) for s in modalities for t in modalities if s != t]
        metrics = {
            pair: AlignmentMSE(shared_dims_only=True, **venn_metric_params)
            for pair in pairs
        }
        directions = [(tgt, [src]) for src, tgt in pairs]
    elif mode == "aggregate":
        directions = []
        metrics = {}
        for tgt in modalities:
            srcs = [m for m in modalities if m != tgt]
            directions.append((tgt, srcs))
            metrics.update(
                {
                    (src, tgt): AlignmentMSE(shared_dims_only=True, **venn_metric_params)
                    for src in srcs
                }
            )
    else:
        raise ValueError(f"Unknown conditional measurement mode: {mode!r}")

    for data in _iter_eval_batches(model, cfg, num_samples, batch_size, device):
        for tgt, srcs in directions:
            pred = model.conditional_sample(
                data,
                srcs,
                tgt,
                mode=mode,
                decoder_sample_kwargs=decoder_sample_kwargs(decoder_options, tgt),
            ).detach()
            for src in srcs:
                metrics[(src, tgt)].update(
                    pred,
                    data[tgt],
                    src_modalities=[src],
                    tgt_modality=tgt,
                )

    if mode == "direct":
        alignment = {
            f"{src}_to_{tgt}": metrics[(src, tgt)].compute()["AlignMSE"]
            for src, tgt in pairs
        }
    else:
        alignment = {}
        for tgt, srcs in directions:
            per_src = [metrics[(src, tgt)].compute()["AlignMSE"] for src in srcs]
            per_src = [value for value in per_src if not math.isnan(value)]
            key = "".join(srcs) + "_to_" + tgt
            alignment[key] = float(sum(per_src) / len(per_src)) if per_src else float("nan")
    vals = [v for v in alignment.values() if not math.isnan(v)]
    alignment["avg"] = float(sum(vals) / len(vals)) if vals else float("nan")
    return {"alignment": alignment}


# Per-method evaluation


def benchmark_method(
    method_dir: Path,
    *,
    num_samples: int,
    batch_size: int,
    prior_steps: int | None,
    decoder_steps: int | None,
    checkpoint_name: str = "checkpoint-last.pth",
    config_path: Path | None = None,
    checkpoint_path: Path | None = None,
    device: torch.device,
):
    cfg, model = load_method(
        method_dir,
        num_samples,
        device,
        checkpoint_name=checkpoint_name,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
    )
    seed = int(cfg.eval.seed)
    seed_everything(seed)
    modalities = list(cfg.model.routing.modalities)
    n_modalities = int(cfg.toy.n_modalities)
    region_dim = int(cfg.toy.region_dim)
    region_dim_by_size = (
        OmegaConf.to_container(cfg.toy.get("region_dim_by_size", {})) or {}
    )
    marginal_dir = Path(str(cfg.toy.marginal_dir)).resolve()
    prior_sample_kwargs, decoder_options = resolve_sample_kwargs(
        cfg,
        prior_overrides={"sampler_steps": prior_steps} if prior_steps is not None else {},
        decoder_overrides={"default": {"sampler_steps": decoder_steps}}
        if decoder_steps is not None else {},
    )
    resolved_prior_steps, resolved_decoder_steps = resolve_sampling_steps(
        model, prior_sample_kwargs, decoder_options,
    )

    venn_metric_params = {
        "n_modalities": n_modalities,
        "region_dim": region_dim,
        "region_dim_by_size": region_dim_by_size,
        "modality_names": modalities,
    }

    t0 = time.time()
    uncond = measure_unconditional(
        model,
        modalities,
        num_samples,
        batch_size,
        marginal_dir,
        venn_metric_params,
        device,
        prior_sample_kwargs,
        decoder_options,
    )
    t1 = time.time()
    o2o = measure_conditional(
        model,
        cfg,
        modalities,
        num_samples,
        batch_size,
        venn_metric_params,
        device,
        decoder_options,
        mode="direct",
    )
    t2 = time.time()
    m2o = measure_conditional(
        model,
        cfg,
        modalities,
        num_samples,
        batch_size,
        venn_metric_params,
        device,
        decoder_options,
        mode="aggregate",
    )
    t3 = time.time()

    return {
        "config": {
            "num_samples": num_samples,
            "batch_size": batch_size,
            "prior_steps": resolved_prior_steps,
            "decoder_steps": resolved_decoder_steps,
            "seed": seed,
            "ground_truth": {
                "modalities": modalities,
                "n_modalities": n_modalities,
                "region_dim": region_dim,
                "region_dim_by_size": region_dim_by_size,
                "K": int(cfg.toy.K),
                "mixture_seed": int(cfg.toy.mixture_seed),
                "observation_noise_std": float(cfg.toy.get("observation_noise_std", 0.0)),
                "observation_noise_seed": int(cfg.toy.get("observation_noise_seed", 314159)),
                "evaluation_dataset": OmegaConf.to_container(cfg.get("dataset_eval") or cfg.dataset, resolve=True),
                "marginal_dir": str(marginal_dir),
            },
            "sampling": {
                "prior_kwargs": prior_sample_kwargs,
                "decoder_kwargs": decoder_options,
                "prior": OmegaConf.to_container(cfg.model.routing.prior, resolve=True),
                "decoders": OmegaConf.to_container(cfg.model.routing.decoders, resolve=True),
            },
        },
        "unconditional_generation": uncond,
        "one_to_one_translation": o2o,
        "many_to_one_translation": m2o,
        "timing_sec": {
            "unconditional": round(t1 - t0, 2),
            "one_to_one": round(t2 - t1, 2),
            "many_to_one": round(t3 - t2, 2),
        },
    }


# Summary


def write_summary(output_dir: Path):
    runtime.write_summary(output_dir, _render_markdown)


def _render_markdown(rows: list[tuple[str, dict]]) -> str:
    for name, result in rows:
        if not all(key in result for key in (
            "config", "unconditional_generation", "one_to_one_translation",
            "many_to_one_translation",
        )) or not all(key in result.get("config", {}) for key in ("prior_steps", "decoder_steps")):
            raise ValueError(f"Not a GMM report: {name}")
    protocol_fields = ("num_samples", "batch_size", "prior_steps", "decoder_steps", "seed", "ground_truth")
    reference = rows[0][1].get("config", {})
    for name, result in rows[1:]:
        config = result.get("config", {})
        mismatched = [field for field in protocol_fields if config.get(field) != reference.get(field)]
        if mismatched:
            raise ValueError(f"Cannot compare mixed GMM protocols: {name} differs in {mismatched}; use separate output directories")

    lines: list[str] = []
    cfg0 = rows[0][1].get("config", {})
    lines.append(
        f"# Gaussian-mixture benchmark (N={cfg0.get('num_samples')}, "
        f"prior integration steps={cfg0.get('prior_steps')}, "
        f"decoder integration steps={cfg0.get('decoder_steps')})"
    )
    lines.append("")

    # Discover all metric keys from first row
    sample = rows[0][1]
    fid_keys = list(sample["unconditional_generation"]["fidelity"])
    coh_keys = list(sample["unconditional_generation"]["coherence"])
    o2o_keys = list(sample["one_to_one_translation"]["alignment"])
    m2o_keys = list(sample["many_to_one_translation"]["alignment"])

    def section(title, blocks, keys):
        matrix = []
        for k in keys:
            cells = [k]
            vals = [block.get(k) for block in blocks]
            valid = [
                v
                for v in vals
                if v is not None and not (isinstance(v, float) and math.isnan(v))
            ]
            best = min(valid) if valid else None
            for v in vals:
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    cells.append("—")
                else:
                    mark = "**" if best is not None and abs(v - best) < 1e-9 else ""
                    cells.append(f"{mark}{v:.4f}{mark}")
            matrix.append(cells)
        lines.extend([f"## {title}", *markdown_table(
            ["Metric", *(name for name, _ in rows)], matrix, ["---"] * (len(rows) + 1),
        ), ""])

    section("Unconditional Generation — Fidelity (SWD ↓)",
            [r["unconditional_generation"]["fidelity"] for _, r in rows], fid_keys)
    section("Unconditional Generation — Pairwise Coherence (MSE ↓)",
            [r["unconditional_generation"]["coherence"] for _, r in rows], coh_keys)
    section("One→One Translation — AlignmentMSE (↓)",
            [r["one_to_one_translation"]["alignment"] for _, r in rows], o2o_keys)
    section("Many→One Translation — AlignmentMSE (↓)",
            [r["many_to_one_translation"]["alignment"] for _, r in rows], m2o_keys)

    return "\n".join(lines)


# CLI


def discover_methods(
    final_dir: Path, checkpoint_name: str = "checkpoint-last.pth"
) -> list[str]:
    out: list[str] = []
    for p in sorted(final_dir.iterdir()):
        if not p.is_dir():
            continue
        if (p / "config.yaml").exists() and (p / checkpoint_name).exists():
            out.append(p.name)
    return out


def build_parser(*, prog: str | None = None) -> argparse.ArgumentParser:
    parser = build_benchmark_parser(
        __doc__ or "Gaussian-mixture benchmark",
        prog=prog,
    )
    parser.add_argument(
        "--method", help="Single method tag; requires --config and --checkpoint."
    )
    parser.add_argument("--config", help="Single-method training config YAML.")
    parser.add_argument("--checkpoint", help="Single-method model checkpoint path.")
    parser.add_argument(
        "--final-dir",
        default="output/gmm",
        help="Parent directory holding <method>/checkpoint-last.pth.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Method subdirectory names (default: auto-detect).",
    )
    parser.add_argument(
        "--output-dir",
        default="output/gmm/benchmark",
        help="Where per-method JSONs and the Markdown summary land.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20000,
        help="Sample budget for every category.",
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--prior-steps", type=int, default=None, help="Override prior integration steps."
    )
    parser.add_argument(
        "--decoder-steps", type=int, default=None, help="Override integration steps per modality decoder."
    )
    parser.add_argument(
        "--checkpoint-name",
        default="checkpoint-last.pth",
        help="Filename of the checkpoint inside each method dir.",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    """Execute the GMM benchmark for already-parsed arguments."""
    for name in ("batch_size", "num_samples", "prior_steps", "decoder_steps"):
        if getattr(args, name) is not None and getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.methods is not None and any(value is not None for value in (args.method, args.config, args.checkpoint)):
        raise ValueError("--methods cannot be combined with single-method options")
    final_dir = Path(args.final_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    direct_values = {
        "--method": args.method,
        "--config": args.config,
        "--checkpoint": args.checkpoint,
    }
    if any(value is not None for value in direct_values.values()):
        missing = [flag for flag, value in direct_values.items() if value is None]
        if missing:
            raise ValueError(f"single-method mode requires {', '.join(missing)}")
        runs = [
            (
                args.method,
                Path(args.checkpoint).resolve().parent,
                Path(args.config).resolve(),
                Path(args.checkpoint).resolve(),
            )
        ]
    else:
        methods = args.methods or discover_methods(final_dir, args.checkpoint_name)
        if not methods:
            raise SystemExit(f"No methods found under {final_dir}.")
        runs = [(name, final_dir / name, None, None) for name in methods]
    device = resolve_device(require_cuda=True)

    print(f"Methods: {[name for name, *_ in runs]}")
    print(
        f"N={args.num_samples}, prior integration steps={args.prior_steps}, "
        f"decoder integration steps={args.decoder_steps}"
    )
    print()

    for name, method_dir, config_path, checkpoint_path in runs:
        resolved_checkpoint = checkpoint_path or (method_dir / args.checkpoint_name)
        print(f"  [{name}] benchmarking ({resolved_checkpoint.name})...", flush=True)
        result = benchmark_method(
            method_dir,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            prior_steps=args.prior_steps,
            decoder_steps=args.decoder_steps,
            checkpoint_name=args.checkpoint_name,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device=device,
        )
        write_json(output_dir / f"{name}.json", result)
        write_summary(output_dir)
        print(f"  [{name}] done. timings={result['timing_sec']}")

    print(f"\nSummary written to {output_dir / 'results.md'}")
    return 0


def main(argv: Sequence[str] | None = None, *, prog: str | None = None) -> int:
    return run_cli(build_parser, run, argv, prog=prog)
