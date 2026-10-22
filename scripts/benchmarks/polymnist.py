"""PolyMNIST coherence and one-/multi-label translation benchmark.

Coherence requires all generated images to match the jointly generated digit
and quadrant. Translation scores verifier predictions against the input label
or label pair over the config's full evaluation split.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from omegaconf import OmegaConf

from scripts.model_loading import load_benchmark_config, resolve_sampling_steps
from muni.data.sources.labels import ClassSubdataset
from scripts.benchmarks import runtime
from scripts.benchmarks.runtime import (
    build_benchmark_parser,
    ci_cell,
    markdown_table,
    metric_mean,
    resolve_device,
    run_cli,
    summarize_samples,
    write_json,
)
from muni.evaluation.checkpoints import build_muni_model_from_checkpoint
from muni.evaluation.metrics.utils import per_sample_class_correct
from muni.evaluation.sampling import (
    decoder_sample_kwargs, iter_conditional_batches, iter_prior_batches, resolve_sample_kwargs,
)
import muni.util.misc as misc

if TYPE_CHECKING:
    from verifiers.verify import FrozenVerifier


IMAGE_MODALITIES = ("m0", "m1", "m2")
LABEL_MODALITIES = ("digit", "quadrant")


def _stable_route_seed(base: int, *parts: str) -> int:
    acc = int(base)
    for part in parts:
        for ch in str(part):
            acc = (acc * 131 + ord(ch)) % 2_147_483_647
    return acc


def _make_progress_logger(label: str, total: int):
    step = max(1, math.ceil(max(total, 1) / 5))
    next_mark = 0

    def log(done: int, *, force: bool = False) -> None:
        nonlocal next_mark
        if total <= 0:
            return
        if force or done >= total or done >= next_mark:
            pct = 100.0 * float(done) / float(total)
            print(f"[polybench] {label}: {done}/{total} ({pct:.1f}%)", flush=True)
            while next_mark <= done:
                next_mark += step

    return log


def _cat_or_empty(parts: list[torch.Tensor]) -> torch.Tensor:
    if not parts:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(parts, dim=0)


def _key_sort_value(key: str) -> tuple[int, str]:
    try:
        return int(key), key
    except ValueError:
        return 0, key


def _filter_stem_range(keys: set[str], stem_range) -> list[str]:
    if stem_range is None:
        return sorted(keys, key=_key_sort_value)
    lo, hi = int(stem_range[0]), int(stem_range[1])
    out: list[str] = []
    for key in keys:
        try:
            stem = int(key)
        except ValueError:
            continue
        if lo <= stem < hi:
            out.append(key)
    return sorted(out, key=_key_sort_value)


def load_eval_labels(
    cfg,
    *,
    max_samples: int | None = None,
) -> dict[str, torch.Tensor]:
    """Load only digit/quadrant labels from the config's full eval split.

    The PolyMNIST benchmark does not need real RGB inputs; reading labels
    directly avoids eager image caching while still respecting the config's
    ``dataset_eval.stem_range`` and label files.
    """
    dataset_cfg = cfg.get("dataset_eval", None) or cfg.dataset
    if "pair_groups" in dataset_cfg:
        raise ValueError(
            "PolyMNIST benchmark expects a single dataset_eval.subdatasets block; pair_groups are not supported here."
        )
    subdatasets_cfg = dataset_cfg.subdatasets

    label_sets: dict[str, ClassSubdataset] = {}
    for name in LABEL_MODALITIES:
        sd_cfg = subdatasets_cfg[name]
        if sd_cfg.type != "class":
            raise TypeError(
                f"Expected {name} subdataset type 'class', got {sd_cfg.type!r}."
            )
        params = OmegaConf.to_container(sd_cfg.get("params", {}) or {}, resolve=True)
        label_sets[name] = ClassSubdataset(
            **{str(key): value for key, value in params.items()}
        )

    common = set(label_sets["digit"].keys()) & set(label_sets["quadrant"].keys())
    keys = _filter_stem_range(common, dataset_cfg.get("stem_range", None))
    if max_samples is not None:
        keys = keys[: int(max_samples)]

    labels = {
        name: torch.stack([label_sets[name].get(key) for key in keys]).long()
        for name in LABEL_MODALITIES
    }
    return labels


@torch.no_grad()
def measure_unconditional_coherence(
    model,
    verifier: FrozenVerifier,
    *,
    n_samples: int,
    batch_size: int,
    device: torch.device,
    seed: int,
    prior_sample_kwargs=None,
    decoder_options=None,
) -> dict[str, Any]:
    misc.seed_everything(seed)
    strict_parts: list[torch.Tensor] = []
    digit_all_parts: list[torch.Tensor] = []
    quadrant_all_parts: list[torch.Tensor] = []
    per_image_digit: dict[str, list[torch.Tensor]] = {m: [] for m in IMAGE_MODALITIES}
    per_image_quadrant: dict[str, list[torch.Tensor]] = {
        m: [] for m in IMAGE_MODALITIES
    }

    done = 0
    progress = _make_progress_logger("unconditional", n_samples)
    progress(0, force=True)
    for outputs in iter_prior_batches(
        model, ("digit", "quadrant", *IMAGE_MODALITIES),
        n_samples, batch_size, device,
        prior_sample_kwargs=prior_sample_kwargs,
        decoder_options=decoder_options,
    ):
        gen_digit = outputs["digit"].flatten().long()
        gen_quadrant = outputs["quadrant"].flatten().long()
        bsz = int(gen_digit.shape[0])

        digit_all = torch.ones(bsz, device=device, dtype=torch.bool)
        quadrant_all = torch.ones(bsz, device=device, dtype=torch.bool)
        for mod in IMAGE_MODALITIES:
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                img = outputs[mod]
                pred = verifier.predict_classes(img)
            digit_scores = per_sample_class_correct(
                pred["digit"].flatten().long(), gen_digit.to(pred["digit"].device)
            )
            quadrant_scores = per_sample_class_correct(
                pred["quadrant"].flatten().long(),
                gen_quadrant.to(pred["quadrant"].device),
            )
            digit_ok = digit_scores.bool().to(device)
            quadrant_ok = quadrant_scores.bool().to(device)
            digit_all &= digit_ok.to(device)
            quadrant_all &= quadrant_ok.to(device)
            per_image_digit[mod].append(digit_scores)
            per_image_quadrant[mod].append(quadrant_scores)

        strict = digit_all & quadrant_all
        strict_parts.append(strict.float().cpu())
        digit_all_parts.append(digit_all.float().cpu())
        quadrant_all_parts.append(quadrant_all.float().cpu())
        done += bsz
        progress(done)

    out: dict[str, Any] = {
        "strict_all_labels_all_images": summarize_samples(_cat_or_empty(strict_parts)),
        "digit_all_images": summarize_samples(_cat_or_empty(digit_all_parts)),
        "quadrant_all_images": summarize_samples(_cat_or_empty(quadrant_all_parts)),
        "per_image": {},
    }
    for mod in IMAGE_MODALITIES:
        out["per_image"][mod] = {
            "digit": summarize_samples(_cat_or_empty(per_image_digit[mod])),
            "quadrant": summarize_samples(_cat_or_empty(per_image_quadrant[mod])),
        }
    return out


@torch.no_grad()
def _run_label_route(
    model,
    verifier: FrozenVerifier,
    eval_labels: dict[str, torch.Tensor],
    *,
    src_labels: Sequence[str],
    tgt_image: str,
    batch_size: int,
    device: torch.device,
    seed: int,
    deterministic_encode: bool,
    decoder_options=None,
) -> dict[str, torch.Tensor]:
    n = int(eval_labels[src_labels[0]].shape[0])
    parts: dict[str, list[torch.Tensor]] = {name: [] for name in src_labels}
    strict_parts: list[torch.Tensor] = []
    misc.seed_everything(seed)
    route = f"{'+'.join(src_labels)}->{tgt_image}"
    progress = _make_progress_logger(route, n)
    progress(0, force=True)
    for sl, img in iter_conditional_batches(
        model, eval_labels, list(src_labels), tgt_image, n, batch_size, device,
        mode="direct" if len(src_labels) == 1 else "aggregate",
        sample_kwargs=decoder_sample_kwargs(decoder_options, tgt_image),
        deterministic_encode=deterministic_encode,
    ):
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            if len(src_labels) == 1:
                pred = verifier.predict_classes(img, heads=(src_labels[0],))
            else:
                pred = verifier.predict_classes(img)
        targets = {
            name: eval_labels[name][sl].flatten().long().to(pred[name].device)
            for name in src_labels
        }
        scores = {
            name: per_sample_class_correct(pred[name], targets[name])
            for name in src_labels
        }
        for name in src_labels:
            parts[name].append(scores[name])
        if len(src_labels) > 1:
            strict_parts.append(
                (scores[src_labels[0]].bool() & scores[src_labels[1]].bool()).float()
            )
        progress(sl.stop)
    values = {name: _cat_or_empty(parts[name]) for name in src_labels}
    if len(src_labels) > 1:
        values["strict_both_labels"] = _cat_or_empty(strict_parts)
    return values


@torch.no_grad()
def measure_one_to_one_alignment(
    model,
    verifier: FrozenVerifier,
    eval_labels: dict[str, torch.Tensor],
    *,
    batch_size: int,
    device: torch.device,
    seed: int,
    deterministic_encode: bool,
    decoder_options=None,
) -> dict[str, Any]:
    routes: dict[str, Any] = {}
    digit_parts: list[torch.Tensor] = []
    quadrant_parts: list[torch.Tensor] = []

    for label in LABEL_MODALITIES:
        for image_mod in IMAGE_MODALITIES:
            key = f"{label}_to_{image_mod}"
            result = _run_label_route(
                model,
                verifier,
                eval_labels,
                src_labels=(label,),
                tgt_image=image_mod,
                batch_size=batch_size,
                device=device,
                seed=_stable_route_seed(seed, label, image_mod),
                deterministic_encode=deterministic_encode,
                decoder_options=decoder_options,
            )
            values = result[label]
            routes[key] = {"summary": summarize_samples(values)}
            if label == "digit":
                digit_parts.append(values)
            else:
                quadrant_parts.append(values)

    all_parts = digit_parts + quadrant_parts
    return {
        "routes": routes,
        "digit_avg": summarize_samples(_cat_or_empty(digit_parts)),
        "quadrant_avg": summarize_samples(_cat_or_empty(quadrant_parts)),
        "avg": summarize_samples(_cat_or_empty(all_parts)),
    }


@torch.no_grad()
def measure_multi_to_one_alignment(
    model,
    verifier: FrozenVerifier,
    eval_labels: dict[str, torch.Tensor],
    *,
    batch_size: int,
    device: torch.device,
    seed: int,
    deterministic_encode: bool,
    decoder_options=None,
) -> dict[str, Any]:
    routes: dict[str, Any] = {}
    pooled_parts: list[torch.Tensor] = []
    for image_mod in IMAGE_MODALITIES:
        result = _run_label_route(
            model,
            verifier,
            eval_labels,
            src_labels=LABEL_MODALITIES,
            tgt_image=image_mod,
            batch_size=batch_size,
            device=device,
            seed=_stable_route_seed(seed, "digit+quadrant", image_mod),
            deterministic_encode=deterministic_encode,
            decoder_options=decoder_options,
        )
        values = result["strict_both_labels"]
        routes[image_mod] = {
            name: summarize_samples(result[name])
            for name in ("strict_both_labels", *LABEL_MODALITIES)
        }
        pooled_parts.append(values)
    strict = {m: routes[m]["strict_both_labels"] for m in IMAGE_MODALITIES}
    strict["avg"] = summarize_samples(_cat_or_empty(pooled_parts))
    return {"strict_accuracy": strict, "routes": routes}


def run_benchmark(
    *,
    method: str,
    config_path: str,
    checkpoint_path: str,
    output_dir: Path,
    verifier_ckpt: str | None,
    batch_size: int,
    gen_batch_size: int,
    prior_steps: int | None,
    decoder_steps: int | None,
    decoder_cfg: float | None,
    no_compile: bool,
    deterministic_encode: bool,
    max_samples: int | None,
    seed: int,
) -> dict[str, Any]:
    from verifiers.verify import FrozenVerifier

    device = resolve_device()
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    cfg = load_benchmark_config(
        config_path, checkpoint_path, no_compile=bool(no_compile)
    )
    if verifier_ckpt is None:
        verifier_ckpt = cfg.paths.verifier_ckpt
    misc.seed_everything(int(cfg.train.seed))
    model = build_muni_model_from_checkpoint(cfg, checkpoint_path, device)
    prior_overrides = {"sampler_steps": int(prior_steps)} if prior_steps is not None else {}
    decoder_overrides = {}
    if decoder_steps is not None:
        decoder_overrides["sampler_steps"] = int(decoder_steps)
    if decoder_cfg is not None:
        decoder_overrides["cfg_scale"] = float(decoder_cfg)
    prior_sample_kwargs, decoder_options = resolve_sample_kwargs(
        cfg,
        prior_overrides=prior_overrides,
        decoder_overrides={"default": decoder_overrides},
    )
    resolved_prior_steps, resolved_decoder_steps = resolve_sampling_steps(
        model, prior_sample_kwargs, decoder_options,
    )

    eval_labels = load_eval_labels(cfg, max_samples=max_samples)
    n_eval = int(eval_labels["digit"].shape[0])
    verifier = FrozenVerifier.from_checkpoint(verifier_ckpt, device=device)

    print(
        f"[polybench] method={method} n_eval={n_eval} "
        f"prior integration steps={resolved_prior_steps} decoder integration steps={resolved_decoder_steps} "
        f"decoder_cfg={decoder_cfg if decoder_cfg is not None else 'config'}",
        flush=True,
    )

    t0 = time.time()
    coherence = measure_unconditional_coherence(
        model,
        verifier,
        n_samples=n_eval,
        batch_size=gen_batch_size,
        device=device,
        seed=seed,
        prior_sample_kwargs=prior_sample_kwargs,
        decoder_options=decoder_options,
    )
    t1 = time.time()
    one_to_one = measure_one_to_one_alignment(
        model,
        verifier,
        eval_labels,
        batch_size=batch_size,
        device=device,
        seed=seed + 10_000,
        deterministic_encode=deterministic_encode,
        decoder_options=decoder_options,
    )
    t2 = time.time()
    multi_to_one = measure_multi_to_one_alignment(
        model,
        verifier,
        eval_labels,
        batch_size=batch_size,
        device=device,
        seed=seed + 20_000,
        deterministic_encode=deterministic_encode,
        decoder_options=decoder_options,
    )
    t3 = time.time()

    result = {
        "method": method,
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "verifier_ckpt": str(verifier_ckpt),
        "n_eval": n_eval,
        "prior_steps": resolved_prior_steps,
        "decoder_steps": resolved_decoder_steps,
        "decoder_cfg": decoder_cfg,
        "deterministic_encode": bool(deterministic_encode),
        "unconditional_coherence": coherence,
        "one_to_one_alignment": one_to_one,
        "multi_to_one_alignment": multi_to_one,
        "timing_sec": {
            "unconditional_coherence": round(t1 - t0, 2),
            "one_to_one_alignment": round(t2 - t1, 2),
            "multi_to_one_alignment": round(t3 - t2, 2),
            "total": round(t3 - t0, 2),
        },
    }
    out_path = output_dir / f"{method}.json"
    write_json(out_path, result)
    write_summary(output_dir)
    return result


def _fmt(entry: Any) -> str:
    value = metric_mean(entry)
    if value is None:
        return "—"
    if isinstance(entry, dict) and "n" in entry:
        n = int(entry.get("n", 0))
        std = float(entry.get("std", float("nan")))
        if n > 0 and not math.isnan(std):
            return ci_cell(value, std, n)
    return f"{value:.4f}"


def write_summary(output_dir: Path) -> None:
    runtime.write_summary(output_dir, _render_markdown)


def _render_markdown(reports: list[tuple[str, dict[str, Any]]]) -> str:
    rows = []
    for name, payload in reports:
        if not all(key in payload for key in ("unconditional_coherence", "prior_steps", "decoder_steps")):
            raise ValueError(f"Not a PolyMNIST report: {name}")
        rows.append((payload.get("method", name), payload))
    lines = [
        "# PolyMNIST-Quadrant-Labels Benchmark",
        "",
        f"_Last update: {time.strftime('%Y-%m-%d %H:%M:%S')}_",
        "",
        "Cells are `accuracy ± 95% CI half-width`; higher is better. "
        "The default run uses the full `dataset_eval` split from each config.",
        "",
        "## Sampling Budgets",
        "",
    ]
    lines += markdown_table(
        ["method", "Prior integration steps", "Image decoder integration steps"],
        [[method, str(payload.get("prior_steps", "—")),
          str(payload.get("decoder_steps", "—"))] for method, payload in rows],
        ["---", "---:", "---"],
    )
    lines += [
        "## Unconditional Coherence",
        "",
    ]
    lines += markdown_table(
        ["method", "n", "strict all labels/images ↑", "digit all images ↑", "quadrant all images ↑"],
        [[method, str(payload.get("n_eval", "—")),
          *(_fmt(payload["unconditional_coherence"].get(key)) for key in (
              "strict_all_labels_all_images", "digit_all_images", "quadrant_all_images"))]
         for method, payload in rows],
        ["---", "---:", "---:", "---:", "---:"],
    )
    lines += ["", "## One->One Alignment", ""]
    lines += markdown_table(
        ["method", "digit->image avg ↑", "quadrant->image avg ↑", "all single-label avg ↑"],
        [[method, *(_fmt(payload.get("one_to_one_alignment", {}).get(key))
                    for key in ("digit_avg", "quadrant_avg", "avg"))]
         for method, payload in rows],
        ["---", "---:", "---:", "---:"],
    )
    lines += ["", "## Multi->One Alignment", ""]
    lines += markdown_table(
        ["method", "{digit,quadrant}->m0 ↑", "->m1 ↑", "->m2 ↑", "avg ↑"],
        [[method, *(_fmt(payload.get("multi_to_one_alignment", {}).get("strict_accuracy", {}).get(key))
                    for key in ("m0", "m1", "m2", "avg"))]
         for method, payload in rows],
        ["---", "---:", "---:", "---:", "---:"],
    )
    return "\n".join(lines) + "\n"


def build_parser(*, prog: str | None = None) -> argparse.ArgumentParser:
    parser = build_benchmark_parser(
        __doc__ or "PolyMNIST benchmark",
        prog=prog,
    )
    parser.add_argument(
        "--method", required=True, help="Method tag for JSON/markdown rows."
    )
    parser.add_argument("--config", required=True, help="Training config YAML.")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path.")
    parser.add_argument(
        "--output-dir",
        default="output/neurips2026_poly_mnist/benchmark",
        help="Directory for <method>.json and results.md.",
    )
    parser.add_argument(
        "--verifier-ckpt",
        default=None,
        help="Override paths.verifier_ckpt.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Translation batch size."
    )
    parser.add_argument(
        "--gen-batch-size", type=int, default=128, help="Prior-generation batch size."
    )
    parser.add_argument(
        "--prior-steps", type=int, default=None, help="Override prior integration steps (sampler_steps)."
    )
    parser.add_argument(
        "--decoder-steps",
        type=int,
        default=None,
        help="Override image decoder integration steps (sampler_steps).",
    )
    parser.add_argument(
        "--decoder-cfg",
        type=float,
        default=None,
        help="Override image decoder sampler_cfg_scale.",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile at model build.",
    )
    parser.add_argument(
        "--deterministic-encode",
        action="store_true",
        help="Use posterior means for conditional translation.",
    )
    parser.add_argument(
        "--num-samples",
        "--max-samples",
        dest="max_samples",
        type=int,
        default=None,
        help="Debug cap only. Default evaluates the full config evaluation split.",
    )
    parser.add_argument("--seed", type=int, default=43)
    return parser


def run(args: argparse.Namespace) -> int:
    """Execute the PolyMNIST benchmark for already-parsed arguments."""
    result = run_benchmark(
        method=args.method,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=Path(args.output_dir),
        verifier_ckpt=args.verifier_ckpt,
        batch_size=args.batch_size,
        gen_batch_size=args.gen_batch_size,
        prior_steps=args.prior_steps,
        decoder_steps=args.decoder_steps,
        decoder_cfg=args.decoder_cfg,
        no_compile=args.no_compile,
        deterministic_encode=args.deterministic_encode,
        max_samples=args.max_samples,
        seed=args.seed,
    )

    print(
        json.dumps(
            {
                "method": result["method"],
                "n_eval": result["n_eval"],
                "coherence": result["unconditional_coherence"][
                    "strict_all_labels_all_images"
                ],
                "one_to_one_avg": result["one_to_one_alignment"]["avg"],
                "multi_to_one": result["multi_to_one_alignment"]["strict_accuracy"],
                "timing_sec": result["timing_sec"],
            },
            indent=2,
        )
    )
    print(f"[polybench] wrote {Path(args.output_dir) / (args.method + '.json')}")
    print(f"[polybench] summary {Path(args.output_dir) / 'results.md'}")
    return 0


def main(argv: Sequence[str] | None = None, *, prog: str | None = None) -> int:
    return run_cli(build_parser, run, argv, prog=prog)
