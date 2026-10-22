"""Run the reportable image-text-audio fidelity, alignment, and coherence protocols."""

import argparse
import json
import math
import time
import re
import uuid
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from omegaconf import OmegaConf
from tqdm import tqdm

from muni.config_utils import load_config
import muni.util.misc as misc
from scripts.benchmarks.runtime import (
    build_benchmark_parser,
    resolve_device,
    run_cli,
)
from scripts.model_loading import (
    load_evaluation_model,
    resolve_cli_sample_kwargs,
    resolve_sampling_steps,
)
from muni.evaluation.sampling import decoder_sample_kwargs, iter_prior_batches
from muni.evaluation.metrics.common import ImageFID
from muni.evaluation.metrics.image_text_audio import (
    AudioFAD,
    AudioImageSimilarity,
    CIDEr,
    CLAPScore,
    CLIPScore,
    compute_alignment,
)
from muni.evaluation.data import EvaluationFeatureCache, load_captions_json
from muni.evaluation.image_text_audio.decoding import (
    SourceDecodeCache,
    decode_generated_output,
    decode_features,
    iter_decode_image_chunks,
    resolve_decoder_config,
    translate_features,
)
from muni.evaluation.image_text_audio.encoders import clip_encode_images


# Task Definitions

ONE_TO_ONE_TASKS = [
    # (display, src, tgt, dataset_key, fidelity_kind, alignment_kind)
    ("T->I", "text", "image", "text-image", "fid", "clip"),
    ("I->T", "image", "text", "text-image", "cider", "clip"),
    ("T->A", "text", "audio", "text-audio", "fad", "clap"),
    ("A->T", "audio", "text", "text-audio", "cider", "clap"),
    ("I->A", "image", "audio", "image-audio", "fad", "ais"),
    ("A->I", "audio", "image", "image-audio", "fid", "ais"),
]

MANY_TO_ONE_TASKS = [
    # (display, src_mods, tgt, dataset_key, [(metric, label, src_for_metric)])
    (
        "(I+A)->T",
        ["image", "audio"],
        "text",
        "triplet",
        [
            ("clip", "CLIP(I->T)", "image"),
            ("clap", "CLAP(A->T)", "audio"),
        ],
    ),
    (
        "(T+A)->I",
        ["text", "audio"],
        "image",
        "triplet",
        [
            ("clip", "CLIP(T->I)", "text"),
            ("ais", "AIS(A->I)", "audio"),
        ],
    ),
    (
        "(T+I)->A",
        ["text", "image"],
        "audio",
        "triplet",
        [
            ("clap", "CLAP(T->A)", "text"),
            ("ais", "AIS(I->A)", "image"),
        ],
    ),
]

# Feature cache


def _sample_caps(args) -> tuple[int | None, dict[str, int]]:
    limit = getattr(args, "max_samples", None)
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("--max-samples must be a positive integer")
    raw = getattr(args, "max_samples_by_target", None)
    overrides = json.loads(raw) if isinstance(raw, str) else (raw or {})
    if not isinstance(overrides, dict) or set(overrides) - {"text", "image", "audio"}:
        raise ValueError("--max-samples-by-target must map text/image/audio to positive integer caps")
    if any(type(value) is not int or value <= 0 for value in overrides.values()):
        raise ValueError("--max-samples-by-target caps must be positive integers")
    return limit, overrides


def _max_samples_for(args, tgt_modality: str) -> int | None:
    """Return a target override or the global sample cap admitted by preflight."""
    return (args.max_samples_by_target or {}).get(tgt_modality, args.max_samples)


def _decode_image_metric_features(
    gen_feats,
    metric_hub,
    eval_cfg,
    args,
    device,
    *,
    need_fid: bool = False,
):
    """Decode image features into bounded-memory FID and CLIP features."""
    fid_parts: list[np.ndarray] = []
    clip_parts: list[torch.Tensor] = []
    for images in iter_decode_image_chunks(
        gen_feats,
        device=device,
        batch_size=args.decode_batch_size,
        checkpoint_paths=eval_cfg.get("checkpoint_paths"),
        decoder_name=eval_cfg.get("decoders", {}).get("image"),
        renderer_settings=eval_cfg.get("image_renderer"),
        show_progress=True,
    ):
        if need_fid:
            fid_parts.append(
                metric_hub["fid"].extract_features(images, value_range="zero_one")
            )
        clip_parts.append(
            clip_encode_images(images, device, batch_size=args.metric_batch_size)
        )
    return {
        "fid": np.concatenate(fid_parts) if fid_parts else np.empty((0, 2048)),
        "clip": torch.cat(clip_parts) if clip_parts else torch.empty((0, 512)),
    }


# Alignment dispatch


def _fidelity_score(
    kind: str,
    *,
    gen_decoded,
    common: list[str],
    ds_cfg: dict,
    metric_hub: dict,
    gen_image_features: np.ndarray | None = None,
) -> float:
    if kind == "fid":
        if gen_image_features is not None:
            return metric_hub["fid"].compute_from_features(
                gen_image_features,
                reference_dir=ds_cfg.get("images_dir"),
                fid_statistics_file=ds_cfg.get("fid_stats"),
            )
        return metric_hub["fid"].compute_for(
            gen_decoded,
            reference_dir=ds_cfg.get("images_dir"),
            fid_statistics_file=ds_cfg.get("fid_stats"),
            value_range="zero_one",
        )
    if kind == "fad":
        wavs = np.stack(gen_decoded) if isinstance(gen_decoded, list) else gen_decoded
        return metric_hub["fad"].compute(
            wavs,
            reference_audio_dir=ds_cfg.get("audio_dir"),
            fad_stats_file=ds_cfg.get("fad_stats"),
        )
    if kind == "cider":
        reference_path = ds_cfg.get("cider_references_json")
        captions = load_captions_json(reference_path)
        if captions is None:
            raise ValueError("CIDEr requires cider_references_json")
        preds = {k: [t] for k, t in zip(common, gen_decoded)}
        missing = [key for key in common if key not in captions]
        if missing:
            raise ValueError(f"{reference_path}: CIDEr reference captions missing keys {missing[:3]}")
        refs = {key: captions[key] for key in common}
        return metric_hub["cider"].compute(preds, refs)
    raise ValueError(f"Unknown fidelity kind: {kind!r}")


def _selection(value: str | None, allowed: set[str], option: str) -> set[str] | None:
    if value is None:
        return None
    selected = {part.strip() for part in value.split(",") if part.strip()}
    if not selected or selected - allowed:
        raise ValueError(f"{option} must select from {sorted(allowed)}; got {value!r}")
    return selected


def _selected_one_to_one_tasks(args):
    benchmarks = _selection(args.benchmarks, {task[3] for task in ONE_TO_ONE_TASKS}, "--benchmarks")
    targets = _selection(args.targets, {task[2] for task in ONE_TO_ONE_TASKS}, "--targets")
    selected = [
        task for task in ONE_TO_ONE_TASKS
        if (benchmarks is None or task[3] in benchmarks)
        and (targets is None or task[2] in targets)
    ]
    if not selected:
        raise ValueError("--benchmarks and --targets select no one-to-one routes")
    return selected


def _preflight_eval_inputs(eval_cfg, args, tasks: set[str]) -> None:
    """Require references only for selected fidelity routes."""
    _, args.max_samples_by_target = _sample_caps(args)
    datasets = eval_cfg.get("datasets", {})
    if "many2one" in tasks and not datasets.get("triplet"):
        raise ValueError("many2one requires the triplet evaluation dataset")
    if "one2one" not in tasks:
        return
    reference_keys = {"fid": "fid_stats", "fad": "fad_stats", "cider": "cider_references_json"}
    for display, _, target, dataset, kind, _ in _selected_one_to_one_tasks(args):
        cap = _max_samples_for(args, target)
        if kind in {"fid", "fad"} and cap is not None and cap < 2:
            raise ValueError(f"{display}: {kind.upper()} requires at least two generated samples; effective {target} cap is {cap}")
        configured = datasets.get(dataset)
        if not configured:
            raise ValueError(f"{display}: evaluation dataset {dataset!r} is not configured; select available --benchmarks explicitly")
        reference_key = reference_keys[kind]
        path = configured.get(reference_key)
        if not path or not Path(path).is_file():
            raise FileNotFoundError(f"{display}: required {reference_key} file not found: {path}")


# Per-task evaluators


def evaluate_conditional(
    model,
    source_cache,
    metric_hub,
    eval_cfg,
    args,
    device,
    feature_cache: EvaluationFeatureCache,
    decoder_options,
    tasks,
):
    """Execute ordered direct and aggregate cohorts through one media route."""
    fidelity, alignment, many2one = OrderedDict(), OrderedDict(), OrderedDict()
    for protocol in ("one2one", "many2one"):
        if protocol not in tasks:
            continue
        routes = _selected_one_to_one_tasks(args) if protocol == "one2one" else MANY_TO_ONE_TASKS
        for route in routes:
            if protocol == "one2one":
                display, src, tgt, ds_key, fid_kind, align_kind = route
                src_mods = [src]
                metrics_spec = [(align_kind, None, src)]
            else:
                display, src_mods, tgt, ds_key, metrics_spec = route
                fid_kind = None
            ds_cfg = eval_cfg.get("datasets", {}).get(ds_key)
            if protocol == "many2one" and ds_cfg is None:
                print(f"[many2one] Skipping {display}: dataset '{ds_key}' not in eval config")
                for _, label, _ in metrics_spec:
                    many2one[f"{display} {label}"] = float("nan")
                continue

            print(f"\n{'─' * 60}\n[{protocol}] {display}  (dataset: {ds_key})\n{'─' * 60}")
            common, aligned = feature_cache.get_aligned_subset(
                ds_key, ds_cfg, src_mods + [tgt], _max_samples_for(args, tgt)
            )
            if not aligned or not common:
                if protocol == "one2one":
                    raise ValueError(f"{display}: {ds_key} requires aligned {src} and {tgt} features")
                print("  Missing features — skipping")
                for _, label, _ in metrics_spec:
                    many2one[f"{display} {label}"] = float("nan")
                continue
            print(f"  Aligned samples: {len(common)}")

            t0 = time.time()
            gen_feats = translate_features(
                model,
                {m: aligned[m] for m in src_mods},
                src_mods,
                tgt,
                args.translate_batch_size,
                device,
                mode="direct" if protocol == "one2one" else "aggregate",
                sample_kwargs=decoder_sample_kwargs(decoder_options, tgt),
                show_progress=True,
            )
            print(f"  Translation: {time.time() - t0:.1f}s")

            image_features = None
            if tgt == "image":
                if protocol == "one2one":
                    t0 = time.time()
                image_features = _decode_image_metric_features(
                    gen_feats, metric_hub, eval_cfg, args, device, need_fid=fid_kind == "fid"
                )
                gen_decoded = None
                if protocol == "one2one":
                    print(
                        f"  Decoding image + metric features: {time.time() - t0:.1f}s "
                        f"({len(common)} samples)"
                    )
            elif isinstance(gen_feats, list) and gen_feats and isinstance(gen_feats[0], str):
                gen_decoded = gen_feats
                if protocol == "one2one":
                    print(f"  Decoding {tgt}: skipped (raw text from AR decoder, {len(common)} samples)")
            else:
                if protocol == "one2one":
                    t0 = time.time()
                gen_decoded = decode_features(
                    gen_feats, tgt,
                    device=device,
                    batch_size=args.decode_batch_size,
                    checkpoint_paths=eval_cfg.get("checkpoint_paths"),
                    decoder_name=eval_cfg.get("decoders", {}).get(tgt),
                    renderer_settings=eval_cfg.get("image_renderer"),
                    show_progress=True,
                )
                if protocol == "one2one":
                    print(f"  Decoding {tgt}: {time.time() - t0:.1f}s  ({len(common)} samples)")

            if protocol == "one2one":
                if fid_kind:
                    print(f"  Computing {fid_kind.upper()} …")
                    fidelity[display] = _fidelity_score(
                        fid_kind,
                        gen_decoded=gen_decoded,
                        common=common,
                        ds_cfg=ds_cfg,
                        metric_hub=metric_hub,
                        gen_image_features=image_features["fid"] if image_features is not None else None,
                    )
                    print(f"  Fidelity ({fid_kind.upper()}): {fidelity[display]:.4f}")
                else:
                    fidelity[display] = float("nan")
                    print("  Fidelity: skipped")
                print("  Computing alignment …")

            for metric_type, label, src_for_metric in metrics_spec:
                src_decoded = source_cache.get(
                    ds_key, src_for_metric, aligned[src_for_metric], common, ds_cfg
                )
                score, _ = compute_alignment(
                    metric_type,
                    src_for_metric,
                    src_decoded,
                    gen_decoded,
                    device,
                    metric=metric_hub[metric_type],
                    target_image_features=image_features["clip"] if image_features is not None else None,
                )
                if protocol == "one2one":
                    alignment[display] = score
                    print(f"  Alignment ({metric_type.upper()}): {score:.4f}")
                else:
                    many2one[f"{display} {label}"] = score
                    print(f"  {label}: {score:.4f}")
    return fidelity, alignment, many2one


@torch.no_grad()
def evaluate_unconditional(
    model,
    metric_hub,
    eval_cfg,
    args,
    device,
    prior_sample_kwargs,
    decoder_options,
):
    """Measure coherence of modalities decoded from shared prior samples."""
    n_samples = int(args.num_uncond_samples)
    if n_samples <= 0:
        return {}

    misc.seed_everything(int(args.seed))
    images: list[torch.Tensor] = []
    audio: list[np.ndarray] = []
    text: list[str] = []
    batch_size = max(1, int(args.translate_batch_size))

    batches = iter_prior_batches(
        model, model.modalities, n_samples, batch_size, device,
        prior_sample_kwargs=prior_sample_kwargs, decoder_options=decoder_options,
    )
    for joint in tqdm(
        batches, total=(n_samples + batch_size - 1) // batch_size, desc="[uncond] batches"
    ):

        for modality, output in joint.items():
            decoded = decode_generated_output(
                output,
                modality,
                device=device,
                batch_size=args.decode_batch_size,
                checkpoint_paths=eval_cfg.get("checkpoint_paths"),
                decoder_name=eval_cfg.get("decoders", {}).get(modality),
                renderer_settings=eval_cfg.get("image_renderer"),
                show_progress=False,
            )
            if modality == "image":
                images.append(decoded)
            elif modality == "audio":
                audio.extend(decoded)
            elif modality == "text":
                text.extend(decoded)

    image = torch.cat(images, dim=0)[:n_samples]
    audio = audio[:n_samples]
    text = text[:n_samples]
    clip, _ = compute_alignment("clip", "text", text, image, device, metric=metric_hub["clip"])
    clap, _ = compute_alignment("clap", "text", text, audio, device, metric=metric_hub["clap"])
    ais, _ = compute_alignment("ais", "audio", audio, image, device, metric=metric_hub["ais"])
    return {
        "n_samples": n_samples,
        "alignment": {
            "CLIP(T,I)": float(clip),
            "CLAP(T,A)": float(clap),
            "AIS(I,A)": float(ais),
        },
    }


# Pretty-printing


def _fmt(v: float) -> str:
    return f"{v:.2f}" if not (math.isnan(v) or math.isinf(v)) else "—"


def print_results_tables(fidelity, alignment, many2one, model_name="Model"):
    w = 78
    headers = ["", "T->I", "I->T", "T->A", "A->T", "I->A", "A->I"]
    col = 10

    print(
        f"\n{'=' * w}\nTable 2: Fidelity Assessment — One-to-One Benchmarks\n{'=' * w}"
    )
    metrics_row = ["", "FID↓", "CIDEr↑", "FAD↓", "CIDEr↑", "FAD↓", "FID↓"]
    vals = [model_name] + [_fmt(fidelity.get(h, float("nan"))) for h in headers[1:]]
    print("".join(h.rjust(col) for h in headers))
    print("".join(m.rjust(col) for m in metrics_row))
    print("".join(v.rjust(col) for v in vals))

    print(f"\n{'=' * w}\nTable 3: Alignment Results — One-to-One Benchmarks\n{'=' * w}")
    metrics_row2 = ["", "CLIP↑", "CLIP↑", "CLAP↑", "CLAP↑", "AIS↑", "AIS↑"]
    vals2 = [model_name] + [_fmt(alignment.get(h, float("nan"))) for h in headers[1:]]
    print("".join(h.rjust(col) for h in headers))
    print("".join(m.rjust(col) for m in metrics_row2))
    print("".join(v.rjust(col) for v in vals2))

    def _print_multi_table(title, data):
        if not data:
            return
        print(f"\n{'=' * w}\n{title}\n{'=' * w}")
        # Group by task: e.g. "(I+A)->T CLIP(I->T)" -> task="(I+A)->T"
        tasks_seen = OrderedDict()
        for full_key in data:
            parts = full_key.split(" ", 1)
            task = parts[0]
            metric = parts[1] if len(parts) > 1 else full_key
            tasks_seen.setdefault(task, []).append((metric, data[full_key]))
        header1 = f"{'':>12}"
        header2 = f"{'Model':>12}"
        vals_line = f"{model_name:>12}"
        for task, metrics_list in tasks_seen.items():
            span = len(metrics_list) * 12
            header1 += task.center(span)
            for metric_name, score in metrics_list:
                header2 += metric_name.rjust(12)
                vals_line += _fmt(score).rjust(12)
        print(header1)
        print(header2)
        print(vals_line)

    _print_multi_table("Table 4: Many-to-One Generation Alignment", many2one)
    print(f"{'=' * w}\n")


# CLI / Main


def write_run_results(output_dir, method, tasks, results):
    """Create an independent report; exclusive creation forbids overwriting."""
    prefix = re.sub(r"[^A-Za-z0-9_-]+", "-", method).strip("-") or "method"
    path = Path(output_dir) / f"{prefix}_{'-'.join(sorted(tasks))}_{uuid.uuid4().hex}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(results, indent=2, allow_nan=False)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(content + "\n")
    return path


def build_parser(*, prog: str | None = None) -> argparse.ArgumentParser:
    p = build_benchmark_parser(
        "Benchmark evaluation for cross-modal generation",
        prog=prog,
    )
    p.add_argument(
        "--config",
        type=str,
        required=True,
        help="Model config YAML (e.g. configs/image_text_audio/muni_poe.yaml)",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a MUNI model checkpoint (.pth).",
    )
    p.add_argument(
        "--eval-config",
        type=str,
        required=True,
        help="Evaluation config YAML with test dataset paths",
    )
    p.add_argument(
        "--tasks",
        type=str,
        default="all",
        help="Comma-separated: one2one, many2one, uncond, all",
    )
    p.add_argument(
        "--num-uncond-samples",
        type=int,
        default=1000,
        help="Number of shared-prior triples for unconditional coherence.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Evaluation RNG seed (paper protocol: 42).",
    )
    p.add_argument(
        "--benchmarks",
        type=str,
        default=None,
        help="Filter one2one by dataset key. Comma-separated: text-image, text-audio, image-audio (default: all)",
    )
    p.add_argument(
        "--output-dir", type=str, default=None, help="Directory to save results JSON"
    )
    p.add_argument(
        "--num-samples",
        "--max-samples",
        dest="max_samples",
        type=int,
        default=None,
        help="Limit number of test samples per dataset (for debugging)",
    )
    p.add_argument(
        "--max-samples-by-target",
        type=str,
        default=None,
        help="Per-target-modality sample cap, JSON: '{\"image\": 256}'. "
        "Overrides --max-samples for the listed target modalities.",
    )
    p.add_argument(
        "--batch-size",
        "--translate-batch-size",
        dest="translate_batch_size",
        type=int,
        default=256,
        help="Batch size for model translation",
    )
    p.add_argument(
        "--decode-batch-size",
        type=int,
        default=8,
        help="Batch size for decoding features to outputs",
    )
    p.add_argument(
        "--metric-batch-size",
        type=int,
        default=64,
        help="Batch size for metric computation (FID, FAD, CLIPScore, etc.)",
    )
    p.add_argument(
        "--checkpoint-key",
        type=str,
        default="",
        help="State dict key in checkpoint (default: model_ema1)",
    )
    p.add_argument(
        "--no-compile", action="store_true", help="Disable torch.compile for backbones"
    )
    p.add_argument(
        "--method",
        "--model-name",
        dest="model_name",
        type=str,
        default="MUNI",
        help="Method name for display in result tables",
    )
    p.add_argument(
        "--sampler-eps",
        type=float,
        default=None,
        help="Override Flow prior/image decoder denom_eps.",
    )
    p.add_argument(
        "--image-steps",
        type=int,
        default=None,
        help="Override image flow decoder integration steps (sampler_steps).",
    )
    p.add_argument(
        "--cfg-scale",
        type=float,
        default=None,
        help="Override decoder CFG scale for MUNI translation sampling. Default uses the model config.",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override decoder init-noise temperature.",
    )
    p.add_argument(
        "--cfg-rescale-phi",
        type=float,
        default=0.0,
        help="CFG rescale mix coefficient. 1.0 = full rescale.",
    )
    p.add_argument(
        "--targets",
        type=str,
        default=None,
        help="Comma-separated target modalities to keep in one2one (e.g. 'text' = skip T->I/T->A).",
    )
    return p


def run(args: argparse.Namespace) -> int:
    """Execute the image-text-audio benchmark for already-parsed arguments."""
    for name in ("translate_batch_size", "decode_batch_size", "metric_batch_size"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")

    tasks = _selection(args.tasks.lower(), {"all", "one2one", "many2one", "uncond"}, "--tasks")
    if "all" in tasks:
        tasks = {"one2one", "many2one", "uncond"}

    eval_cfg = OmegaConf.to_container(load_config(args.eval_config), resolve=True)
    _preflight_eval_inputs(eval_cfg, args, tasks)
    device = resolve_device()
    misc.seed_everything(int(args.seed))
    if device.type == "cuda":
        cudnn.benchmark = True

    print("Loading model …")
    model, model_cfg = load_evaluation_model(args, device)
    prior_sample_kwargs, decoder_options = resolve_cli_sample_kwargs(model_cfg, args)
    prior_steps, decoder_steps = resolve_sampling_steps(model, prior_sample_kwargs, decoder_options)
    decoder_names, checkpoint_paths, renderer_settings = resolve_decoder_config(model_cfg, eval_cfg)
    eval_cfg["decoders"] = decoder_names
    eval_cfg["checkpoint_paths"] = checkpoint_paths
    eval_cfg["image_renderer"] = renderer_settings
    print(f"Image renderer: {decoder_names['image']} {renderer_settings}")

    # Source media is reused only for the same ordered sample keys.
    source_cache = SourceDecodeCache(
        device=device,
        checkpoint_paths=checkpoint_paths,
        decoder_names=decoder_names,
        decode_batch_size=args.decode_batch_size,
        renderer_settings=renderer_settings,
    )
    feature_cache = EvaluationFeatureCache()

    metric_hub = {
        "clip": CLIPScore(device, batch_size=args.metric_batch_size),
        "clap": CLAPScore(device, batch_size=args.metric_batch_size),
        "ais": AudioImageSimilarity(device, image_batch_size=args.metric_batch_size),
        "fid": ImageFID(
            num_samples=2**31 - 1,
            batch_size=args.metric_batch_size,
            device=device,
        ),
        "fad": AudioFAD(device=str(device), batch_size=args.metric_batch_size),
        "cider": CIDEr(),
    }

    fidelity, alignment_res, many2one = OrderedDict(), OrderedDict(), OrderedDict()
    uncond = {}

    if tasks & {"one2one", "many2one"}:
        fidelity, alignment_res, many2one = evaluate_conditional(
            model,
            source_cache,
            metric_hub,
            eval_cfg,
            args,
            device,
            feature_cache,
            decoder_options,
            tasks,
        )
    if "uncond" in tasks:
        uncond = evaluate_unconditional(
            model,
            metric_hub,
            eval_cfg,
            args,
            device,
            prior_sample_kwargs,
            decoder_options,
        )

    print_results_tables(fidelity, alignment_res, many2one, args.model_name)

    out_dir = args.output_dir or (
        str(Path(args.checkpoint).parent / "benchmark")
        if args.checkpoint
        else "benchmark_results"
    )
    all_results = {
        "fidelity": dict(fidelity),
        "alignment": dict(alignment_res),
        "many_to_one": dict(many2one),
        "uncond": uncond,
        "protocol": {
            "method": args.model_name,
            "config": str(Path(args.config).resolve()),
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_key": args.checkpoint_key,
            "eval_config": str(Path(args.eval_config).resolve()),
            "seed": int(args.seed),
            "tasks": sorted(tasks),
            "filters": {"benchmarks": args.benchmarks, "targets": args.targets},
            "selected_one_to_one": [
                task[0] for task in _selected_one_to_one_tasks(args)
            ] if "one2one" in tasks else [],
            "requested_caps": {"max_samples": args.max_samples,
                               "by_target": args.max_samples_by_target,
                               "unconditional": args.num_uncond_samples},
            "cohorts": feature_cache.cohort_counts(),
            "unconditional_samples": uncond.get("n_samples", 0),
            "batch_sizes": {"translate": args.translate_batch_size,
                            "decode": args.decode_batch_size,
                            "metric": args.metric_batch_size},
            "metrics": {"ais": {"audio_batch_size": AudioImageSimilarity.audio_batch_size}},
            "sampling": {"prior_steps": prior_steps,
                         "decoder_steps": decoder_steps,
                         "prior_kwargs": prior_sample_kwargs,
                         "decoder_kwargs": decoder_options,
                         "prior": OmegaConf.to_container(model_cfg.model.routing.prior, resolve=True),
                         "decoders": OmegaConf.to_container(model_cfg.model.routing.decoders, resolve=True)},
            "rendering": {"decoders": decoder_names,
                          "checkpoint_paths": (
                              OmegaConf.to_container(checkpoint_paths, resolve=True)
                              if OmegaConf.is_config(checkpoint_paths) else checkpoint_paths
                          ),
                          "image_renderer": renderer_settings},
        },
    }

    out_path = write_run_results(out_dir, args.model_name, tasks, all_results)
    print(f"Results saved to {out_path}")
    return 0


def main(argv: Sequence[str] | None = None, *, prog: str | None = None) -> int:
    return run_cli(build_parser, run, argv, prog=prog)
