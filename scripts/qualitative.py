"""Generate qualitative image-text-audio co-generation and translation examples."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from omegaconf import OmegaConf

from muni.config_utils import load_config
from scripts.model_loading import (
    load_evaluation_model,
    resolve_cli_sample_kwargs,
    resolve_sampling_steps,
)
from muni.evaluation.sampling import (
    collect_decoder_outputs, decoder_sample_kwargs, iter_prior_decodes,
)
from muni.evaluation.data import EvaluationFeatureCache, align_keys, load_raw_gt, load_test_features
from muni.evaluation.image_text_audio.decoding import (
    decode_features,
    decode_generated_output,
    resolve_decoder_config,
    translate_features,
)
from scripts.qualitative_output import render_captioned_grid, save_texts, save_translation, save_wavs
import muni.util.misc as misc


MODALITIES = ("text", "image", "audio")

# src → tgt dataset-key lookup (matches run_benchmark)
_PAIR_DS_KEY = {
    frozenset({"text", "image"}): "text-image",
    frozenset({"text", "audio"}): "text-audio",
    frozenset({"image", "audio"}): "image-audio",
}

TRANSLATION_PAIRS = [
    ("text", "image"),
    ("image", "text"),
    ("text", "audio"),
    ("audio", "text"),
    ("image", "audio"),
    ("audio", "image"),
]

_ABBR = {"text": "T", "image": "I", "audio": "A"}


# CLI


def parse_args():
    p = argparse.ArgumentParser(description="Qualitative co-gen / translation")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--eval-config", required=True)
    p.add_argument("--output-dir", required=True)

    p.add_argument("--num-unconditional", type=int, default=64)
    p.add_argument("--num-translation", type=int, default=16)
    p.add_argument("--grid-cols", type=int, default=4)

    p.add_argument("--translate-batch-size", type=int, default=64)
    p.add_argument("--decode-batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--model-name", default="Model")
    p.add_argument("--checkpoint-key", default="")
    p.add_argument("--no-compile", action="store_true")

    p.add_argument(
        "--image-steps",
        type=int,
        default=None,
        help="Override image flow decoder integration steps (sampler_steps).",
    )
    p.add_argument("--sampler-eps", type=float, default=None)
    p.add_argument(
        "--cfg-scale",
        type=float,
        default=None,
        help="Override decoder CFG scale (None = model-config default).",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override decoder sampler_temperature (init-noise scale).",
    )
    p.add_argument(
        "--cfg-rescale-phi",
        type=float,
        default=0.0,
        help="CFG rescale mix coefficient. 1.0 = full rescale.",
    )

    p.add_argument("--skip-unconditional", action="store_true")
    p.add_argument("--skip-translation", action="store_true")
    p.add_argument(
        "--pairs",
        default=None,
        help="Comma-separated subset of translation pairs, e.g. 'T2I,A2I'.",
    )
    args = p.parse_args()
    try:
        args.pairs = _selected_pairs(args.pairs)
        for name in ("translate_batch_size", "decode_batch_size"):
            if getattr(args, name) <= 0:
                raise ValueError(f"--{name.replace('_', '-')} must be positive")
    except ValueError as exc:
        p.error(str(exc))
    return args


def _selected_pairs(value):
    if value is None:
        return None
    selected = {tag.strip().upper() for tag in value.split(",") if tag.strip()}
    allowed = {_abbr_tag(src, tgt) for src, tgt in TRANSLATION_PAIRS}
    if not selected or selected - allowed:
        raise ValueError(f"--pairs must select from {sorted(allowed)}; got {sorted(selected)}")
    return selected


# Unconditional co-generation


def run_unconditional(
    model,
    args,
    device,
    out_dir,
    eval_cfg,
    prior_sample_kwargs,
    decoder_options,
):
    n = args.num_unconditional
    cols = args.grid_cols
    per_grid = cols * cols
    if n <= 0:
        return

    uncond_dir = out_dir / "unconditional"
    uncond_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n== Unconditional co-generation: {n} samples ==")

    if not hasattr(model, "sample_prior") or not hasattr(model, "decoders"):
        print(
            "  [skip] model lacks sample_prior / decoders — baseline without a joint prior."
        )
        return

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # Single shared prior across modalities → coherent triples.
    z_parts = []
    for i in range(0, n, args.translate_batch_size):
        b = min(args.translate_batch_size, n - i)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            z_parts.append(
                model.sample_prior(
                    b,
                    device,
                    sample_kwargs=prior_sample_kwargs,
                ).float()
            )
    z_prior = torch.cat(z_parts, dim=0)[:n]

    feats: dict[str, torch.Tensor | list[str]] = {}
    for mod in MODALITIES:
        if mod not in model.decoders:
            print(f"  [skip] decoder for '{mod}' is missing.")
            continue
        feats[mod] = collect_decoder_outputs(
            out.float().cpu() if isinstance(out, torch.Tensor) else out
            for out in iter_prior_decodes(
                model, mod, z_prior.to(device), args.translate_batch_size,
                sample_kwargs=decoder_sample_kwargs(decoder_options, mod),
            )
        )

    ckpt_paths = eval_cfg.get("checkpoint_paths")
    decoders = eval_cfg.get("decoders", {})

    raw: dict[str, list] = {}
    for mod, f in feats.items():
        raw[mod] = decode_generated_output(
            f,
            mod,
            device=device,
            batch_size=args.decode_batch_size,
            checkpoint_paths=ckpt_paths,
            decoder_name=decoders.get(mod),
            renderer_settings=eval_cfg.get("image_renderer"),
            show_progress=True,
        )

    # Image + caption grids
    if "image" in raw and "text" in raw:
        for k, start in enumerate(range(0, n, per_grid)):
            end = min(start + per_grid, n)
            items = [(raw["image"][i], raw["text"][i]) for i in range(start, end)]
            grid = render_captioned_grid(
                items,
                ncols=cols,
                title=f"Unconditional co-gen — samples [{start:02d}..{end - 1:02d}]",
            )
            path = uncond_dir / f"grid_{k:02d}.png"
            grid.save(path)
            print(f"  wrote {path.relative_to(out_dir)}")

    if "audio" in raw:
        paths = save_wavs(raw["audio"], uncond_dir / "audio", sample_rate=16000)
        print(
            f"  wrote {len(paths)} wav(s) to {(uncond_dir / 'audio').relative_to(out_dir)}"
        )

    if "text" in raw:
        save_texts(
            raw["text"],
            uncond_dir / "captions.txt",
            title=f"Unconditional co-generated captions ({len(raw['text'])} samples)",
        )
        print(f"  wrote {(uncond_dir / 'captions.txt').relative_to(out_dir)}")


# Translation


def _abbr_tag(src, tgt) -> str:
    return f"{_ABBR[src]}2{_ABBR[tgt]}"


def _load_translation_batch(eval_cfg, src, tgt, n, *, cache=None):
    """Return ``(keys, src_feats, raw_src)`` or None on miss."""
    ds_key = _PAIR_DS_KEY[frozenset({src, tgt})]
    ds_cfg = eval_cfg.get("datasets", {}).get(ds_key)
    if ds_cfg is None:
        print(f"  [skip] dataset '{ds_key}' missing from eval config")
        return None
    feat_dir = ds_cfg["feat_dir"]

    load_features = load_test_features if cache is None else cache.load_features
    src_feats, src_keys = load_features(feat_dir, src, n)
    tgt_feats, tgt_keys = load_features(feat_dir, tgt, n)
    if src_feats is None or tgt_feats is None:
        print(f"  [skip] features missing for {src}/{tgt} in '{ds_key}'")
        return None
    common, (si, _) = align_keys(src_keys, tgt_keys)
    keys = common[:n]
    if not keys:
        print(f"  [skip] no aligned samples for '{ds_key}'")
        return None
    src_feats = src_feats[si][:n]

    raw_src = load_raw_gt(ds_cfg, src, keys)
    return keys, src_feats, raw_src


def run_translation(
    model,
    args,
    device,
    out_dir,
    eval_cfg,
    decoder_options,
):
    n = args.num_translation
    if n <= 0:
        return

    trans_dir = out_dir / "translation"
    trans_dir.mkdir(parents=True, exist_ok=True)

    wanted_tags = args.pairs

    ckpt_paths = eval_cfg.get("checkpoint_paths")
    decoder_names = eval_cfg.get("decoders", {})
    feature_cache = EvaluationFeatureCache()

    for src, tgt in TRANSLATION_PAIRS:
        tag = _abbr_tag(src, tgt)
        if wanted_tags is not None and tag not in wanted_tags:
            continue
        print(f"\n== Translation {tag}: {n} samples ==")
        bundle = _load_translation_batch(eval_cfg, src, tgt, n, cache=feature_cache)
        if bundle is None:
            continue
        keys, src_feats, raw_src = bundle

        if raw_src is None:
            raw_src = decode_features(
                src_feats,
                src,
                device=device,
                batch_size=args.decode_batch_size,
                checkpoint_paths=ckpt_paths,
                decoder_name=decoder_names.get(src),
                renderer_settings=eval_cfg.get("image_renderer"),
                show_progress=True,
            )

        gen_feats = translate_features(
            model,
            {src: src_feats},
            [src],
            tgt,
            args.translate_batch_size,
            device,
            mode="direct",
            sample_kwargs=decoder_sample_kwargs(decoder_options, tgt),
            show_progress=False,
        )
        raw_tgt = decode_generated_output(
            gen_feats,
            tgt,
            device=device,
            batch_size=args.decode_batch_size,
            checkpoint_paths=ckpt_paths,
            decoder_name=decoder_names.get(tgt),
            renderer_settings=eval_cfg.get("image_renderer"),
            show_progress=True,
        )

        save_translation(
            trans_dir / tag,
            src,
            tgt,
            raw_src,
            raw_tgt,
            tag=tag,
            ncols=args.grid_cols,
        )


# Main


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    misc.seed_everything(args.seed)
    if device.type == "cuda":
        cudnn.benchmark = True

    eval_cfg = OmegaConf.to_container(load_config(args.eval_config), resolve=True)

    print("Loading model …")
    model, model_cfg = load_evaluation_model(args, device)
    prior_sample_kwargs, decoder_options = resolve_cli_sample_kwargs(model_cfg, args)
    prior_steps, decoder_steps = resolve_sampling_steps(
        model, prior_sample_kwargs, decoder_options,
    )
    decoder_names, checkpoint_paths, renderer_settings = resolve_decoder_config(model_cfg, eval_cfg)
    eval_cfg["decoders"] = decoder_names
    eval_cfg["checkpoint_paths"] = checkpoint_paths
    eval_cfg["image_renderer"] = renderer_settings
    print(f"Image renderer: {decoder_names['image']} {renderer_settings}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Leave a small manifest for bookkeeping.
    manifest = {
        "model_name": args.model_name,
        "config": args.config,
        "checkpoint": args.checkpoint,
        "cfg_scale": args.cfg_scale,
        "prior_steps": prior_steps,
        "image_steps": decoder_steps["image"],
        "temperature": args.temperature,
        "cfg_rescale_phi": args.cfg_rescale_phi,
        "num_unconditional": args.num_unconditional,
        "num_translation": args.num_translation,
        "seed": args.seed,
    }
    import json

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if not args.skip_unconditional:
        run_unconditional(
            model,
            args,
            device,
            out_dir,
            eval_cfg,
            prior_sample_kwargs,
            decoder_options,
        )
    if not args.skip_translation:
        run_translation(
            model,
            args,
            device,
            out_dir,
            eval_cfg,
            decoder_options,
        )

    print(f"\nQualitative outputs saved under: {out_dir}")


if __name__ == "__main__":
    main()
