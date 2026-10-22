"""MUNI checkpoint loading shared by executable evaluation workflows."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, open_dict

from muni.config_utils import load_config
from muni.evaluation.checkpoints import build_muni_model_from_checkpoint
from muni.evaluation.sampling import decoder_sample_kwargs, resolve_sample_kwargs
from muni.framework.model import MultimodalWrapper


def load_benchmark_config(
    config_path: str,
    checkpoint_path: str,
    *,
    no_compile: bool,
) -> DictConfig:
    """Load a single-process benchmark configuration."""
    cfg = load_config(config_path)
    with open_dict(cfg):
        cfg.checkpoint.output_dir = str(Path(checkpoint_path).parent)
        cfg.train.compile_backbones = not no_compile
        cfg.distributed.distributed = False
    return cfg


def load_evaluation_model(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[MultimodalWrapper, DictConfig]:
    """Build a MUNI model and install checkpoint weights."""
    cfg = load_config(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    with open_dict(cfg):
        if args.no_compile:
            cfg.train.compile_backbones = False

    model = build_muni_model_from_checkpoint(
        cfg,
        args.checkpoint,
        device,
        checkpoint_key=args.checkpoint_key or None,
        checkpoint=checkpoint,
    )

    return model, cfg


def resolve_cli_sample_kwargs(
    cfg: DictConfig,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Translate shared evaluation flags into role-scoped adapter options."""
    prior: dict[str, Any] = {}
    decoder: dict[str, Any] = {}
    if args.sampler_eps is not None:
        prior["sampler_eps"] = float(args.sampler_eps)
        decoder["sampler_eps"] = float(args.sampler_eps)
    if args.image_steps is not None:
        decoder["sampler_steps"] = int(args.image_steps)
    if args.temperature is not None:
        decoder["temperature"] = float(args.temperature)
    if args.cfg_scale is not None:
        decoder["cfg_scale"] = float(args.cfg_scale)
    if args.cfg_rescale_phi > 0.0:
        decoder["cfg_rescale_phi"] = float(args.cfg_rescale_phi)
    return resolve_sample_kwargs(
        cfg,
        prior_overrides=prior,
        decoder_overrides={"image": decoder},
    )


def resolve_sampling_steps(
    model: MultimodalWrapper,
    prior_kwargs: dict[str, Any],
    decoder_options: dict[str, dict[str, Any]],
) -> tuple[int | None, dict[str, int]]:
    """Read effective integration budgets from the samplers and runtime options."""
    prior_steps = prior_kwargs.get("sampler_steps")
    if prior_steps is None:
        prior_steps = getattr(model.prior, "sampler_steps", None)
    decoder_steps = {}
    for modality, decoder in model.decoders.items():
        if hasattr(decoder, "sampler_steps"):
            steps = decoder_sample_kwargs(decoder_options, modality).get("sampler_steps")
            decoder_steps[modality] = int(decoder.sampler_steps if steps is None else steps)
    return prior_steps, decoder_steps
