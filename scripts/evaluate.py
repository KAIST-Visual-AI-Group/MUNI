"""Standalone config-driven evaluation entrypoint."""

import argparse
from collections.abc import Sequence
import json
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

from muni.config_utils import (
    ConfigOverrideError,
    MissingConfigEnvironmentError,
    load_config,
    print_config,
)
from muni.data import build_dataset, build_eval_loader
from muni.evaluation.checkpoints import (
    load_ema_checkpoint,
)
from muni.evaluation.runner import run_evaluation_suite
from muni.framework.model import build_model
import muni.util.misc as misc
from muni.util.misc import print_on_main as print


def _resolve_config_path(config_arg: str | None, checkpoint_path: Path) -> Path:
    if config_arg:
        return Path(config_arg)

    candidate = checkpoint_path.parent / "config.yaml"
    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        "Could not infer config.yaml next to checkpoint. Pass --config explicitly."
    )


def _needs_eval_data(cfg) -> bool:
    evaluators = cfg.get("evaluators", None)
    if not evaluators:
        return False
    return any(
        phase in evaluators
        for phase in ("reconstruction", "translation", "information")
    )


def _build_eval_loader(cfg):
    dataset_cfg = cfg.get("dataset_eval", None) or cfg.dataset
    eval_cfg = cfg.get("eval", {}) or {}
    return build_eval_loader(
        build_dataset(dataset_cfg),
        batch_size=int(eval_cfg.get("batch_size", cfg.train.batch_size)),
        num_workers=int(eval_cfg.get("num_workers", cfg.train.num_workers)),
        pin_memory=bool(eval_cfg.get("pin_mem", cfg.train.pin_mem)),
        num_replicas=misc.get_world_size(),
        rank=misc.get_rank(),
    )


def _print_summary(
    *,
    checkpoint_path: Path,
    config_path: Path,
    epoch: int,
    state_key: str,
    load_info: dict[str, object],
    results,
    results_path: Path,
) -> None:
    print("\n" + "=" * 72)
    print("Standalone Evaluation Summary")
    print("=" * 72)
    print(f"checkpoint : {checkpoint_path}")
    print(f"config     : {config_path}")
    print(f"epoch      : {epoch}")
    print(f"state key  : {state_key}")
    print(f"key adapt  : {load_info['translated_keys']} translated")
    print(f"results    : {results_path}")
    print(json.dumps(results, indent=2, sort_keys=True))
    print("=" * 72)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "Standalone config-driven evaluation",
        epilog=(
            "Config overrides: key=value for existing keys; "
            "+key=value for intentional new keys."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--config",
        default="",
        help="Config path. Defaults to config.yaml beside the checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-key",
        default="",
        help="State dict key. Defaults to model_ema1, then model.",
    )
    parser.add_argument(
        "--save-tag",
        default="default",
        help="Suffix for standalone_eval_<tag>.json.",
    )
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("config_overrides", nargs="*", metavar="key=value")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    checkpoint_path = Path(args.checkpoint).resolve()
    config_path = _resolve_config_path(args.config or None, checkpoint_path).resolve()
    try:
        cfg = load_config(config_path, cli_overrides=args.config_overrides)
    except (ConfigOverrideError, MissingConfigEnvironmentError) as exc:
        parser.error(str(exc))

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    misc.init_distributed_mode(cfg)
    device = torch.device(cfg.distributed.device)
    misc.seed_everything(cfg.train.seed + misc.get_rank())
    cudnn.benchmark = True

    if args.print_config and misc.is_main_process():
        print_config(cfg, title="Standalone Evaluation Configuration")

    model = build_model(cfg)
    state_key, load_info = load_ema_checkpoint(
        model,
        device,
        checkpoint,
        checkpoint_key=args.checkpoint_key or None,
    )
    epoch = int(checkpoint.get("epoch", -1))
    data_loader = _build_eval_loader(cfg) if _needs_eval_data(cfg) else []

    results = run_evaluation_suite(
        model,
        data_loader,
        cfg,
        log_writer=None,
        step=epoch,
        device=device,
    )

    if misc.is_main_process():
        save_tag = args.save_tag.strip() or "default"
        results_path = checkpoint_path.parent / f"standalone_eval_{save_tag}.json"
        with results_path.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, sort_keys=True)
        _print_summary(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            epoch=epoch,
            state_key=state_key,
            load_info=load_info,
            results=results,
            results_path=results_path,
        )

    if misc.is_dist_avail_and_initialized():
        torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
