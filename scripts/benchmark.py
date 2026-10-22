"""Dispatch the task-specific paper benchmark entrypoints."""

import argparse
import importlib
import os
import sys
from collections.abc import Sequence
from types import ModuleType


_BENCHMARK_MODULES = {
    "image-text-audio": "scripts.benchmarks.image_text_audio",
    "polymnist": "scripts.benchmarks.polymnist",
    "gmm": "scripts.benchmarks.gmm",
}


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _build_parser(*, add_help: bool) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.benchmark",
        description="Run the image-text-audio, PolyMNIST-Quadrant-Labels, or GMM benchmark.",
        add_help=add_help,
        allow_abbrev=False,
    )
    parser.add_argument(
        "benchmark",
        choices=tuple(_BENCHMARK_MODULES),
        help="Benchmark suite to run; remaining options are forwarded to that suite.",
    )
    parser.add_argument(
        "--gpu",
        type=_non_negative_int,
        default=None,
        help="Expose one physical GPU before importing the selected benchmark.",
    )
    return parser


def build_parser() -> argparse.ArgumentParser:
    return _build_parser(add_help=True)


def _load_benchmark_module(benchmark: str) -> ModuleType:
    return importlib.import_module(_BENCHMARK_MODULES[benchmark])


def main(argv: Sequence[str] | None = None) -> int:
    cli_args = list(sys.argv[1:] if argv is None else argv)
    if not cli_args or cli_args[0] in {"-h", "--help"}:
        build_parser().parse_args(cli_args)

    args, forwarded = _build_parser(add_help=False).parse_known_args(cli_args)
    if args.gpu is not None:
        # An explicit CLI choice takes precedence over an inherited shell
        # mask.  ``setdefault`` silently selected the wrong physical device
        # when CUDA_VISIBLE_DEVICES already named multiple GPUs.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    module = _load_benchmark_module(args.benchmark)
    return int(
        module.main(
            forwarded,
            prog=f"python -m scripts.benchmark {args.benchmark}",
        )
        or 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
