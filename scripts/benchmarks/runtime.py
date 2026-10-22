"""Process and filesystem primitives shared by paper benchmark entrypoints.

The four protocols share metric kernels from ``muni.evaluation``. This module
owns their common command-line/filesystem lifecycle and report-number semantics.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import math
import os
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Iterator

import torch


def build_benchmark_parser(
    description: str,
    *,
    prog: str | None = None,
) -> argparse.ArgumentParser:
    """Create the consistently formatted parser used by every benchmark."""
    return argparse.ArgumentParser(
        prog=prog,
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )


def run_cli(
    build_parser: Callable[..., argparse.ArgumentParser],
    run: Callable[[argparse.Namespace], int | None],
    argv: Sequence[str] | None = None,
    *,
    prog: str | None = None,
) -> int:
    """Parse one benchmark command and normalize its process exit status."""
    result = run(build_parser(prog=prog).parse_args(argv))
    return int(result or 0)


def resolve_device(*, require_cuda: bool = False) -> torch.device:
    """Resolve the benchmark device without changing backend or RNG policy."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if require_cuda:
        raise RuntimeError("This benchmark requires an available CUDA device.")
    return torch.device("cpu")


def summarize_samples(values: torch.Tensor) -> dict[str, float | int]:
    """Return the shared ``mean/std/n`` payload used in benchmark reports."""
    samples = values.float().flatten().cpu()
    count = int(samples.numel())
    if count == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": float(samples.mean().item()),
        "std": float(samples.std(unbiased=False).item()) if count > 1 else 0.0,
        "n": count,
    }


def metric_mean(entry: Any) -> float | None:
    """Read a report scalar; missing/non-numeric/NaN entries are not ranked."""
    value = entry.get("mean") if isinstance(entry, dict) else entry
    if value is None:
        return None
    try:
        mean = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(mean) else mean


def ci_cell(mean: float, std: float, n: int) -> str:
    """Format a per-sample metric using the benchmark CI arithmetic."""
    ci = float("nan") if n <= 0 or math.isnan(std) else 1.959963984540054 * std / math.sqrt(n)
    return f"{mean:.4f}±{ci:.4f}" if not math.isnan(ci) else f"{mean:.4f}±—"


def markdown_table(
    headers: Sequence[str], rows: Sequence[Sequence[str]], separators: Sequence[str]
) -> list[str]:
    """Emit an already formatted matrix without imposing domain policies."""
    return [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(separators) + "|",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def read_reports(output_dir: Path) -> list[tuple[str, dict[str, Any]]]:
    """Read all reports in stable filename order; empty/malformed inputs fail."""
    rows = []
    for path in sorted(output_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            raise ValueError(f"Cannot read benchmark report {path}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Benchmark report {path} must be an object")
        rows.append((path.stem, payload))
    if not rows:
        raise ValueError(f"No benchmark reports in {output_dir}")
    return rows


def write_summary(
    output_dir: Path, renderer: Callable[[list[tuple[str, dict[str, Any]]]], str]
) -> Path:
    """Own the complete locked scan-render-atomic-publication lifecycle."""
    summary_path = output_dir / "results.md"
    with summary_write_lock(summary_path):
        return _write_text_atomic(summary_path, renderer(read_reports(output_dir)))


@contextmanager
def summary_write_lock(path: str | Path) -> Iterator[None]:
    """Serialize the read-render-write cycle for one benchmark summary.

    The hidden lock is kept beside the summary so concurrent jobs using a
    shared output filesystem coordinate on the same inode.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_path.with_name(f".{output_path.name}.lock")
    with lock_path.open("a", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def write_json(path: str | Path, payload: Any) -> Path:
    """Atomically write an indented benchmark JSON payload."""
    return _write_text_atomic(path, json.dumps(payload, indent=2, allow_nan=False))


def _write_text_atomic(path: str | Path, content: str) -> Path:
    """Write through a unique sibling file, then atomically replace *path*.

    Benchmark methods are commonly evaluated in parallel into one output
    directory. A fixed ``results.md.tmp`` name lets one process rename or
    delete another process's in-progress file. A unique sibling preserves the
    same final-file semantics without that collision.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(content)
            temporary_path = Path(handle.name)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return output_path
