"""Regenerate reference statistics from local encoded media."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

import numpy as np
from omegaconf import OmegaConf
from muni.config_utils import load_config


def arguments(kind: str):
    parser = argparse.ArgumentParser(description=f"Regenerate {kind.upper()} reference statistics from encoded media.")
    parser.add_argument("--config", default="configs/data/image_text_audio_references.yaml")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true", help="Check media counts and unique IDs without inference or outputs")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return args


def configuration(args, kind: str):
    config = OmegaConf.to_container(load_config(args.config), resolve=True)
    if not isinstance(config, dict):
        raise ValueError("Reference config must be a mapping")
    modality = "images" if kind == "fid" else "audio"
    supported = [name for name, entry in config["datasets"].items() if modality in entry]
    names = supported if args.datasets is None else args.datasets
    if not names or len(set(names)) != len(names):
        raise ValueError("--datasets must select at least one unique supported cohort")
    for name in names:
        if name not in supported or Path(name).name != name or name in (".", ".."):
            raise ValueError(f"Unsupported {kind.upper()} dataset {name!r}; choose from {supported}")
    return config, names


class MediaSource:
    """All input records, retaining only IDs, locations and one bounded Parquet batch."""

    def __init__(self, name: str, config: dict, modality: str):
        self.name = name
        expected = config["samples"]
        if type(expected) is not int or expected < 2:
            raise ValueError(f"{name}: samples must be an integer of at least two")
        self.keys = []
        self.locations = []
        self.root = Path(config["images" if modality == "image" else "audio"]).expanduser()
        self.parquet = None
        self._cached_group = None
        self._cached_start = 0
        self._batch_iterator = None
        self._cached_rows = None
        seen = set()
        if self.root.is_dir():
            extensions = ({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
                          if modality == "image" else {".wav", ".flac", ".ogg", ".aif", ".aiff", ".mp3"})
            for path in sorted(self.root.rglob("*")):
                if path.is_file() and path.suffix.lower() in extensions:
                    self._accept(path.stem, path, seen)
        elif self.root.is_file() and self.root.suffix.lower() == ".parquet":
            import pyarrow.parquet as pq
            self.parquet = pq.ParquetFile(self.root)
            if not {"stem", "bytes"}.issubset(self.parquet.schema.names):
                raise ValueError(f"{name}: packed media requires Parquet stem,bytes columns: {self.root}")
            if self.parquet.metadata.num_rows != expected:
                raise ValueError(f"{name}: expected {expected} {modality} records, found {self.parquet.metadata.num_rows}")
            for group in range(self.parquet.num_row_groups):
                offset = 0
                for batch in self.parquet.iter_batches(batch_size=64, row_groups=[group], columns=["stem"]):
                    for index, key in enumerate(batch.column("stem").to_pylist()):
                        self._accept(key, (group, offset + index), seen)
                    offset += batch.num_rows
        else:
            raise ValueError(f"{name}: expected encoded media directory or packed Parquet file: {self.root}")
        if len(self) != expected:
            raise ValueError(f"{name}: expected {expected} {modality} records, found {len(self)}")

    def _accept(self, key, location, seen):
        if not isinstance(key, str) or not key or Path(key).name != key or key in (".", ".."):
            raise ValueError(f"{self.name}: invalid input stem {key!r} at {location}")
        if key in seen:
            raise ValueError(f"{self.name}: duplicate input stem {key!r}")
        seen.add(key)
        self.keys.append(key)
        self.locations.append(location)

    def __len__(self):
        return len(self.keys)

    def encoded(self, index: int) -> bytes:
        location = self.locations[index]
        if self.parquet is None:
            blob = location.read_bytes()
        else:
            group, row = location
            if self._cached_group != group or row < self._cached_start:
                self._batch_iterator = self.parquet.iter_batches(
                    batch_size=64, row_groups=[group], columns=["bytes"]
                )
                self._cached_group = group
                self._cached_start = 0
                self._cached_rows = next(self._batch_iterator).column("bytes")
            while row >= self._cached_start + len(self._cached_rows):
                self._cached_start += len(self._cached_rows)
                self._cached_rows = next(self._batch_iterator).column("bytes")
            blob = self._cached_rows[row - self._cached_start].as_py()
        if not isinstance(blob, bytes) or not blob:
            raise ValueError(f"{self.name}: invalid encoded media for input {self.keys[index]!r}")
        return blob


def save_statistics(path: Path, mu, sigma):
    """Publish a complete NPZ atomically without replacing an existing result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            np.savez(handle, mu=np.asarray(mu, dtype=np.float32), sigma=np.asarray(sigma, dtype=np.float64))
            handle.flush()
            os.fsync(handle.fileno())
        # Exclusive creation also protects against concurrent writers.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run(kind: str, prepare):
    args = arguments(kind)
    try:
        config, names = configuration(args, kind)
        modality = "image" if kind == "fid" else "audio"
        filename = "fid_stats.npz" if kind == "fid" else "fad_stats_16k.npz"
        cohorts = []
        for name in names:
            output = Path(config["output_root"]).expanduser() / name / filename
            if not args.dry_run and output.exists():
                raise ValueError(f"Output exists: {output}; choose a fresh output_root or remove the existing result")
            if not args.dry_run and output.parent.exists() and any(output.parent.glob(f".{filename}.*.tmp")):
                raise ValueError(f"Partial result in {output.parent}; inspect/remove the interrupted temporary file first")
            cohort = MediaSource(name, config["datasets"][name], modality)
            cohorts.append((cohort, output))
            print(f"Checked {name}: {len(cohort)} {modality} records")
        if args.dry_run:
            print(f"{kind.upper()} dry-run complete: media counts and unique IDs checked")
            return
        compute = prepare(args)
        for cohort, output in cohorts:
            mu, sigma = compute(cohort)
            save_statistics(output, mu, sigma)
            print(f"Saved {output}: mu {mu.shape}, sigma {sigma.shape}")
    except (ValueError, OSError, KeyError, ImportError) as error:
        raise SystemExit(f"{kind.upper()} reference preparation: {error}") from error
