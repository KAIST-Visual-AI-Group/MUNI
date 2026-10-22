"""Build MUNI feature caches from explicitly configured supplied metadata and media."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import OmegaConf

from .image_text_audio.records import Record, read_records, required_path, validate_media

MODALITIES = {
    "laion_coco_aesthetic": ("text", "image"),
    "flickr30k": ("text", "image"),
    "audiocaps": ("text", "audio"),
    "vggsound": ("image", "audio"),
    "coco_val2014": ("text", "image"),
    "audiocaps_test": ("text", "image", "audio"),
    "vggsound_test": ("image", "audio"),
}
EVAL_DATASETS = {"coco_val2014", "audiocaps_test", "vggsound_test"}


def feature_schema(modalities: tuple[str, ...]) -> pa.Schema:
    return pa.schema(
        [pa.field("stem", pa.string(), nullable=False)] + [pa.field(name, pa.binary()) for name in ("text", "image", "audio")],
        metadata={
            b"modality_shape": json.dumps({name: [768] if name in modalities else None for name in ("text", "image", "audio")}).encode(),
            b"modality_dtype": b"float32",
        },
    )


def write_features(directory: Path, dataset: str, records: list[Record], encoders, batch_size: int) -> None:
    modalities = MODALITIES[dataset]
    schema = feature_schema(modalities)
    with pq.ParquetWriter(directory / "feats.parquet", schema, compression="zstd", use_dictionary=False) as writer:
        for start in range(0, len(records), batch_size):
            chunk = records[start:start + batch_size]
            data = {"stem": [record.stem for record in chunk]}
            for modality in ("text", "image", "audio"):
                if modality not in modalities:
                    data[modality] = [None] * len(chunk)
                    continue
                if modality == "text":
                    values = encoders.encode_text([record.captions[0] for record in chunk])
                elif modality == "image":
                    values = encoders.encode_images([record.image for record in chunk])
                else:
                    values = encoders.encode_audio([record.audio for record in chunk])
                data[modality] = [row.tobytes() for row in values]
            writer.write_table(pa.table(data, schema=schema))


def write_media(directory: Path, records: list[Record], modality: str, batch_size: int) -> None:
    schema = pa.schema([pa.field("stem", pa.string(), nullable=False), pa.field("bytes", pa.binary(), nullable=False), pa.field("ext", pa.string(), nullable=False)])
    filename = "images.parquet" if modality == "image" else "audio.parquet"
    with pq.ParquetWriter(directory / filename, schema, compression="zstd", use_dictionary=False) as writer:
        for start in range(0, len(records), batch_size):
            chunk = records[start:start + batch_size]
            paths = [getattr(record, modality) for record in chunk]
            writer.write_table(pa.table({"stem": [record.stem for record in chunk], "bytes": [path.read_bytes() for path in paths], "ext": [path.suffix.lower() for path in paths]}, schema=schema))


def write_captions(directory: Path, records: list[Record]) -> None:
    """Preserve ordered references independently of first-reference features."""
    captions = {record.stem: list(record.captions) for record in records}
    (directory / "captions.json").write_text(
        json.dumps(captions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def build_dataset(destination: Path, dataset: str, records: list[Record], config: dict, encoders, batch_size: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Lock exclusively so two builders cannot replace one another's output.
    lock = destination.parent / f".{destination.name}.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    temporary = None
    try:
        os.close(descriptor)
        if destination.exists():
            raise FileExistsError(f"{dataset}: output already exists: {destination}; choose a fresh output_root")
        temporary = Path(tempfile.mkdtemp(prefix=f".{dataset}.", dir=destination.parent))
        write_features(temporary, dataset, records, encoders, batch_size)
        if records[0].captions:
            write_captions(temporary, records)
        if dataset in EVAL_DATASETS:
            for modality in MODALITIES[dataset]:
                if modality != "text":
                    write_media(temporary, records, modality, batch_size)
            for filename, source in (config.get("reference_stats") or {}).items():
                shutil.copyfile(Path(source).expanduser(), temporary / filename)
        temporary.rename(destination)
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)
        lock.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data/image_text_audio.yaml"))
    parser.add_argument("--datasets", nargs="+", choices=tuple(MODALITIES))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error("--batch-size and --limit must be positive")
    return args


def main() -> None:
    args = parse_args()
    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    if not isinstance(config, dict) or not isinstance(config.get("datasets"), dict):
        raise ValueError(f"{args.config}: expected mapping with datasets")
    configured = config["datasets"]
    unknown = set(configured) - set(MODALITIES)
    if unknown:
        raise ValueError(f"{args.config}: unknown datasets: {sorted(unknown)}")
    names = args.datasets or list(configured)
    if not names or len(names) != len(set(names)):
        raise ValueError("select at least one dataset, with no duplicate selections")
    output_value = config.get("output_root")
    if not isinstance(output_value, str) or not output_value:
        raise ValueError(f"{args.config}: output_root must be an explicit path")
    output_root = Path(output_value).expanduser()
    prepared = {}
    for dataset in names:
        entry = configured.get(dataset)
        if not isinstance(entry, dict):
            raise ValueError(f"{dataset}: required config entry missing/malformed in {args.config}")
        destination = output_root / ("eval" if dataset in EVAL_DATASETS else "train") / dataset
        if not args.dry_run and destination.exists():
            raise FileExistsError(f"{dataset}: output already exists: {destination}; choose a fresh output_root")
        stats = entry.get("reference_stats") or {}
        if not isinstance(stats, dict) or any(name not in ("fid_stats.npz", "fad_stats_16k.npz") for name in stats):
            raise ValueError(f"{dataset}: reference_stats accepts only fid_stats.npz/fad_stats_16k.npz path mappings")
        if stats and dataset not in EVAL_DATASETS:
            raise ValueError(f"{dataset}: reference_stats is only supported for evaluation groups")
        for name in stats:
            required_path(dataset, stats, name)
        records = read_records(dataset, entry, limit=args.limit)
        if args.dry_run:
            for record in records:
                validate_media(dataset, record)
        prepared[dataset] = records
        print(f"{dataset}: selected {len(records)} supplied records", flush=True)
    if args.dry_run:
        return
    from .image_text_audio.features import FeatureEncoders
    encoders = FeatureEncoders(config.get("assets") or {}, {modality for name in names for modality in MODALITIES[name]}, args.device)
    for dataset, records in prepared.items():
        destination = output_root / ("eval" if dataset in EVAL_DATASETS else "train") / dataset
        build_dataset(destination, dataset, records, configured[dataset], encoders, args.batch_size)
        print(f"{dataset}: wrote {len(records)} records to {destination}", flush=True)


if __name__ == "__main__":
    main()
