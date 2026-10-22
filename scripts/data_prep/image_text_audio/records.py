"""Read supplied native training tables and aligned prepared evaluation media."""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Record:
    stem: str
    captions: tuple[str, ...] = ()
    image: Path | None = None
    audio: Path | None = None


def fail(dataset: str, key: str, path: Path, message: str) -> ValueError:
    return ValueError(f"{dataset}: key={key!r}, path={path}: {message}")


def required_path(dataset: str, config: dict, field: str, *, directory: bool = False) -> Path:
    value = config.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{dataset}: required path {field!r} is missing")
    path = Path(value).expanduser()
    if not (path.is_dir() if directory else path.is_file()):
        raise fail(dataset, field, path, "required directory/file is absent")
    return path


def text(value: Any, dataset: str, key: str, path: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise fail(dataset, key, path, "expected nonempty string")
    return value.strip()


def references(value: Any, dataset: str, key: str, path: Path) -> tuple[str, ...]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not value:
        raise fail(dataset, key, path, "expected caption string or nonempty list")
    return tuple(text(item, dataset, key, path) for item in value)


def table(dataset: str, path: Path) -> list[dict]:
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        if fields is None or len(set(fields)) != len(fields):
            raise fail(dataset, "header", path, "missing or duplicate CSV columns")
        rows = list(reader)
    for index, row in enumerate(rows, 2):
        if None in row or any(value is None for value in row.values()):
            raise fail(dataset, str(index), path, "CSV row does not match header width")
    return rows


def keyed_media(dataset: str, root: Path, suffixes: set[str]) -> dict[str, Path]:
    result = {}
    for path in sorted(root.iterdir()):
        if path.is_file() and path.suffix.lower() in suffixes:
            if path.stem in result:
                raise fail(dataset, path.stem, path, "duplicate media key")
            result[path.stem] = path
    return result


def media_path(dataset: str, root: Path, key: str, suffixes: set[str]) -> Path:
    matches = [root / f"{key}{suffix}" for suffix in sorted(suffixes)]
    matches = [path for path in matches if path.is_file()]
    if len(matches) != 1:
        raise fail(dataset, key, root, "expected exactly one matching supplied media file")
    return matches[0]


def clip_start(value: Any, dataset: str, key: str, path: Path) -> str:
    try:
        number = float(value)
        if not math.isfinite(number) or number < 0 or not number.is_integer():
            raise ValueError()
    except (TypeError, ValueError):
        raise fail(dataset, key, path, "start_time must be a finite nonnegative integer") from None
    return str(int(number))


def read_records(dataset: str, config: dict, limit: int | None = None) -> list[Record]:
    if dataset == "vggsound_test" and config.get("captions") is None:
        image_root = required_path(dataset, config, "images_dir", directory=True)
        audio_root = required_path(dataset, config, "audio_dir", directory=True)
        images = keyed_media(dataset, image_root, {".png", ".jpg", ".jpeg"})
        audio = keyed_media(dataset, audio_root, {".wav"})
        if set(images) != set(audio):
            missing = sorted(set(images) - set(audio))
            extra = sorted(set(audio) - set(images))
            raise fail(dataset, (missing or extra)[0], audio_root, f"misaligned audio keys; missing={missing[:5]}, extra={extra[:5]}")
        records = [Record(key, image=images[key], audio=audio[key]) for key in sorted(images)]
        return checked_records(dataset, records, image_root, limit)
    if dataset in ("coco_val2014", "audiocaps_test", "vggsound_test"):
        path = required_path(dataset, config, "captions")
        from muni.data.sources.features import _load_caption_mapping

        captions = _load_caption_mapping(path)
        modalities = ("image",) if dataset == "coco_val2014" else ("image", "audio")
        media = {}
        for modality in modalities:
            root = required_path(dataset, config, f"{modality}s_dir" if modality == "image" else "audio_dir", directory=True)
            media[modality] = keyed_media(dataset, root, {".png", ".jpg", ".jpeg"} if modality == "image" else {".wav"})
            if set(media[modality]) != set(captions):
                missing = sorted(set(captions) - set(media[modality]))
                extra = sorted(set(media[modality]) - set(captions))
                raise fail(dataset, (missing or extra or [""])[0], root, f"misaligned {modality} keys; missing={missing[:5]}, extra={extra[:5]}")
        records = []
        for key, value in sorted(captions.items()):
            if not key or Path(key).name != key:
                raise fail(dataset, key, path, "key must be a nonempty opaque filename stem")
            refs = references(value, dataset, key, path)
            records.append(Record(key, refs, media["image"][key], media.get("audio", {}).get(key)))
    else:
        path = required_path(dataset, config, "metadata")
        rows = [] if dataset == "vggsound" else table(dataset, path)
        records = []
        if dataset in ("laion_coco_aesthetic", "flickr30k"):
            root = required_path(dataset, config, "images_dir", directory=True)
            for index, row in enumerate(rows, 2):
                if dataset == "flickr30k":
                    filename = text(row.get("filename"), dataset, str(index), path)
                    if Path(filename).name != filename:
                        raise fail(dataset, filename, path, "filename must not contain directories")
                    try:
                        raw = json.loads(row.get("raw", ""))
                    except (TypeError, json.JSONDecodeError):
                        raise fail(dataset, filename, path, "raw must be a JSON caption list") from None
                    if not isinstance(raw, list):
                        raise fail(dataset, filename, path, "raw must be a JSON caption list")
                    records.append(Record(Path(filename).stem, references(raw, dataset, filename, path), root / filename))
                else:
                    key_field, caption_field = ("sha256", "caption") if path.suffix.lower() == ".parquet" else ("image_id", "text")
                    key = text(row.get(key_field), dataset, str(index), path)
                    records.append(Record(key, (text(row.get(caption_field), dataset, key, path),), media_path(dataset, root, key, {".png", ".jpg", ".jpeg"})))
        elif dataset == "audiocaps":
            root = required_path(dataset, config, "audio_dir", directory=True)
            groups: dict[str, list[tuple[str, str]]] = {}
            ids = set()
            for index, row in enumerate(rows, 2):
                caption_id = text(row.get("audiocap_id"), dataset, str(index), path)
                if caption_id in ids:
                    raise fail(dataset, caption_id, path, "duplicate audiocap_id")
                ids.add(caption_id)
                youtube = text(row.get("youtube_id"), dataset, caption_id, path)
                start = clip_start(row.get("start_time"), dataset, caption_id, path)
                key = f"{youtube}_{start}"
                groups.setdefault(key, []).append((caption_id, text(row.get("caption"), dataset, key, path)))
            for key, group in sorted(groups.items()):
                refs = tuple(caption for _, caption in group)
                records.append(Record(key, refs, audio=root / f"{key}.wav"))
        elif dataset == "vggsound":
            image_root = required_path(dataset, config, "images_dir", directory=True)
            audio_root = required_path(dataset, config, "audio_dir", directory=True)
            # Native VGG metadata has four headerless columns; a declared
            # header with the same field names is also accepted.
            with path.open(newline="", encoding="utf-8-sig") as handle:
                raw_rows = list(csv.reader(handle))
            fields = ("youtube_id", "start_time", "label", "split")
            if raw_rows and set(raw_rows[0]) == set(fields):
                fields = tuple(raw_rows.pop(0))
            for index, row in enumerate(raw_rows, 1):
                if len(row) != 4:
                    raise fail(dataset, str(index), path, "expected youtube_id,start_time,label,split")
                rows.append(dict(zip(fields, row)))
            split = config.get("split", "train")
            if split != "train":
                raise fail(dataset, "split", path, "training VGG group requires split=train")
            for index, row in enumerate(rows, 1):
                row_split = text(row.get("split"), dataset, str(index), path)
                if row_split not in ("train", "test"):
                    raise fail(dataset, str(index), path, "split must be train or test")
                youtube = text(row.get("youtube_id"), dataset, str(index), path)
                start = clip_start(row.get("start_time"), dataset, youtube, path)
                if row_split != split:
                    continue
                key = f"{youtube}_{int(start):06d}"
                records.append(Record(key, image=image_root / f"{key}.png", audio=audio_root / f"{key}.wav"))
        else:
            raise ValueError(f"Unknown dataset {dataset!r}")
    return checked_records(dataset, records, path, limit)


def checked_records(dataset: str, records: list[Record], path: Path, limit: int | None) -> list[Record]:
    if not records:
        raise ValueError(f"{dataset}: no supplied records")
    seen = set()
    for record in records:
        if not record.stem or Path(record.stem).name != record.stem or record.stem in seen:
            raise fail(dataset, record.stem, path, "duplicate or invalid key")
        seen.add(record.stem)
    return records if limit is None else records[:limit]


def read_audio(path: Path):
    import numpy as np
    import soundfile as sf

    try:
        if path.suffix.lower() != ".wav":
            raise ValueError("only WAV audio is accepted")
        samples, rate = sf.read(path, dtype="float32", always_2d=True)
        if not samples.size or rate <= 0 or not np.isfinite(samples).all():
            raise ValueError("empty/nonfinite audio or invalid sample rate")
    except Exception as error:
        raise ValueError(f"{path}: invalid audio: {error}") from error
    return samples, rate


def validate_media(dataset: str, record: Record) -> None:
    from PIL import Image
    for modality in ("image", "audio"):
        path = getattr(record, modality)
        if path is None:
            continue
        try:
            if modality == "image":
                if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                    raise ValueError("only PNG/JPG images are accepted")
                with Image.open(path) as image:
                    image.load()
                    if min(image.size) < 1:
                        raise ValueError("empty image")
            else:
                read_audio(path)
        except Exception as error:
            raise fail(dataset, record.stem, path, f"invalid {modality}: {error}") from error
