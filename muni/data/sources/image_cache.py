"""Persistent, mmap-friendly uint8 caches for aligned image modalities."""

from __future__ import annotations

import os
import hashlib
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


IMAGE_CACHE_KIND = "aligned_image_uint8"
IMAGE_CACHE_VERSION = 1
_LOADED_CACHES: dict[tuple[Path, bool], tuple[tuple[int, ...], dict[str, Any]]] = {}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_temporary_file(path: Path) -> Path:
    """Stage beside the destination; the caller owns cleanup and publication."""
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    return Path(name)


def publish_file(temporary: Path, path: Path, *, overwrite: bool) -> None:
    if overwrite:
        os.replace(temporary, path)
    else:
        # A same-filesystem hard link publishes atomically without replacing a winner.
        os.link(temporary, path)


def _validate_cache_payload(
    payload: Any, *, path: Path | None = None
) -> dict[str, Any]:
    where = f" at {path}" if path is not None else ""
    kind = payload.get("kind")
    if not isinstance(kind, str) or kind.rsplit(".", 1)[-1] != IMAGE_CACHE_KIND:
        raise ValueError(
            f"Unsupported image cache kind{where}: {kind!r}; "
            f"expected a producer namespace ending in {IMAGE_CACHE_KIND!r}."
        )
    if payload.get("version") != IMAGE_CACHE_VERSION:
        raise ValueError(
            f"Unsupported image cache version{where}: {payload.get('version')!r}; "
            f"expected {IMAGE_CACHE_VERSION}."
        )

    modalities = payload.get("modalities")
    keys = payload.get("keys")
    images = payload.get("images")
    img_size = payload.get("img_size")
    if (
        not isinstance(modalities, list)
        or not modalities
        or not all(isinstance(x, str) for x in modalities)
    ):
        raise TypeError(f"Image cache modalities{where} must be a non-empty list[str].")
    if len(set(modalities)) != len(modalities):
        raise ValueError(
            f"Image cache modalities{where} contain duplicates: {modalities}."
        )
    if (
        not isinstance(keys, list)
        or not keys
        or not all(isinstance(x, str) for x in keys)
    ):
        raise TypeError(f"Image cache keys{where} must be a non-empty list[str].")
    if keys != sorted(keys) or len(set(keys)) != len(keys):
        raise ValueError(f"Image cache keys{where} must be sorted and unique.")
    expected_shape = (len(modalities), len(keys), 3, img_size, img_size)
    if images.dtype != torch.uint8 or tuple(images.shape) != expected_shape:
        raise ValueError(
            f"Image cache tensor{where} must be uint8 with shape {expected_shape}, "
            f"got dtype={images.dtype}, shape={tuple(images.shape)}."
        )
    return payload


def load_image_tensor_cache(
    path: str | Path, *, mmap: bool = False, labels_csv: str | Path | None = None
) -> dict[str, Any]:
    """Load and validate one cache without decoding its source images."""
    cache_path = Path(path).expanduser().resolve()
    cache_key = (cache_path, bool(mmap))
    stat = cache_path.stat()
    identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    cached = _LOADED_CACHES.get(cache_key)
    if cached is None or cached[0] != identity:
        # Retire both loading modes for the old file; live readers keep their snapshot.
        for key in list(_LOADED_CACHES):
            if key[0] == cache_path and _LOADED_CACHES[key][0] != identity:
                del _LOADED_CACHES[key]
        payload = torch.load(
            cache_path, map_location="cpu", weights_only=True, mmap=bool(mmap)
        )
        payload = _validate_cache_payload(payload, path=cache_path)
        _LOADED_CACHES[cache_key] = (identity, payload)
    else:
        payload = cached[1]
    if labels_csv is not None:
        labels_path = Path(labels_csv).expanduser().resolve()
        metadata = payload.get("metadata", {})
        expected = metadata.get("labels_sha256") if isinstance(metadata, dict) else None
        if not expected:
            raise ValueError(
                f"Image cache {cache_path} lacks metadata.labels_sha256; rebuild it "
                f"with the matching producer and use its output labels CSV: {labels_path}."
            )
        if sha256_file(labels_path) != expected:
            raise ValueError(
                f"Image cache {cache_path} labels_sha256 does not match {labels_path}; "
                "use the labels CSV produced with this cache, or rebuild both together."
            )
    return payload


def save_image_tensor_cache(
    path: str | Path,
    *,
    modalities: Sequence[str],
    keys: Sequence[str],
    images: torch.Tensor,
    img_size: int,
    metadata: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    """Validate and atomically save a cache in the runtime loader's format."""
    cache_path = Path(path).expanduser().resolve()
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "kind": IMAGE_CACHE_KIND,
        "version": IMAGE_CACHE_VERSION,
        "modalities": list(modalities),
        "keys": list(keys),
        "img_size": int(img_size),
        "images": images,
        "metadata": dict(metadata or {}),
    }
    _validate_cache_payload(payload)

    temporary_path = create_temporary_file(cache_path)
    try:
        torch.save(payload, temporary_path)
        publish_file(temporary_path, cache_path, overwrite=overwrite)
    finally:
        temporary_path.unlink(missing_ok=True)
    return cache_path
