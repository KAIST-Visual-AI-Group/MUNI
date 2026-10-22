"""Load aligned evaluation features and raw media from packed Parquet."""

import io
import os

from itertools import islice
import numpy as np
import torch
import muni.util.misc as misc


def iter_eval_batch_matches(data_loader, device, requests, max_attempts=50):
    """Yield first-match batches; full consumers retain the loader's extra fetch."""
    pending = set(requests)
    if not data_loader or not pending:
        return
    for index, batch in enumerate(data_loader):
        if index >= max_attempts or not pending:
            break
        data = {
            key: misc.move_to_device(value, device)
            for key, value in misc.extract_dict_data(batch).items()
        }
        for request in [item for item in pending if set(item).issubset(data)]:
            pending.remove(request)
            yield request, data


def load_test_features(
    path: str,
    modality: str,
    max_samples: int | None = None,
) -> tuple[torch.Tensor | None, list[str]]:
    """Return ``(features, keys)`` from a packed feature Parquet."""
    if max_samples is not None and (type(max_samples) is not int or max_samples <= 0):
        raise ValueError("max_samples must be a positive integer or None")
    from muni.data.sources.features import (
        _iter_feats_parquet,
        _read_feats_parquet_shape,
    )

    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    shape = _read_feats_parquet_shape(path, modality, parquet.metadata.metadata, allow_absent=True)
    if shape is None:
        return None, []
    count = parquet.metadata.num_rows
    retained = count if max_samples is None else min(count, max_samples)
    features = torch.empty((retained, *shape), dtype=torch.float32)
    keys = []
    # Admit only rows consumed by this prefix; larger requests admit their own tail.
    for stem, blob in islice(_iter_feats_parquet(path, modality, shape, parquet), retained):
        features[len(keys)].numpy()[...] = np.frombuffer(blob, dtype=np.float32).reshape(shape)
        keys.append(stem)
    return (features if keys else None), keys


def load_captions_json(path: str | None) -> dict[str, list[str]] | None:
    """Load captions from a JSON file.

    Supports two formats:
      - MS-COCO:  ``{"images": [...], "annotations": [...]}``
      - Simple:   ``{"sample_key": ["cap1", "cap2", ...]}``  (or str value)
    """
    if path is None:
        return None

    from muni.data.sources.features import _load_caption_mapping

    data = _load_caption_mapping(path)

    if "annotations" in data:
        from collections import defaultdict

        id_to_key: dict[int, str] = {}
        image_keys: set[str] = set()
        for img in data.get("images", []):
            if img["id"] in id_to_key:
                raise ValueError(f"{path}: duplicate image ID {img['id']!r}")
            key = os.path.splitext(img["file_name"])[0]
            if key in image_keys:
                raise ValueError(f"{path}: duplicate image stem {key!r}")
            id_to_key[img["id"]] = key
            image_keys.add(key)
        caps: dict[str, list[str]] = defaultdict(list)
        for ann in data["annotations"]:
            key = id_to_key[ann["image_id"]]
            if not isinstance(ann["caption"], str):
                raise ValueError(f"{path}: invalid caption for {key!r}")
            caps[key].append(ann["caption"])
        return dict(caps)

    result: dict[str, list[str]] = {}
    for k, v in data.items():
        values = [v] if isinstance(v, str) else v
        if not isinstance(values, list) or not values or any(
            not isinstance(caption, str) for caption in values
        ):
            raise ValueError(f"{path}: captions for {k!r} must be a nonempty string list")
        result[k] = values
    return result


def _load_blob_parquet_index(path: str, keys: list[str] | None = None) -> dict[str, bytes]:
    """Validate all media rows, retaining only requested keys when supplied."""
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    wanted = set(keys) if keys is not None else None
    seen = set()
    index = {}
    for batch in parquet.iter_batches(batch_size=256, columns=["stem", "bytes"]):
        for stem, blob in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist(), strict=True):
            if not isinstance(stem, str):
                raise ValueError(f"{path}: invalid media stem {stem!r}")
            if stem in seen:
                raise ValueError(f"{path}: duplicate media stem {stem!r}")
            seen.add(stem)
            if not isinstance(blob, bytes) or not blob:
                raise ValueError(f"{path}: null or invalid media bytes for stem {stem!r}")
            if wanted is None or stem in wanted:
                index[stem] = blob
    return index


def load_raw_images(
    images_dir: str | None,
    keys: list[str],
) -> torch.Tensor | None:
    """Load raw images as ``(N, 3, H, W)`` float tensor in ``[0, 1]``.

    An unset source returns ``None``; malformed or unaligned sources raise.
    """
    if not images_dir:
        return None
    from PIL import Image
    from torchvision.transforms.functional import to_tensor, resize

    index = _load_blob_parquet_index(images_dir, keys)
    imgs = []
    for k in keys:
        with Image.open(io.BytesIO(index[k])) as source:
            img = source.convert("RGB")
        t = to_tensor(img)
        if t.shape[1] != 224 or t.shape[2] != 224:
            t = resize(t, [224, 224], antialias=True)
        imgs.append(t)
    return torch.stack(imgs)


def _decode_wav_bytes(blob: bytes, sr: int) -> np.ndarray:
    """Decode WAV bytes to a mono float32 np.ndarray resampled to *sr*."""
    import soundfile as sf
    wav, file_sr = sf.read(io.BytesIO(blob), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if file_sr != sr:
        import torchaudio.functional as audio_functional

        wav = audio_functional.resample(torch.from_numpy(wav), file_sr, sr).numpy()
    return wav


def load_raw_audio(
    audio_dir: str | None,
    keys: list[str],
    sr: int = 16000,
) -> list[np.ndarray] | None:
    """Load raw audio waveforms (resampled to *sr*).

    An unset source returns ``None``; malformed or unaligned sources raise.
    """
    if not audio_dir:
        return None

    index = _load_blob_parquet_index(audio_dir, keys)
    wavs = []
    for k in keys:
        wavs.append(_decode_wav_bytes(index[k], sr))
    return wavs


def align_keys(*key_lists: list[str]) -> tuple[list[str], list[list[int]]]:
    """Intersect key lists and return ``(common_keys, [index_map_per_list])``."""
    common = sorted(set.intersection(*(set(kl) for kl in key_lists)))
    idx_maps = []
    for kl in key_lists:
        lookup = {k: i for i, k in enumerate(kl)}
        idx_maps.append([lookup[k] for k in common])
    return common, idx_maps


class EvaluationFeatureCache:
    """Cache validated CPU feature prefixes and cohorts within one invocation."""

    def __init__(self):
        self._feature_cache: dict[
            tuple[str, str, int | None], tuple[torch.Tensor | None, list[str]]
        ] = {}
        self._aligned_cache: dict[
            tuple[str, tuple[str, ...], int | None],
            tuple[list[str], dict[str, torch.Tensor]],
        ] = {}

    def load_features(self, feat_dir, modality, max_samples):
        path = str(feat_dir)
        for (cached_path, cached_modality, cap), (features, keys) in self._feature_cache.items():
            if (cached_path, cached_modality) == (path, modality) and (
                cap is None or (max_samples is not None and cap >= max_samples)
            ):
                return (features[:max_samples] if features is not None else None), keys[:max_samples]
        cache_key = (path, modality, max_samples)
        self._feature_cache[cache_key] = load_test_features(feat_dir, modality, max_samples)
        return self._feature_cache[cache_key]

    def aligned_features(self, feat_dir, modalities, max_samples):
        """Return fresh aligned tensors, keeping cached source features read-only."""
        all_feats, all_keys = {}, {}
        for modality in modalities:
            features, keys = self.load_features(feat_dir, modality, max_samples)
            if features is None:
                return [], {}
            all_feats[modality], all_keys[modality] = features, keys
        common, index_maps = align_keys(*(all_keys[modality] for modality in modalities))
        return common, {
            modality: all_feats[modality][index_maps[index]]
            for index, modality in enumerate(modalities)
        }

    def get_aligned_subset(self, ds_key, ds_cfg, modalities, max_samples):
        cache_key = (ds_key, tuple(modalities), max_samples)
        if cache_key not in self._aligned_cache:
            self._aligned_cache[cache_key] = self.aligned_features(
                ds_cfg["feat_dir"], modalities, max_samples
            )
        return self._aligned_cache[cache_key]

    def cohort_counts(self):
        return [
            {"dataset": dataset, "modalities": list(modalities),
             "effective_cap": cap, "aligned_samples": len(common)}
            for (dataset, modalities, cap), (common, _) in self._aligned_cache.items()
        ]


def load_raw_gt(eval_ds_cfg: dict, modality: str, keys: list[str]):
    """Load raw GT data for a modality using eval dataset config.

    Returns:
      - text:  ``list[str]`` or ``None``
      - image: ``Tensor (N, 3, H, W)`` or ``None``
      - audio: ``list[np.ndarray]`` or ``None``
    """
    if modality == "text":
        captions = load_captions_json(eval_ds_cfg.get("captions_json"))
        if captions is None:
            return None
        return [captions[key][0] for key in keys]
    if modality == "image":
        return load_raw_images(eval_ds_cfg.get("images_dir"), keys)
    if modality == "audio":
        return load_raw_audio(eval_ds_cfg.get("audio_dir"), keys)
    return None


def resolve_eval_dataset(eval_datasets_cfg, *modalities: str):
    """Return the configured evaluation dataset containing every modality."""
    if not eval_datasets_cfg:
        return None
    required = set(modalities)
    for name, dataset_cfg in eval_datasets_cfg.items():
        if required.issubset(set(str(name).split("-"))):
            return dataset_cfg
    return None


def load_aligned_eval_batch(
    eval_dataset_cfg,
    modalities: list[str],
    num_samples: int,
    device: torch.device,
    *,
    cache: EvaluationFeatureCache | None = None,
) -> tuple[dict[str, torch.Tensor] | None, list[str] | None]:
    """Load key-aligned evaluation features for the requested modalities."""
    feature_cache = cache if cache is not None else EvaluationFeatureCache()
    keys, features = feature_cache.aligned_features(
        eval_dataset_cfg["feat_dir"], modalities, num_samples
    )
    if not keys:
        return None, None
    return {modality: features[modality].to(device) for modality in modalities}, keys


def required_eval_samples(eval_node) -> int:
    """Return the largest metric or visualization sample request."""
    metric_counts = [
        int(metric_cfg.get("params", {}).get("num_samples", 32))
        for metric_cfg in eval_node.get("metrics", [])
    ]
    logger_counts = [
        int(logger_cfg.get("params", {}).get("num_vis_samples", 16))
        for logger_cfg in eval_node.get("loggers", [])
    ]
    return max(metric_counts + logger_counts + [1])
