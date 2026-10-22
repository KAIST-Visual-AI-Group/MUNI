"""Parquet latent-feature data sources."""

import torch

from muni.data.core import BaseSubdataset, register_subdataset


def _load_caption_mapping(path):
    """Admit JSON objects without silently overwriting duplicate caption IDs."""
    import json

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{path}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    with open(path, encoding="utf-8") as source:
        captions = json.load(source, object_pairs_hook=unique_object)
    if not isinstance(captions, dict):
        raise ValueError(f"{path}: captions must be a JSON object")
    return captions


def _read_feats_parquet_shape(
    path: str, array_key: str, metadata, *, allow_absent: bool = False
) -> tuple[int, ...] | None:
    """Validate requested shape metadata; explicit null denotes absence."""
    import json

    meta = metadata or {}
    shape = json.loads(meta[b"modality_shape"])[array_key]
    if shape is None:
        if allow_absent:
            return None
        raise ValueError(f"{path}: requested modality '{array_key}' is absent (null shape)")
    if not isinstance(shape, list) or any(
        type(dim) is not int or dim <= 0 for dim in shape
    ):
        raise ValueError(f"{path}: invalid shape for '{array_key}': {shape!r}")
    if b"modality_dtype" in meta:
        dtype_blob = meta[b"modality_dtype"]
        if dtype_blob == b"float32":
            dtype = "float32"
        else:
            dtype = json.loads(dtype_blob)
            if isinstance(dtype, dict):
                dtype = dtype.get(array_key)
        if dtype != "float32":
            raise ValueError(f"{path}: requested modality '{array_key}' must have float32 dtype")
    return tuple(shape)


def _iter_feats_parquet(path: str, array_key: str, shape: tuple[int, ...], parquet):
    """Yield validated requested float32 rows in file order, without copies."""
    import math
    seen: set[str] = set()
    expected_bytes = math.prod(shape) * 4
    for batch in parquet.iter_batches(batch_size=256, columns=["stem", array_key]):
        for stem, blob in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist(), strict=True):
            if not isinstance(stem, str):
                raise ValueError(f"{path}: invalid stem {stem!r}; expected a string")
            if stem in seen:
                raise ValueError(f"{path}: duplicate stem {stem!r}")
            seen.add(stem)
            if not isinstance(blob, bytes) or len(blob) != expected_bytes:
                raise ValueError(
                    f"{path}: invalid float32 payload for '{array_key}', stem {stem!r}; "
                    f"expected {expected_bytes} bytes for shape {shape}"
                )
            yield stem, blob


def _decode_feature_row(blob: bytes, shape: tuple[int, ...]) -> torch.Tensor:
    import numpy as np

    return torch.from_numpy(np.frombuffer(blob, dtype=np.float32).reshape(shape).copy())


def _load_feature_storage(paths, parquets, array_key, shape, cache_in_memory):
    """Keep encoded rows OR fill one tensor in file order, never both full payloads."""
    import numpy as np

    features = torch.empty((sum(p.metadata.num_rows for p in parquets), *shape), dtype=torch.float32) if cache_in_memory else None
    rows = {}
    multi = len(paths) > 1
    for file_index, (path, parquet) in enumerate(zip(paths, parquets, strict=True)):
        for stem, blob in _iter_feats_parquet(path, array_key, shape, parquet):
            key = f"{file_index}_{stem}" if multi else stem
            if features is not None:
                index = len(rows)
                rows[key] = index
                features[index].numpy()[...] = np.frombuffer(blob, dtype=np.float32).reshape(shape)
            else:
                rows[key] = blob
    return sorted(rows), rows, features


@register_subdataset
class LatentFeatureParquetSubdataset(BaseSubdataset):
    """Load keyed latent features from Parquet."""

    registry_name = "latent_feature_parquet"

    def __init__(
        self,
        parquet_paths: str | list[str],
        array_key: str,
        cache_in_memory: bool = True,
    ):
        if isinstance(parquet_paths, str):
            parquet_paths = [parquet_paths]

        self.array_key = array_key
        self._paths = [str(p) for p in parquet_paths]

        import pyarrow.parquet as pq

        parquets = [pq.ParquetFile(path) for path in self._paths]
        self._shape = _read_feats_parquet_shape(self._paths[0], array_key, parquets[0].metadata.metadata)
        for path, parquet in zip(self._paths[1:], parquets[1:], strict=True):
            shape = _read_feats_parquet_shape(path, array_key, parquet.metadata.metadata)
            if shape != self._shape:
                raise ValueError(f"{path}: shape {shape} for '{array_key}' differs from {self._shape}")
        self._keys, self._rows, self._features = (
            _load_feature_storage(self._paths, parquets, array_key, self._shape, cache_in_memory)
        )

    def keys(self) -> list[str]:
        return self._keys

    def get(self, key: str, augment_state: dict | None = None) -> torch.Tensor:
        if self._features is not None:
            return self._features[self._rows[key]]
        return _decode_feature_row(self._rows[key], self._shape)


@register_subdataset
class TextFeatureOnlyParquetSubdataset(LatentFeatureParquetSubdataset):
    """Load text features from Parquet without CE tokens."""

    registry_name = "text_feature_only_parquet"

    def __init__(
        self,
        parquet_paths,
        array_key: str = "text",
        cache_in_memory: bool = True,
    ):
        super().__init__(parquet_paths, array_key, cache_in_memory)

    def get(self, key: str, augment_state: dict | None = None):
        return {"feature": super().get(key, augment_state)}


@register_subdataset
class TextFeatureAndTokenParquetSubdataset(LatentFeatureParquetSubdataset):
    """Parquet features with sidecar captions and fixed-length tokens."""

    registry_name = "text_feature_and_tokens_parquet"

    def __init__(
        self,
        parquet_paths,
        captions_sources,
        tokenizer_path: str,
        array_key: str = "text",
        max_seq_len: int = 64,
        pad_token_id: int | None = None,
        eos_token_id: int | None = None,
        cache_in_memory: bool = True,
    ):
        if isinstance(parquet_paths, str):
            parquet_paths = [parquet_paths]

        from transformers import AutoTokenizer

        if isinstance(captions_sources, str):
            captions_sources = [captions_sources]

        self.max_seq_len = int(max_seq_len)
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        # Gemma defaults to left padding, while the downstream next-token CE
        # contract requires ``[BOS, ..., EOS, pad, ...]``.
        self._tokenizer.padding_side = "right"
        self._pad_token_id = (
            int(pad_token_id)
            if pad_token_id is not None
            else (
                self._tokenizer.pad_token_id
                if self._tokenizer.pad_token_id is not None
                else (self._tokenizer.eos_token_id or 0)
            )
        )
        self._eos_token_id = (
            int(eos_token_id)
            if eos_token_id is not None
            else (self._tokenizer.eos_token_id or self._pad_token_id)
        )

        merged_captions: dict[str, str] = {}
        for src in captions_sources:
            captions = _load_caption_mapping(src)
            for key, refs in captions.items():
                if not key:
                    raise ValueError(f"{src}: caption IDs must be non-empty.")
                if key in merged_captions:
                    raise ValueError(f"Duplicate caption ID {key!r} across caption sources.")
                if isinstance(refs, str):
                    refs = [refs]
                if (
                    not isinstance(refs, list) or not refs
                    or not all(isinstance(ref, str) and ref.strip() for ref in refs)
                ):
                    raise ValueError(f"{src}: caption ID {key!r} requires non-empty string references.")
                merged_captions[key] = refs[0]
        self._captions = merged_captions

        super().__init__(parquet_paths, array_key, cache_in_memory)
        self._tokens = (
            torch.empty((len(self._keys), self.max_seq_len), dtype=torch.long)
            if cache_in_memory else None
        )
        if self._tokens is not None:
            multi = len(self._paths) > 1
            for key in self._keys:
                stem = key.split("_", 1)[1] if multi else key
                self._tokens[self._rows[key]].copy_(self._tokenize(self._captions[stem]))
            self._captions = {}

    def get(self, key: str, augment_state: dict | None = None):
        feature = super().get(key, augment_state)
        if self._tokens is not None:
            tokens = self._tokens[self._rows[key]]
        else:
            stem = key.split("_", 1)[1] if len(self._paths) > 1 else key
            tokens = self._tokenize(self._captions[stem])
        return {"feature": feature, "tokens": tokens}

    def _tokenize(self, caption: str) -> torch.Tensor:
        enc = self._tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
            add_special_tokens=True,
        )
        ids = enc.input_ids[0].long()
        mask = enc.attention_mask[0].bool()
        ids = ids.masked_fill(~mask, self._pad_token_id)
        real_len = int(mask.sum().item())
        if real_len >= self.max_seq_len:
            if int(ids[-1].item()) != self._eos_token_id:
                ids[-1] = self._eos_token_id
        elif 0 < real_len and int(ids[real_len - 1].item()) != self._eos_token_id:
            ids[real_len] = self._eos_token_id
        return ids


