"""Batched media decoding and source-side caching for evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from muni.evaluation.image_text_audio.models import (
    CodecBinding,
    _checkpoint_bundle_key,
    bind_codec,
    resolve_image_renderer_settings,
)
from muni.evaluation.data import load_raw_gt
from muni.evaluation.sampling import collect_decoder_outputs, iter_conditional_batches


_DEFAULT_NAMES = {
    "image": "stable-unclip-small",
    "audio": "audioldm-m-full",
    "text": "gemmae",
}


def bind_codecs(checkpoint_paths=None, decoder_names=None, renderer_settings=None):
    names = {
        modality: (decoder_names or {}).get(modality) or default
        for modality, default in _DEFAULT_NAMES.items()
    }
    bundle_key = _checkpoint_bundle_key(checkpoint_paths)
    renderer = resolve_image_renderer_settings(
        renderer_settings, decoder_name=names["image"],
    )
    return {
        modality: CodecBinding.from_config(
            modality, name, checkpoint_paths, bundle_key,
            renderer if modality == "image" else {},
        )
        for modality, name in names.items()
    }


def resolve_decoder_config(model_cfg, eval_cfg):
    """Resolve decoder identifiers and checkpoint paths with model overrides."""
    names, paths, renderer = decoder_config_inputs(model_cfg, eval_cfg)
    return names, paths, resolve_image_renderer_settings(
        renderer, decoder_name=names["image"],
    )


def decoder_config_inputs(model_cfg, eval_cfg):
    """Select raw decoder inputs once at the run configuration boundary."""
    image_text_audio_cfg = model_cfg.get("image_text_audio", {})
    decoder_names = {
        "image": image_text_audio_cfg.get(
            "image_decoder",
            eval_cfg.get("decoders", {}).get("image", "stable-unclip-small"),
        ),
        "audio": image_text_audio_cfg.get(
            "audio_decoder",
            eval_cfg.get("decoders", {}).get("audio", "audioldm-m-full"),
        ),
        "text": image_text_audio_cfg.get(
            "text_decoder", eval_cfg.get("decoders", {}).get("text", "gemmae")
        ),
    }
    checkpoint_paths = image_text_audio_cfg.get(
        "checkpoints", eval_cfg.get("checkpoint_paths")
    )
    renderer_settings = dict(image_text_audio_cfg.get("image_renderer", {}) or {})
    renderer_settings.update(eval_cfg.get("image_renderer", {}))
    return decoder_names, checkpoint_paths, renderer_settings


@torch.no_grad()
def translate_features(
    model,
    feats_dict: dict[str, torch.Tensor],
    src_mods: list[str],
    tgt: str,
    batch_size: int,
    device: torch.device,
    *,
    mode: str,
    sample_kwargs=None,
    show_progress: bool = True,
):
    """Translate a source cohort in order, retaining adapter-native outputs."""
    model.eval()
    n = min(len(v) for v in feats_dict.values())
    src_label = src_mods[0] if mode == "direct" else f"({'+'.join(src_mods)})"
    batches = iter_conditional_batches(
        model, feats_dict, src_mods, tgt, n, batch_size, device,
        mode=mode, sample_kwargs=sample_kwargs,
    )
    if show_progress:
        batches = tqdm(
            batches,
            desc=f"Translating {src_label}->{tgt}",
            total=(n + batch_size - 1) // batch_size,
        )
    return collect_decoder_outputs(
        out.float().cpu() if isinstance(out, torch.Tensor) else out
        for _, out in batches
    )






def _normalise_audio_output(out) -> list[np.ndarray]:
    if isinstance(out, torch.Tensor):
        out = out.cpu().numpy()
    if isinstance(out, np.ndarray):
        if out.ndim == 1:
            out = out[np.newaxis]
        return [out[i].squeeze().astype(np.float32) for i in range(out.shape[0])]
    return [np.asarray(w, dtype=np.float32).squeeze() for w in out]


def _normalise_image_output(out) -> torch.Tensor:
    if isinstance(out, torch.Tensor):
        return out.clamp(0, 1).float().cpu()
    from torchvision.transforms.functional import to_tensor

    return torch.stack([to_tensor(im) for im in out])


@torch.no_grad()
def iter_decode_image_chunks(
    feats: torch.Tensor,
    *,
    device,
    batch_size: int | None = None,
    checkpoint_paths: dict | None = None,
    decoder_name: str | None = None,
    renderer_settings: dict | None = None,
    show_progress: bool = False,
):
    """Decode image features without materialising the full image corpus."""
    codec = bind_codec(
        "image", decoder_name or _DEFAULT_NAMES["image"],
        checkpoint_paths, renderer_settings,
    )
    yield from iter_decode_bound_images(
        feats, codec, device=device, batch_size=batch_size, show_progress=show_progress,
    )


@torch.no_grad()
def iter_decode_bound_images(
    feats, codec, *, device, batch_size=None, show_progress=False,
):
    dec = codec.load(device)
    n = int(feats.shape[0])
    bs = n if batch_size is None or batch_size <= 0 else int(batch_size)
    batches = range(0, n, bs)
    if show_progress and n > bs:
        batches = tqdm(
            batches,
            desc="Decoding image",
            total=(n + bs - 1) // bs,
        )
    for start in batches:
        yield _normalise_image_output(dec(feats[start : start + bs].to(device)))


@torch.no_grad()
def decode_features(
    feats: torch.Tensor,
    modality: str,
    *,
    device,
    batch_size: int | None = None,
    checkpoint_paths: dict | None = None,
    decoder_name: str | None = None,
    renderer_settings: dict | None = None,
    show_progress: bool = False,
) -> torch.Tensor | list[np.ndarray] | list[str]:
    """Decode a tensor of 768-d latent features into raw modality outputs."""
    codec = bind_codec(
        modality, decoder_name or _DEFAULT_NAMES[modality],
        checkpoint_paths, renderer_settings,
    )
    return decode_bound_features(
        feats, codec, device=device, batch_size=batch_size, show_progress=show_progress,
    )


@torch.no_grad()
def decode_bound_features(
    feats, codec, *, device, batch_size=None, show_progress=False,
):
    modality = codec.modality
    n = int(feats.shape[0])
    bs = n if batch_size is None or batch_size <= 0 else int(batch_size)

    if modality == "image":
        parts = list(
            iter_decode_bound_images(
                feats, codec, device=device, batch_size=bs,
                show_progress=show_progress,
            )
        )
        return (
            torch.cat(parts, dim=0)
            if parts
            else torch.empty(
                (0, 3, 0, 0),
                dtype=torch.float32,
            )
        )

    dec = codec.load(device)
    rng = range(0, n, bs)
    if show_progress and n > bs:
        rng = tqdm(
            rng,
            desc=f"Decoding {modality}",
            total=(n + bs - 1) // bs,
        )

    if modality not in ("audio", "text"):
        raise ValueError(f"Unknown modality: {modality!r}")
    outputs = []
    for i in rng:
        out = dec(feats[i : i + bs].to(device))
        outputs.extend(_normalise_audio_output(out) if modality == "audio" else out)
    return outputs


def decode_generated_output(
    output: torch.Tensor | list[str],
    modality: str,
    **decode_kwargs,
) -> torch.Tensor | list[np.ndarray] | list[str]:
    """Decode feature tensors while preserving already-decoded text output."""
    if isinstance(output, list):
        return output
    return decode_features(output, modality, **decode_kwargs)


def decode_bound_output(output, codec, **kwargs):
    """Decode bound feature tensors; adapter-native text lists pass through."""
    if isinstance(output, list):
        return output
    return decode_bound_features(output, codec, **kwargs)




class SourceDecodeCache:
    """Cache decoded source media by dataset, modality, and ordered sample keys."""

    def __init__(
        self,
        *,
        device,
        checkpoint_paths: dict | None = None,
        decoder_names: dict[str, str] | None = None,
        decode_batch_size: int = 8,
        renderer_settings: dict | None = None,
        codecs: dict[str, CodecBinding] | None = None,
    ):
        self.device = device
        self.codecs = codecs or bind_codecs(
            checkpoint_paths, decoder_names, renderer_settings,
        )
        self.decode_batch_size = int(decode_batch_size)
        self._cache: dict[tuple[str, str, tuple[str, ...]], Any] = {}

    def get(
        self,
        cache_key: str,
        modality: str,
        feats: torch.Tensor,
        keys: list[str],
        ds_cfg: dict,
    ):
        """Use raw source media from disk when available, otherwise decode features."""
        cache_id = (str(cache_key), modality, tuple(keys))
        cached = self._cache.get(cache_id)
        if cached is not None:
            return cached

        raw = load_raw_gt(ds_cfg, modality, keys)
        if raw is None:
            raw = decode_bound_features(
                feats, self.codecs[modality],
                device=self.device,
                batch_size=self.decode_batch_size,
                show_progress=True,
            )

        self._cache[cache_id] = raw
        return raw
