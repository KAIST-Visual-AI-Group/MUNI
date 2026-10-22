"""Batched media decoding and source-side caching for evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from muni.evaluation.image_text_audio.models import (
    load_audio_dec,
    load_image_dec,
    load_text_dec,
    resolve_image_renderer_settings,
)
from muni.evaluation.data import load_raw_gt
from muni.evaluation.sampling import collect_decoder_outputs, iter_conditional_batches


_DEFAULT_NAMES = {
    "image": "stable-unclip-small",
    "audio": "audioldm-m-full",
    "text": "gemmae",
}


def resolve_decoder_config(model_cfg, eval_cfg):
    """Resolve decoder identifiers and checkpoint paths with model overrides."""
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
    renderer_settings = resolve_image_renderer_settings(
        renderer_settings, decoder_name=decoder_names["image"]
    )
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


# Single-modality decoding


def get_decoder(modality: str, device, *, name: str | None, checkpoint_paths,
                 renderer_settings=None):
    """Return the cached codec; callers retain their decoder-name/default policy."""
    if modality == "text":
        return load_text_dec(name, device, checkpoint_paths=checkpoint_paths)
    if modality == "image":
        return load_image_dec(name, device, checkpoint_paths=checkpoint_paths,
                              renderer_settings=renderer_settings)
    if modality == "audio":
        return load_audio_dec(name, device, checkpoint_paths=checkpoint_paths)
    raise ValueError(f"Unknown modality: {modality!r}")


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
    dec = get_decoder(
        "image", device, name=decoder_name or _DEFAULT_NAMES["image"],
        checkpoint_paths=checkpoint_paths,
        renderer_settings=renderer_settings,
    )
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
    n = int(feats.shape[0])
    bs = n if batch_size is None or batch_size <= 0 else int(batch_size)

    if modality == "image":
        parts = list(
            iter_decode_image_chunks(
                feats,
                device=device,
                batch_size=bs,
                checkpoint_paths=checkpoint_paths,
                decoder_name=decoder_name,
                renderer_settings=renderer_settings,
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

    dec = get_decoder(
        modality, device, name=decoder_name or _DEFAULT_NAMES[modality],
        checkpoint_paths=checkpoint_paths,
    )
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


# Source data cache


class SourceDecodeCache:
    """Cache decoded source media by dataset, modality, and ordered sample keys."""

    def __init__(
        self,
        *,
        device,
        checkpoint_paths: dict | None,
        decoder_names: dict[str, str] | None = None,
        decode_batch_size: int = 8,
        renderer_settings: dict | None = None,
    ):
        self.device = device
        self.checkpoint_paths = checkpoint_paths
        self.decoder_names = dict(decoder_names) if decoder_names else {}
        self.decode_batch_size = int(decode_batch_size)
        self.renderer_settings = renderer_settings
        self._cache: dict[tuple[str, str, tuple[str, ...]], Any] = {}

    def _decoder_name(self, modality: str) -> str | None:
        return self.decoder_names.get(modality)

    def get(
        self,
        cache_key: str,
        modality: str,
        feats: torch.Tensor,
        keys: list[str],
        ds_cfg: dict,
    ):
        """Return raw GT for *modality* — from disk if available, else decoded."""
        cache_id = (str(cache_key), modality, tuple(keys))
        cached = self._cache.get(cache_id)
        if cached is not None:
            return cached

        raw = load_raw_gt(ds_cfg, modality, keys)
        if raw is None:
            raw = decode_features(
                feats,
                modality,
                device=self.device,
                batch_size=self.decode_batch_size,
                checkpoint_paths=self.checkpoint_paths,
                decoder_name=self._decoder_name(modality),
                renderer_settings=self.renderer_settings,
                show_progress=True,
            )

        self._cache[cache_id] = raw
        return raw
