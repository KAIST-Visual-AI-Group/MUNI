"""Lazily cached CLIP, CLAP, and Wav2CLIP encoders for image-text-audio metrics.

Returned embeddings are fp32 and L2-normalized; model forwards use fp16 for
CLIP and bfloat16 for CLAP.
"""

from __future__ import annotations


import numpy as np
import torch
import torch.nn.functional as F

from muni.util.misc import preserve_rng_state


# Cache

_CACHE: dict[str, object] = {}

_CLIP_DTYPE = torch.float16  # native CLIP weights dtype
_CLAP_DTYPE = torch.bfloat16  # CLAP loads fp32; bf16 is the safer mixed dtype


def _device_key(device) -> str:
    d = device if isinstance(device, torch.device) else torch.device(device)
    if d.type == "cuda" and d.index is None and torch.cuda.is_available():
        d = torch.device("cuda", torch.cuda.current_device())
    return str(d)


# CLIP (ViT-B/32, OpenAI)


def get_clip(device) -> tuple[torch.nn.Module, object]:
    """Return ``(model, tokenizer)`` for ViT-B-32 OpenAI in fp16, eval mode."""
    key = f"clip:{_device_key(device)}"
    if key not in _CACHE:
        with preserve_rng_state(torch.device(device)):
            import open_clip

            model, _, _ = open_clip.create_model_and_transforms(
                "ViT-B-32",
                pretrained="openai",
                force_quick_gelu=True,
            )
            tokenizer = open_clip.get_tokenizer("ViT-B-32")
            model = model.to(device=device, dtype=_CLIP_DTYPE).eval()
            _CACHE[key] = (model, tokenizer)
    return _CACHE[key]


@torch.no_grad()
def clip_encode_images(
    images: torch.Tensor,
    device,
    *,
    batch_size: int = 64,
) -> torch.Tensor:
    """``(N, C, H, W)`` float in ``[0, 1]`` → ``(N, D)`` L2-normalised fp32 features."""
    from torchvision.transforms.functional import resize, center_crop, normalize

    model, _ = get_clip(device)
    feats = []
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size].to(device=device, dtype=torch.float32)
        batch = resize(batch, 224, antialias=True)
        batch = center_crop(batch, 224)
        batch = normalize(
            batch,
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )
        out = model.encode_image(batch.to(_CLIP_DTYPE))
        feats.append(F.normalize(out.float(), dim=-1).cpu())
    return torch.cat(feats, dim=0)


@torch.no_grad()
def clip_encode_texts(
    texts: list[str],
    device,
    *,
    batch_size: int = 256,
) -> torch.Tensor:
    model, tokenizer = get_clip(device)
    feats = []
    for i in range(0, len(texts), batch_size):
        tokens = tokenizer(texts[i : i + batch_size]).to(device)
        out = model.encode_text(tokens)
        feats.append(F.normalize(out.float(), dim=-1).cpu())
    return torch.cat(feats, dim=0)


# CLAP (LAION larger_clap_music_and_speech)


def get_clap(device) -> tuple[torch.nn.Module, object]:
    key = f"clap:{_device_key(device)}"
    if key not in _CACHE:
        with preserve_rng_state(torch.device(device)):
            from transformers import ClapModel, ClapProcessor

            name = "laion/larger_clap_music_and_speech"
            model = (
                ClapModel.from_pretrained(name, torch_dtype=_CLAP_DTYPE).to(device).eval()
            )
            processor = ClapProcessor.from_pretrained(name)
            _CACHE[key] = (model, processor)
    return _CACHE[key]


def _resample_to(
    audio_arrays: list[np.ndarray], src_sr: int, tgt_sr: int
) -> list[np.ndarray]:
    if src_sr == tgt_sr:
        return [a.astype(np.float32) for a in audio_arrays]
    try:
        import librosa

        return [
            librosa.resample(a.astype(np.float32), orig_sr=src_sr, target_sr=tgt_sr)
            for a in audio_arrays
        ]
    except ImportError:
        import torchaudio.functional as AF

        return [
            AF.resample(
                torch.from_numpy(a.astype(np.float32)),
                orig_freq=src_sr,
                new_freq=tgt_sr,
            ).numpy()
            for a in audio_arrays
        ]


def _extract_features(output) -> torch.Tensor:
    """Handle both raw tensor and BaseModelOutput returns from transformers."""
    if isinstance(output, torch.Tensor):
        return output
    for attr in ("pooler_output", "last_hidden_state"):
        if hasattr(output, attr):
            t = getattr(output, attr)
            if isinstance(t, torch.Tensor):
                return t if t.ndim == 2 else t[:, 0]
    raise TypeError(f"Cannot extract features from {type(output).__name__}")


@torch.no_grad()
def clap_encode_audio(
    audio_arrays: list[np.ndarray],
    device,
    *,
    sample_rate: int = 16000,
    batch_size: int = 16,
) -> torch.Tensor:
    model, processor = get_clap(device)
    resampled = _resample_to(audio_arrays, sample_rate, 48000)
    feats = []
    for i in range(0, len(resampled), batch_size):
        batch = resampled[i : i + batch_size]
        inputs = processor(
            audio=batch, sampling_rate=48000, return_tensors="pt", padding=True
        )
        inputs = {
            k: (
                v.to(device, dtype=_CLAP_DTYPE)
                if v.is_floating_point()
                else v.to(device)
            )
            for k, v in inputs.items()
            if isinstance(v, torch.Tensor)
        }
        out = _extract_features(model.get_audio_features(**inputs))
        feats.append(F.normalize(out.float(), dim=-1).cpu())
    return torch.cat(feats, dim=0)


@torch.no_grad()
def clap_encode_texts(
    texts: list[str],
    device,
    *,
    batch_size: int = 256,
) -> torch.Tensor:
    model, processor = get_clap(device)
    feats = []
    for i in range(0, len(texts), batch_size):
        inputs = processor(
            text=texts[i : i + batch_size], return_tensors="pt", padding=True
        )
        inputs = {
            k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)
        }
        out = _extract_features(model.get_text_features(**inputs))
        feats.append(F.normalize(out.float(), dim=-1).cpu())
    return torch.cat(feats, dim=0)


# Wav2CLIP


def get_wav2clip():
    key = "wav2clip"
    if key not in _CACHE:
        with preserve_rng_state(torch.device("cpu")):
            import wav2clip

            _CACHE[key] = wav2clip.get_model()
    return _CACHE[key]
