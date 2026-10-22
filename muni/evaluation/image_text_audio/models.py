"""Load pretrained decoders for image-text-audio 768-dimensional features."""

from pathlib import Path
import json
from collections.abc import Mapping

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from muni.models.components.transformer import GemmaePrefixProjector
from muni.util.misc import preserve_rng_state

# ``models.py`` is ``muni/evaluation/image_text_audio/models.py``.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_STATS_DIR = _PROJECT_ROOT / "assets" / "image_text_audio" / "stats"

HF_REPO = "yeonwoo378/flowbind"

_DECODER_CACHE: dict[tuple[str, ...], object] = {}
_STATS_CACHE: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor]] = {}


class _DtypeOnly:
    """Expose the dtype required by Diffusers with supplied embeddings."""

    def __init__(self, dtype: torch.dtype):
        self.dtype = dtype


def _normalize_device_key(device) -> str:
    device_obj = device if isinstance(device, torch.device) else torch.device(device)
    if device_obj.type == "cuda":
        index = device_obj.index
        if index is None and torch.cuda.is_available():
            index = torch.cuda.current_device()
        return f"cuda:{index}" if index is not None else "cuda"
    return str(device_obj)


def _get_or_create_cached(
    cache: dict[tuple[str, ...], object],
    key: tuple[str, ...],
    factory,
    *,
    device,
):
    if key not in cache:
        with preserve_rng_state(torch.device(device)):
            cache[key] = factory()
    return cache[key]


def _to_plain_data(node):
    if isinstance(node, Mapping):
        return {str(key): _to_plain_data(value) for key, value in node.items()}
    if isinstance(node, (list, tuple)):
        return [_to_plain_data(value) for value in node]
    return node


def _checkpoint_bundle_key(checkpoint_paths) -> str:
    if checkpoint_paths is None:
        return "default"
    return json.dumps(_to_plain_data(checkpoint_paths), sort_keys=True)


def _as_existing_path(path_like) -> str | None:
    if path_like is None:
        return None
    path = Path(str(path_like)).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Configured local model asset does not exist: {path}")
    return str(path)


def _bundle_get(checkpoint_paths, *keys):
    cur = checkpoint_paths
    for key in keys:
        if not isinstance(cur, Mapping) or key not in cur:
            return None
        cur = cur[key]
    return cur


# ---------------------------------------------------------------------------
# Normalization stats
# ---------------------------------------------------------------------------


def load_mean_std(modal, device):
    """Load precomputed mean/std tensors for the given modality."""
    name_map = {
        "text": ("gemmae_text_mean.pt", "gemmae_text_std.pt"),
        "image": ("unclip_img_mean.pt", "unclip_img_std.pt"),
        "audio": ("clap_audio_mean.pt", "clap_audio_std.pt"),
    }
    mean_file, std_file = name_map[modal]
    device_key = _normalize_device_key(device)
    return _get_or_create_cached(
        _STATS_CACHE,
        (modal, device_key),
        lambda: (
            torch.load(_STATS_DIR / mean_file, map_location=device_key),
            torch.load(_STATS_DIR / std_file, map_location=device_key),
        ),
        device=device_key,
    )


# ---------------------------------------------------------------------------
# Text decoder  (GemmaE)
# ---------------------------------------------------------------------------


@torch.no_grad()
def _load_text_dec_uncached(name: str, device_str: str, checkpoint_paths=None):
    if not isinstance(name, str) or not name.startswith("gemmae"):
        raise ValueError(f"Unsupported text decoder: {name!r}")

    prefix_len = int(name.split("_")[-1]) if "_" in name else 3

    # autocast needs device type ("cuda"), not full device string ("cuda:2")
    device_type = device_str.split(":")[0]
    dtype = torch.bfloat16

    text_decoder_dir = _as_existing_path(
        _bundle_get(checkpoint_paths, "text", "decoder_dir")
    )
    gemmae_ckpt_path = _as_existing_path(
        _bundle_get(checkpoint_paths, "text", "gemmae_ckpt")
    )

    ckpt_path = gemmae_ckpt_path or hf_hub_download(
        repo_id=HF_REPO, filename="gemmae_3.pt"
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")

    latent_projector = GemmaePrefixProjector(768, 1152, prefix_len).to(device_str)
    latent_projector.load_state_dict(ckpt["prefix_proj"])
    latent_projector.eval()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dec_model = AutoModelForCausalLM.from_pretrained(
        text_decoder_dir or "google/gemma-3-1b-pt",
        dtype=dtype,
        device_map=device_str,
        attn_implementation="eager",
    )
    dec_model.load_state_dict(ckpt["model"])
    dec_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        text_decoder_dir or "google/gemma-3-1b-pt"
    )

    mean_pt, std_pt = load_mean_std("text", device_str)

    @torch.autocast(device_type=device_type, dtype=dtype)
    def decode_feat(enc_vec):
        enc_vec = enc_vec.to(device=mean_pt.device, dtype=torch.float32)
        enc_vec = enc_vec * std_pt + mean_pt
        prefix = latent_projector(enc_vec)
        bos_ids = torch.full(
            (enc_vec.shape[0], 1), tokenizer.bos_token_id, device=device_str
        )
        bos_emb = dec_model.get_input_embeddings()(bos_ids).to(dtype)
        inputs_embeds = torch.cat([prefix, bos_emb], dim=1)
        attn = torch.ones(
            (enc_vec.shape[0], prefix_len + 1), device=device_str, dtype=torch.long
        )
        gen_ids = getattr(dec_model, "module", dec_model).generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            max_new_tokens=64,
            do_sample=False,
            top_p=None,
            top_k=None,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
        return [tokenizer.decode(ids, skip_special_tokens=True) for ids in gen_ids]

    return decode_feat


@torch.no_grad()
def load_text_dec(name: str, device, checkpoint_paths=None):
    """Return a cached callable decoding text features."""
    device_key = _normalize_device_key(device)
    if device_key == "cpu" and torch.cuda.is_available():
        device_key = _normalize_device_key(torch.device("cuda"))
    bundle_key = _checkpoint_bundle_key(checkpoint_paths)

    return _get_or_create_cached(
        _DECODER_CACHE,
        ("text", name, device_key, bundle_key),
        lambda: _load_text_dec_uncached(
            name, device_key, checkpoint_paths=checkpoint_paths
        ),
        device=device_key,
    )


# ---------------------------------------------------------------------------
# Image decoders  (Stable-UnCLIP / Stable Cascade)
# ---------------------------------------------------------------------------

_CASCADE_DECODER_NAME = "stable-cascade-large"
_CASCADE_TEXT_SEQUENCE_LENGTH = 77
_CASCADE_TEXT_EMBED_DIM = 1280
_CASCADE_EMPTY_TEXT_EMBEDDINGS = (
    _PROJECT_ROOT
    / "assets"
    / "image_text_audio"
    / "stable_cascade_empty_prompt_bigg_bf16.pt"
)


def resolve_image_renderer_settings(settings=None, *, decoder_name):
    """Resolve controls for the selected decoder, rejecting inactive overrides."""
    settings = dict(settings or {})
    if decoder_name != _CASCADE_DECODER_NAME:
        if settings:
            raise ValueError(
                f"image_renderer controls require {_CASCADE_DECODER_NAME!r}, "
                f"not {decoder_name!r}."
            )
        return {}
    defaults = dict(
        image_size=1024, prior_steps=20, decoder_steps=10,
        prior_guidance_scale=4.0, decoder_guidance_scale=0.0,
        prior_shift=2.0, decoder_shift=1.0, seed=None,
    )
    unknown = settings.keys() - defaults.keys()
    if unknown:
        raise ValueError(f"Unknown image renderer settings: {sorted(unknown)}")
    defaults.update(settings)
    for key in ("image_size", "prior_steps", "decoder_steps"):
        value = defaults[key]
        if isinstance(value, bool) or int(value) != value or value <= 0:
            raise ValueError(f"Image renderer {key} must be a positive integer.")
        defaults[key] = int(value)
    if defaults["image_size"] % 128:
        raise ValueError("Cascade image_size must be divisible by 128.")
    for key in ("prior_shift", "decoder_shift"):
        if not float(defaults[key]) > 0:
            raise ValueError(f"Image renderer {key} must be positive.")
    return defaults


def _load_cascade_pipeline(
    stage: str,
    checkpoint_paths,
    device_key: str,
    dtype: torch.dtype,
):
    """Load one Cascade stage without its unused text/image encoders."""
    from diffusers import (
        StableCascadeDecoderPipeline,
        StableCascadePriorPipeline,
        StableCascadeUNet,
    )

    if stage == "prior":
        path = _as_existing_path(
            _bundle_get(checkpoint_paths, "image", "cascade_prior_dir")
        )
        source = path or "stabilityai/stable-cascade-prior"
        subfolder = "prior"
        unet = StableCascadeUNet.from_pretrained(
            source,
            subfolder=subfolder,
            variant="bf16",
            torch_dtype=dtype,
            local_files_only=bool(path),
        )
        pipe = StableCascadePriorPipeline.from_pretrained(
            source,
            prior=unet,
            tokenizer=None,
            text_encoder=None,
            image_encoder=None,
            feature_extractor=None,
            torch_dtype=dtype,
            local_files_only=bool(path),
        )
    elif stage == "decoder":
        path = _as_existing_path(
            _bundle_get(checkpoint_paths, "image", "cascade_decoder_dir")
        )
        source = path or "stabilityai/stable-cascade"
        subfolder = "decoder"
        unet = StableCascadeUNet.from_pretrained(
            source,
            subfolder=subfolder,
            variant="bf16",
            torch_dtype=dtype,
            local_files_only=bool(path),
        )
        pipe = StableCascadeDecoderPipeline.from_pretrained(
            source,
            decoder=unet,
            tokenizer=None,
            text_encoder=None,
            torch_dtype=dtype,
            local_files_only=bool(path),
        )
    else:
        raise ValueError(f"Unknown Cascade stage: {stage!r}")

    pipe = pipe.to(device_key)
    pipe.text_encoder = _DtypeOnly(dtype)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def _configure_cascade_shift(scheduler, shift: float) -> None:
    """Apply native GDF log-SNR shift; Diffusers' ``scaler`` is a time warp,
    not this shift. DDPMWuerstchenScheduler exposes custom timesteps but no
    log-SNR shift argument, so transform its timestep grid once per call.
    """
    if shift == 1:
        return
    base_alpha_cumprod = scheduler._alpha_cumprod
    base_set_timesteps = scheduler.set_timesteps

    def shifted_set_timesteps(num_inference_steps=None, timesteps=None, device=None):
        base_set_timesteps(num_inference_steps, timesteps=timesteps, device=device)
        current = scheduler.timesteps
        alpha = base_alpha_cumprod(current, current.device)
        snr = alpha / (1 - alpha)
        shifted_alpha = (snr / (shift * shift)) / (1 + snr / (shift * shift))
        init_alpha = scheduler._init_alpha_cumprod.to(current.device)
        angle = torch.acos((shifted_alpha * init_alpha).clamp(0, 1).sqrt())
        transformed = angle * (2 / torch.pi) * (
            1 + scheduler.s.item()
        ) - scheduler.s.to(current.device)
        scheduler.timesteps = transformed.clamp(0, 1)

    scheduler.set_timesteps = shifted_set_timesteps


def _load_cascade_empty_text_embeddings(
    device_key: str, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load the committed empty-caption condition for image variation."""
    path = _CASCADE_EMPTY_TEXT_EMBEDDINGS
    payload = torch.load(path, map_location="cpu", weights_only=True)
    prompt = payload["prompt_embeds"]
    pooled = payload["prompt_embeds_pooled"]
    if tuple(prompt.shape) != (
        1,
        _CASCADE_TEXT_SEQUENCE_LENGTH,
        _CASCADE_TEXT_EMBED_DIM,
    ) or tuple(pooled.shape) != (1, 1, _CASCADE_TEXT_EMBED_DIM):
        raise ValueError(
            "Cascade empty-text asset has incompatible shapes: "
            f"prompt={tuple(prompt.shape)}, pooled={tuple(pooled.shape)}."
        )
    if not torch.isfinite(prompt).all() or not torch.isfinite(pooled).all():
        raise ValueError(f"Cascade empty-text asset contains non-finite values: {path}")
    return (
        prompt.to(device=device_key, dtype=dtype),
        pooled.to(device=device_key, dtype=dtype),
    )


def _load_image_dec_uncached(
    name: str,
    device_key: str,
    checkpoint_paths=None,
    renderer_settings=None,
):
    if name == _CASCADE_DECODER_NAME:
        settings = resolve_image_renderer_settings(renderer_settings, decoder_name=name)
        dtype = torch.bfloat16
        text_embeddings = _load_cascade_empty_text_embeddings(device_key, dtype)
        prior = _load_cascade_pipeline("prior", checkpoint_paths, device_key, dtype)
        decoder = _load_cascade_pipeline("decoder", checkpoint_paths, device_key, dtype)
        _configure_cascade_shift(prior.scheduler, settings["prior_shift"])
        _configure_cascade_shift(decoder.scheduler, settings["decoder_shift"])
        mean_pt, std_pt = load_mean_std("image", device=device_key)
        mean_pt = mean_pt.to(device_key, dtype=dtype)
        std_pt = std_pt.to(device_key, dtype=dtype)
        generator = (
            torch.Generator(device=device_key).manual_seed(int(settings["seed"]))
            if settings["seed"] is not None else None
        )

        def decode_cascade(image_embs):
            image_embs = image_embs.to(device_key, dtype=dtype).detach()
            raw_embs = (image_embs * std_pt + mean_pt).unsqueeze(1)
            prompt, pooled = text_embeddings

            # Diffusers broadcasts one feature over all prompts. One prior call
            # per row therefore preserves the input feature/image pairing.
            images = []
            for raw_emb in raw_embs:
                prior_output = prior(
                    prompt=None,
                    prompt_embeds=prompt,
                    prompt_embeds_pooled=pooled,
                    negative_prompt_embeds=prompt,
                    negative_prompt_embeds_pooled=pooled,
                    image_embeds=raw_emb.unsqueeze(0),
                    height=settings["image_size"],
                    width=settings["image_size"],
                    num_inference_steps=settings["prior_steps"],
                    guidance_scale=settings["prior_guidance_scale"],
                    generator=generator,
                    output_type="pt",
                )
                images.append(
                    decoder(
                        image_embeddings=prior_output.image_embeddings,
                        prompt=None,
                        prompt_embeds=prompt,
                        prompt_embeds_pooled=pooled,
                        negative_prompt_embeds=prompt,
                        negative_prompt_embeds_pooled=pooled,
                        guidance_scale=settings["decoder_guidance_scale"],
                        num_inference_steps=settings["decoder_steps"],
                        generator=generator,
                        output_type="pt",
                    ).images
                )
            return torch.cat(images, dim=0)

        return decode_cascade

    if name != "stable-unclip-small":
        raise NotImplementedError(f"Image decoder '{name}' is not implemented.")

    from diffusers import DiffusionPipeline

    image_decoder_dir = _as_existing_path(
        _bundle_get(checkpoint_paths, "image", "decoder_dir")
    )

    pipe = DiffusionPipeline.from_pretrained(
        image_decoder_dir or "sd2-community/stable-diffusion-2-1-unclip-small",
        torch_dtype=torch.float16,
        local_files_only=bool(image_decoder_dir),
    ).to(device_key)
    pipe.vae.enable_slicing()
    pipe.set_progress_bar_config(disable=True)

    mean_pt, std_pt = load_mean_std("image", device=device_key)
    mean_pt = mean_pt.half()
    std_pt = std_pt.half()

    def _decode(pipe, image_embs):
        image_embs = image_embs.to(device_key, dtype=torch.float16).detach()
        image_embs = image_embs * std_pt + mean_pt
        bs = image_embs.shape[0]
        images = pipe(
            prompt=[""] * bs,
            image_embeds=image_embs,
            output_type="pt",
        ).images
        return images

    return lambda x: _decode(pipe, x.to(device_key))


def load_image_dec(name: str, device, checkpoint_paths=None, renderer_settings=None):
    """Return a callable that maps (B, 768) to images in [0, 1]."""
    device_key = _normalize_device_key(device)
    bundle_key = _checkpoint_bundle_key(checkpoint_paths)
    settings = resolve_image_renderer_settings(renderer_settings, decoder_name=name)
    settings_key = _checkpoint_bundle_key(settings)
    return _get_or_create_cached(
        _DECODER_CACHE,
        ("image", name, device_key, bundle_key, settings_key),
        lambda: _load_image_dec_uncached(
            name, device_key, checkpoint_paths=checkpoint_paths,
            renderer_settings=settings,
        ),
        device=device_key,
    )


# ---------------------------------------------------------------------------
# Audio decoder  (AudioLDM)
# ---------------------------------------------------------------------------


def _load_audio_dec_uncached(name: str, device_key: str, checkpoint_paths=None):
    if name != "audioldm-m-full":
        raise NotImplementedError(f"Audio decoder '{name}' is not implemented.")

    from diffusers import AudioLDMPipeline

    audio_decoder_dir = _as_existing_path(
        _bundle_get(checkpoint_paths, "audio", "decoder_dir")
    )
    audio_ae_ckpt_path = _as_existing_path(
        _bundle_get(checkpoint_paths, "audio", "bridge_ckpt")
    )

    model = AudioLDMPipeline.from_pretrained(
        audio_decoder_dir or "cvssp/audioldm-m-full",
        torch_dtype=torch.float16,
        local_files_only=bool(audio_decoder_dir),
    ).to(device_key)
    model.set_progress_bar_config(disable=True)

    # Only the decoder half is used; retain the checkpoint's decoder weights
    # without allocating its unused encoder half.
    audio_ae = nn.Sequential(
        nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 512),
    ).to(device_key, dtype=torch.float16)
    audio_ae_path = audio_ae_ckpt_path or hf_hub_download(
        repo_id=HF_REPO, filename="lightweight_audio_ae.pth"
    )
    audio_ae.load_state_dict(
        {
            k.removeprefix("decoder."): v.to(device_key).to(torch.float16)
            for k, v in torch.load(audio_ae_path, map_location="cpu").items()
            if k.startswith("decoder.")
        }
    )
    audio_ae.eval()

    audio_mean, audio_std = load_mean_std("audio", device=device_key)
    audio_mean = audio_mean.half()
    audio_std = audio_std.half()

    def _decode(model, audio_ae, audio_embs):
        audio_embs = audio_embs.to(device_key, dtype=torch.float16)
        audio_embs = audio_embs * audio_std + audio_mean
        audio_embs = audio_ae(audio_embs)
        return model(prompt_embeds=audio_embs, num_inference_steps=20).audios

    return lambda x: _decode(model, audio_ae, x.to(device_key))


def load_audio_dec(name: str, device, checkpoint_paths=None):
    """Return a callable that maps (B, 768) -> numpy waveforms."""
    device_key = _normalize_device_key(device)
    bundle_key = _checkpoint_bundle_key(checkpoint_paths)
    return _get_or_create_cached(
        _DECODER_CACHE,
        ("audio", name, device_key, bundle_key),
        lambda: _load_audio_dec_uncached(
            name, device_key, checkpoint_paths=checkpoint_paths
        ),
        device=device_key,
    )
