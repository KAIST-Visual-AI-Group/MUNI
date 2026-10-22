"""Shared metrics for decoded image, audio, and text outputs."""

import hashlib
import os
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from muni.evaluation.metrics.utils import (
    cosine_similarity_scores,
    frechet_distance,
    relative_ranking_scores,
)
from muni.evaluation.image_text_audio.encoders import (
    clap_encode_audio,
    clap_encode_texts,
    clip_encode_images,
    clip_encode_texts,
    get_wav2clip,
)
from muni.util.misc import print_on_main as print
from muni.util.misc import preserve_rng_state


# Reference statistics

_STATS_DIR = Path(__file__).resolve().parents[2] / "data" / "evaluation_stats"


def _ref_stats_cache_path(category: str, key_parts: tuple) -> Path:
    """Stable on-disk filename for cached ``(mu, sigma)`` reference stats."""
    digest = hashlib.sha256(repr(key_parts).encode()).hexdigest()[:16]
    return _STATS_DIR / f"{category}_ref__{digest}.npz"


def _load_or_compute_ref_stats(
    cache_path: Path, compute_fn
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(mu, sigma)``: load from disk or compute then memoise to disk."""
    if cache_path.is_file():
        print(f"  [cache hit] reference stats: {cache_path.name}")
        with np.load(cache_path) as data:
            return data["mu"], data["sigma"]
    mu, sigma = compute_fn()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, mu=mu, sigma=sigma)
    print(f"  [cache write] reference stats: {cache_path.name}")
    return mu, sigma


# CLIPScore (Hessel et al., 2021)


class CLIPScore:
    """Text-image alignment score using CLIP ViT-B/32 (shared encoder)."""

    def __init__(self, device: str | torch.device = "cuda", *, batch_size: int | None = None):
        self.device = torch.device(device)
        self.batch_size = batch_size

    def compute(
        self,
        images: torch.Tensor | None,
        texts: list[str],
        *,
        image_features: torch.Tensor | None = None,
    ) -> tuple[float, torch.Tensor]:
        try:
            img_feats = (
                clip_encode_images(images, self.device, batch_size=self.batch_size or 64)
                if image_features is None
                else image_features
            )
            txt_feats = clip_encode_texts(texts, self.device, batch_size=self.batch_size or 256)
        except ImportError as exc:
            warnings.warn(f"CLIPScore unavailable ({exc}) — returning NaN.")
            nan = torch.full((len(texts),), float("nan"))
            return float("nan"), nan
        scores = cosine_similarity_scores(img_feats, txt_feats)
        return float(scores.mean().item()), scores


# CLAPScore (Elizalde et al., 2023)


class CLAPScore:
    """Text-audio alignment score using LAION CLAP (shared encoder)."""

    def __init__(self, device: str | torch.device = "cuda", *, batch_size: int | None = None):
        self.device = torch.device(device)
        self.batch_size = batch_size

    def compute(
        self,
        audio_arrays: list[np.ndarray],
        texts: list[str],
        sample_rate: int = 16000,
    ) -> tuple[float, torch.Tensor]:
        try:
            a_feats = clap_encode_audio(
                audio_arrays, self.device, sample_rate=sample_rate, batch_size=self.batch_size or 16
            )
            t_feats = clap_encode_texts(texts, self.device, batch_size=self.batch_size or 256)
        except ImportError as exc:
            warnings.warn(f"CLAPScore unavailable ({exc}) — returning NaN.")
            nan = torch.full((len(texts),), float("nan"))
            return float("nan"), nan
        scores = cosine_similarity_scores(a_feats, t_feats)
        return float(scores.mean().item()), scores


# Audio-image similarity (Wu et al., 2022; Biner et al., 2024)


class AudioImageSimilarity:
    """Relative AIS using wav2clip audio + shared CLIP image embeddings.

    A→I: ratio of test-set audios with *worse* cosine similarity than the
         conditioning audio, measured against the generated image.
    I→A: symmetric (ratio of test-set images with worse cosine similarity).
    """

    audio_batch_size = 32

    def __init__(self, device: str | torch.device = "cuda", *, image_batch_size: int | None = None):
        self.device = torch.device(device)
        self.image_batch_size = image_batch_size

    @torch.no_grad()
    def _encode_audio(self, audio_arrays: list[np.ndarray]) -> torch.Tensor:
        """Encode UL's ordered, padded audio cohorts for relative AIS."""
        import wav2clip

        if not audio_arrays:
            return torch.empty((0, 512), dtype=torch.float32)
        model = get_wav2clip()
        features = []
        for start in range(0, len(audio_arrays), self.audio_batch_size):
            chunk = audio_arrays[start : start + self.audio_batch_size]
            max_len = max(int(audio.shape[-1]) for audio in chunk)
            batch = np.zeros((len(chunk), max_len), dtype=np.float32)
            for index, audio in enumerate(chunk):
                flat = audio if audio.ndim == 1 else audio.reshape(-1)
                batch[index, : flat.shape[0]] = flat.astype(np.float32, copy=False)
            embedded = wav2clip.embed_audio(batch, model)
            features.append(F.normalize(torch.from_numpy(embedded).float(), dim=-1))
        return torch.cat(features, dim=0)

    def score_a2i(
        self,
        gen_images: torch.Tensor | None,
        cond_audio: list[np.ndarray],
        ref_audio: list[np.ndarray],
        *,
        image_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        try:
            gen_img_f = (
                clip_encode_images(gen_images, self.device, batch_size=self.image_batch_size or 64)
                if image_features is None
                else image_features
            )
            cond_aud_f = self._encode_audio(cond_audio)
            ref_aud_f = (
                cond_aud_f
                if ref_audio is cond_audio
                else self._encode_audio(ref_audio)
            )
        except ImportError as exc:
            warnings.warn(f"AIS unavailable ({exc}) — returning NaN.")
            count = len(gen_images) if image_features is None else len(image_features)
            return torch.full((count,), float("nan"))
        return relative_ranking_scores(gen_img_f, cond_aud_f, ref_aud_f)

    def score_i2a(
        self,
        gen_audio: list[np.ndarray],
        cond_images: torch.Tensor,
        ref_images: torch.Tensor,
    ) -> torch.Tensor:
        try:
            gen_aud_f = self._encode_audio(gen_audio)
            cond_img_f = clip_encode_images(cond_images, self.device, batch_size=self.image_batch_size or 64)
            ref_img_f = (
                cond_img_f
                if ref_images is cond_images
                else clip_encode_images(ref_images, self.device, batch_size=self.image_batch_size or 64)
            )
        except ImportError as exc:
            warnings.warn(f"AIS unavailable ({exc}) — returning NaN.")
            return torch.full((len(gen_audio),), float("nan"))
        return relative_ranking_scores(gen_aud_f, cond_img_f, ref_img_f)


def select_alignment_kind(source_modality: str, target_modality: str) -> str:
    """Select the canonical metric for one decoded modality pair."""
    pair = frozenset({source_modality, target_modality})
    if pair == frozenset({"text", "image"}):
        return "clip"
    if pair == frozenset({"text", "audio"}):
        return "clap"
    if pair == frozenset({"image", "audio"}):
        return "ais"
    raise ValueError(
        "No alignment metric is registered for modality pair "
        f"({source_modality!r}, {target_modality!r})"
    )


def compute_alignment(
    kind: str,
    source_modality: str,
    source_data,
    target_data,
    device: str | torch.device = "cuda",
    *,
    metric=None,
    target_image_features: torch.Tensor | None = None,
) -> tuple[float, torch.Tensor]:
    """Return mean and per-sample alignment; AIS uses the conditioning set as reference."""
    if kind == "clip":
        metric = CLIPScore(device) if metric is None else metric
        texts, images = (
            (source_data, target_data)
            if source_modality == "text"
            else (target_data, source_data)
        )
        return metric.compute(images, texts, image_features=target_image_features)
    if kind == "clap":
        metric = CLAPScore(device) if metric is None else metric
        texts, audio = (
            (source_data, target_data)
            if source_modality == "text"
            else (target_data, source_data)
        )
        return metric.compute(audio, texts)
    if kind == "ais":
        metric = AudioImageSimilarity(device) if metric is None else metric
        if source_modality == "audio":
            scores = metric.score_a2i(
                target_data, source_data, source_data,
                image_features=target_image_features,
            )
        else:
            scores = metric.score_i2a(target_data, source_data, source_data)
        return float(scores.mean().item()) if len(scores) else float("nan"), scores
    raise ValueError(f"Unknown alignment kind: {kind!r}")


# FAD (Kilgour et al., 2019)


class AudioFAD:
    """Fréchet Audio Distance using VGGish with batched GPU inference."""

    def __init__(self, device: str = "cuda", batch_size: int = 64):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self._model = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        with preserve_rng_state(self.device):
            self._model = torch.hub.load(
                "harritaylor/torchvggish",
                "vggish",
                device=self.device,
                postprocess=False,
                preprocess=False,
            )
            # Strip the final ReLU so we get raw embeddings (use_activation=False).
            self._model.embeddings = torch.nn.Sequential(
                *list(self._model.embeddings.children())[:-1]
            )
            self._model = self._model.to(self.device).eval()

    @torch.no_grad()
    def _embed_waveforms(self, waveforms: list[np.ndarray], sr: int) -> np.ndarray:
        from torchvggish import vggish_input

        all_examples: list[np.ndarray] = []
        boundaries: list[tuple[int, int]] = []
        for index, wav in enumerate(waveforms):
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            examples = vggish_input.waveform_to_examples(wav, sr)
            if isinstance(examples, torch.Tensor):
                examples = examples.numpy()
            if len(examples) == 0:
                raise ValueError(f"FAD clip {index} has no VGGish windows at sample rate {sr}")
            start = len(all_examples)
            all_examples.extend(examples)
            boundaries.append((start, len(all_examples)))

        stacked = torch.from_numpy(np.stack(all_examples)).float()
        feats: list[torch.Tensor] = []
        for i in range(0, len(stacked), self.batch_size):
            batch = stacked[i : i + self.batch_size].to(self.device)
            feats.append(self._model(batch).cpu())
        all_feats = torch.cat(feats, dim=0).numpy()

        return np.stack([all_feats[s:e].mean(axis=0) for s, e in boundaries])

    def _load_wav_files(self, audio_dir: str, sr: int) -> list[np.ndarray]:
        import soundfile as sf

        wav_files = sorted(
            f
            for f in os.listdir(audio_dir)
            if f.endswith(".wav") and os.path.isfile(os.path.join(audio_dir, f))
        )
        waveforms = []
        for f in tqdm(wav_files, desc="Loading reference audio"):
            wav, file_sr = sf.read(os.path.join(audio_dir, f), dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if file_sr != sr:
                import resampy

                wav = resampy.resample(wav, file_sr, sr)
            waveforms.append(wav)
        return waveforms

    def compute(
        self,
        generated_waveforms: np.ndarray | list[np.ndarray],
        reference_audio_dir: str | None = None,
        sample_rate: int = 16000,
        fad_stats_file: str | None = None,
    ) -> float:
        if fad_stats_file:
            if not Path(fad_stats_file).is_file():
                raise FileNotFoundError(f"FAD reference statistics not found: {fad_stats_file}")
        elif not reference_audio_dir or not Path(reference_audio_dir).is_dir():
            raise FileNotFoundError("FAD requires a statistics file or a reference audio directory")

        if (
            isinstance(generated_waveforms, np.ndarray)
            and generated_waveforms.ndim == 2
        ):
            gen_list = [generated_waveforms[i] for i in range(len(generated_waveforms))]
        elif isinstance(generated_waveforms, list):
            gen_list = generated_waveforms
        else:
            gen_list = [generated_waveforms]
        if len(gen_list) < 2:
            raise ValueError("FAD requires at least two generated clips for covariance")
        if not fad_stats_file:
            ref_count = sum(
                path.is_file() and path.name.endswith(".wav")
                for path in Path(reference_audio_dir).iterdir()
            )
            if ref_count < 2:
                raise ValueError("FAD requires at least two reference clips for covariance")

        self._ensure_loaded()

        print(
            f"  Computing VGGish embeddings for {len(gen_list)} generated samples ..."
        )
        gen_embds = self._embed_waveforms(gen_list, sample_rate)
        mu1, sigma1 = np.mean(gen_embds, axis=0), np.cov(gen_embds, rowvar=False)

        # Reference stats: explicit file > auto-cache > on-the-fly compute.
        if fad_stats_file:
            with np.load(fad_stats_file) as data:
                mu2, sigma2 = data["mu"], data["sigma"]
        else:
            ref_dir_abs = str(Path(reference_audio_dir).resolve())
            cache_path = _ref_stats_cache_path("fad", (ref_dir_abs, int(sample_rate)))

            def _compute_ref():
                ref_wavs = self._load_wav_files(reference_audio_dir, sample_rate)
                print(
                    f"  Computing VGGish embeddings for {len(ref_wavs)} reference samples ..."
                )
                ref_embds = self._embed_waveforms(ref_wavs, sample_rate)
                return np.mean(ref_embds, axis=0), np.cov(ref_embds, rowvar=False)

            mu2, sigma2 = _load_or_compute_ref_stats(cache_path, _compute_ref)
        return frechet_distance(mu1, sigma1, mu2, sigma2)


# CIDEr (Vedantam et al., 2015)


class CIDEr:
    """CIDEr × 100 for generated text captions."""

    def compute(
        self,
        predictions: dict[str, list[str]],
        references: dict[str, list[str]],
    ) -> float:
        if not predictions or not references:
            return float("nan")
        common = sorted(set(predictions) & set(references))
        if not common:
            warnings.warn("No common keys for CIDEr — returning NaN.")
            return float("nan")
        try:
            from pycocoevalcap.cider.cider import Cider
        except ImportError:
            warnings.warn("pycocoevalcap not installed; CIDEr will be NaN.")
            return float("nan")
        scorer = Cider()
        score, _ = scorer.compute_score(
            {k: references[k] for k in common},
            {k: predictions[k] for k in common},
        )
        return float(score) * 100.0
