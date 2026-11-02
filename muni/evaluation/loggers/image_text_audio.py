"""Decode and log generated image-text-audio features."""

import torch
import numpy as np

from muni.evaluation.loggers.common import BaseLogger
from muni.evaluation.image_text_audio.models import bind_codec
from muni.util.misc import unwrap_modality_feature




def _log_source(
    logger, writer, step, prefix_name, sources, raw_source_gt, n, device,
    sample_rate=16000, *, audio_destination=False,
):
    """Prefer raw source GT; otherwise lazily decode the destination's source slice."""
    if raw_source_gt is not None and logger.source_decoder_type is not None:
        _log_raw_source(
            writer, step, prefix_name,
            raw_source_gt[:n] if hasattr(raw_source_gt, "__len__") else raw_source_gt,
            logger.source_decoder_type, sample_rate,
        )
    elif sources is not None and logger.source_decoder_type is not None:
        source_decoder = logger.source_codec.load(device)
        # Audio destinations slice before unwrapping nested features.
        feats = (sources[:min(n, 4)] if audio_destination
                 else unwrap_modality_feature(sources)[:n])
        _log_decoded_source(
            writer, step, prefix_name, source_decoder,
            logger.source_decoder_type, feats, device, sample_rate,
        )


def _write_waveforms(
    writer, waveforms, tag_prefix, step, sample_rate, *, raw=False, num_samples=None,
):
    """Emit at most four waveforms, converting raw numpy items only as emitted."""
    if num_samples is None:
        num_samples = len(waveforms) if raw else waveforms.shape[0]
    for i in range(min(num_samples, 4)):
        if raw:
            wav = torch.from_numpy(waveforms[i]).float()
        else:
            wav = waveforms[i] if waveforms.ndim == 2 else waveforms[i : i + 1]
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        writer.add_audio(f"{tag_prefix}_{i}", wav, step, sample_rate=sample_rate)


def _log_decoded_source(
    writer,
    step,
    prefix_name,
    source_decoder,
    source_type,
    feats,
    device,
    sample_rate=16000,
):
    feats = unwrap_modality_feature(feats)
    if source_type == "text":
        with torch.no_grad():
            texts = source_decoder(feats.to(device))
        _write_source_text(writer, step, prefix_name, texts)

    elif source_type == "image":
        import torchvision.utils as vutils

        with torch.no_grad():
            images = source_decoder(feats.to(device))
        if isinstance(images, torch.Tensor):
            images = images.clamp(0, 1)
        else:
            from torchvision.transforms.functional import to_tensor

            images = torch.stack([to_tensor(img) for img in images])
        nrow = max(1, int(feats.shape[0] ** 0.5))
        grid = vutils.make_grid(images, nrow=nrow, normalize=False)
        writer.add_image(f"{prefix_name}/source_image", grid, step)

    elif source_type == "audio":
        with torch.no_grad():
            waveforms = source_decoder(feats.to(device))
        if isinstance(waveforms, np.ndarray):
            waveforms = torch.from_numpy(waveforms)
        waveforms = waveforms.float().cpu()
        _write_waveforms(
            writer, waveforms, f"{prefix_name}/source_audio", step, sample_rate,
            num_samples=feats.shape[0],
        )


def _write_source_text(writer, step, prefix_name, texts):
    lines = ["| idx | source text |", "| --- | --- |"]
    for i, text in enumerate(texts):
        escaped = text.replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {i} | {escaped} |")
    writer.add_text(f"{prefix_name}/source_text", "\n".join(lines), step)


def _log_raw_source(
    writer, step, prefix_name, raw_data, source_type, sample_rate=16000
):
    if source_type == "text":
        _write_source_text(writer, step, prefix_name, raw_data)

    elif source_type == "image":
        import torchvision.utils as vutils

        images = (
            raw_data.clamp(0, 1) if isinstance(raw_data, torch.Tensor) else raw_data
        )
        nrow = max(1, int(images.shape[0] ** 0.5))
        grid = vutils.make_grid(images, nrow=nrow, normalize=False)
        writer.add_image(f"{prefix_name}/source_image", grid, step)

    elif source_type == "audio":
        _write_waveforms(
            writer, raw_data, f"{prefix_name}/source_audio", step, sample_rate, raw=True,
        )


class DecodedImageLogger(BaseLogger):
    """Decode (B, 768) image features via Stable-UnCLIP and log as image grid."""

    def __init__(
        self,
        num_vis_samples: int = 16,
        decoder_name: str = "stable-unclip-small",
        source_decoder_type: str | None = None,
        source_decoder_name: str | None = None,
        checkpoint_paths: dict | None = None,
        renderer_settings: dict | None = None,
        device: str | torch.device | None = None,
    ):
        self.num_vis_samples = num_vis_samples
        self.codec = bind_codec("image", decoder_name, checkpoint_paths, renderer_settings)
        self.source_decoder_type = source_decoder_type
        self.source_codec = (
            bind_codec(source_decoder_type, source_decoder_name, checkpoint_paths, renderer_settings)
            if source_decoder_type is not None else None
        )
        self.device = torch.device(device) if device is not None else None

    def _decode_images(self, feats, device):
        decoder = self.codec.load(device)
        with torch.no_grad():
            images = decoder(feats.to(device))
        if isinstance(images, torch.Tensor):
            return images.clamp(0, 1)
        from torchvision.transforms.functional import to_tensor

        return torch.stack([to_tensor(img) for img in images])

    def log(
        self,
        writer,
        step,
        preds,
        targets,
        prefix_name,
        sources=None,
        raw_target_gt=None,
        raw_source_gt=None,
        **kwargs,
    ):
        import torchvision.utils as vutils

        device = self.device if self.device is not None else preds.device
        n = min(preds.shape[0], self.num_vis_samples)

        pred_images = self._decode_images(preds[:n], device)
        nrow = max(1, int(n**0.5))

        if targets is not None or raw_target_gt is not None:
            if raw_target_gt is not None and isinstance(raw_target_gt, torch.Tensor):
                target_images = raw_target_gt[:n].clamp(0, 1)
            else:
                target_images = self._decode_images(targets[:n], device)
            writer.add_image(
                f"{prefix_name}/pred",
                vutils.make_grid(pred_images, nrow=nrow, normalize=False),
                step,
            )
            writer.add_image(
                f"{prefix_name}/target",
                vutils.make_grid(target_images, nrow=nrow, normalize=False),
                step,
            )
        else:
            grid = vutils.make_grid(pred_images, nrow=nrow, normalize=False)
            writer.add_image(f"{prefix_name}/grid", grid, step)

        _log_source(self, writer, step, prefix_name, sources, raw_source_gt, n, device)


class DecodedAudioLogger(BaseLogger):
    """Decode (B, 768) audio features via AudioLDM and log as audio."""

    def __init__(
        self,
        num_vis_samples: int = 8,
        decoder_name: str = "audioldm-m-full",
        sample_rate: int = 16000,
        source_decoder_type: str | None = None,
        source_decoder_name: str | None = None,
        checkpoint_paths: dict | None = None,
        renderer_settings: dict | None = None,
        device: str | torch.device | None = None,
    ):
        self.num_vis_samples = num_vis_samples
        self.codec = bind_codec("audio", decoder_name, checkpoint_paths)
        self.sample_rate = sample_rate
        self.source_decoder_type = source_decoder_type
        self.source_codec = (
            bind_codec(source_decoder_type, source_decoder_name, checkpoint_paths, renderer_settings)
            if source_decoder_type is not None else None
        )
        self.device = torch.device(device) if device is not None else None

    def _decode_audio(self, feats, device):
        decoder = self.codec.load(device)
        with torch.no_grad():
            waveforms = decoder(feats.to(device))
        if isinstance(waveforms, np.ndarray):
            waveforms = torch.from_numpy(waveforms)
        return waveforms.float().cpu()

    def log(
        self,
        writer,
        step,
        preds,
        targets,
        prefix_name,
        sources=None,
        raw_target_gt=None,
        raw_source_gt=None,
        **kwargs,
    ):
        device = self.device if self.device is not None else preds.device
        n = min(preds.shape[0], self.num_vis_samples)

        pred_waveforms = self._decode_audio(preds[:n], device)

        if targets is not None or raw_target_gt is not None:
            _write_waveforms(
                writer, pred_waveforms, f"{prefix_name}/pred", step, self.sample_rate
            )
            if raw_target_gt is not None and isinstance(raw_target_gt, list):
                _write_waveforms(
                    writer,
                    raw_target_gt[:n],
                    f"{prefix_name}/target",
                    step,
                    self.sample_rate,
                    raw=True,
                )
            else:
                target_waveforms = self._decode_audio(targets[:n], device)
                _write_waveforms(
                    writer,
                    target_waveforms,
                    f"{prefix_name}/target",
                    step,
                    self.sample_rate,
                )
        else:
            _write_waveforms(
                writer, pred_waveforms, f"{prefix_name}/sample", step, self.sample_rate
            )

        _log_source(
            self, writer, step, prefix_name, sources, raw_source_gt, n, device,
            self.sample_rate, audio_destination=True,
        )


class DecodedTextLogger(BaseLogger):
    """Decode text features and log predictions with tokenized ground truth."""

    def __init__(
        self,
        num_vis_samples: int = 16,
        decoder_name: str = "gemmae",
        source_decoder_type: str | None = None,
        source_decoder_name: str | None = None,
        checkpoint_paths: dict | None = None,
        tokenizer_path: str | None = None,
        renderer_settings: dict | None = None,
        device: str | torch.device | None = None,
    ):
        self.num_vis_samples = num_vis_samples
        self.codec = bind_codec("text", decoder_name, checkpoint_paths)
        self.source_decoder_type = source_decoder_type
        self.source_codec = (
            bind_codec(source_decoder_type, source_decoder_name, checkpoint_paths, renderer_settings)
            if source_decoder_type is not None else None
        )
        text_cfg = (checkpoint_paths or {}).get("text", {}) if isinstance(checkpoint_paths, dict) else {}
        self.tokenizer_path = (
            tokenizer_path if tokenizer_path is not None
            else text_cfg.get("decoder_dir") if isinstance(text_cfg, dict) else None
        )
        self._tokenizer = None
        self.device = torch.device(device) if device is not None else None



    def _ensure_tokenizer(self):
        if self._tokenizer is None:
            path = self.tokenizer_path
            if path is None:
                return None
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(path)
        return self._tokenizer

    def log(
        self,
        writer,
        step,
        preds,
        targets,
        prefix_name,
        sources=None,
        raw_target_gt=None,
        raw_source_gt=None,
        target_batches=None,
        **kwargs,
    ):
        # AR decoders return text directly; feature decoders need GemmaE.
        preds_is_strings = (
            isinstance(preds, list) and preds and isinstance(preds[0], str)
        )

        if preds_is_strings:
            n = min(len(preds), self.num_vis_samples)
            device = self.device if self.device is not None else torch.device("cpu")
            pred_texts = preds[:n]
        else:
            n = min(preds.shape[0], self.num_vis_samples)
            device = self.device if self.device is not None else preds.device
            decoder = self.codec.load(device)
            with torch.no_grad():
                pred_texts = decoder(preds[:n].to(device))

        def escape(text: str) -> str:
            return text.replace("|", "\\|").replace("\n", " ")

        if targets is not None or raw_target_gt is not None:
            if raw_target_gt is not None and isinstance(raw_target_gt, list):
                target_texts = raw_target_gt[:n]
            else:
                target_texts = []
                for target_batch in target_batches if target_batches is not None else [targets]:
                    remaining = n - len(target_texts)
                    if remaining <= 0:
                        break
                    features = unwrap_modality_feature(target_batch)
                    count = min(remaining, len(features))
                    if isinstance(target_batch, dict) and "tokens" in target_batch:
                        tok = self._ensure_tokenizer()
                        texts = (
                            tok.batch_decode(target_batch["tokens"][:count], skip_special_tokens=True)
                            if tok is not None else ["<tokens>" for _ in range(count)]
                        )
                    elif preds_is_strings:
                        texts = ["<not-available>" for _ in range(count)]
                    else:
                        decoder = self.codec.load(device)
                        with torch.no_grad():
                            texts = decoder(features[:count].to(device))
                    target_texts.extend(texts)
            lines = ["| idx | pred | target |", "| --- | --- | --- |"]
            for i in range(min(len(pred_texts), len(target_texts))):
                lines.append(
                    f"| {i} | {escape(pred_texts[i])} | {escape(target_texts[i])} |"
                )
        else:
            lines = ["| idx | generated |", "| --- | --- |"]
            for i, text in enumerate(pred_texts):
                lines.append(f"| {i} | {escape(text)} |")

        writer.add_text(f"{prefix_name}/samples", "\n".join(lines), step)

        _log_source(self, writer, step, prefix_name, sources, raw_source_gt, n, device)
