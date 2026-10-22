"""Fixed feature basis; all checkpoints are explicit local assets."""
from __future__ import annotations

import importlib
import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .records import read_audio

STATS = {
    "text": ("gemmae_text_mean.pt", "gemmae_text_std.pt"),
    "image": ("unclip_img_mean.pt", "unclip_img_std.pt"),
    "audio": ("clap_audio_mean.pt", "clap_audio_std.pt"),
}


def local_asset(assets: dict, name: str, directory: bool = False) -> Path:
    value = assets.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"assets.{name}: explicit local path required")
    path = Path(value).expanduser()
    if not (path.is_dir() if directory else path.is_file()):
        raise ValueError(f"assets.{name}: required local asset absent: {path}")
    return path.resolve()


def checkpoint(path: Path) -> dict:
    # Released CLAP bundles include NumPy training metadata; use trusted weights.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if "state_dict" in state:
        state = state["state_dict"]
    return {key.removeprefix("module."): value for key, value in state.items()}


class FeatureEncoders:
    def __init__(self, assets: dict, modalities: set[str], device: str):
        self.device = torch.device(device)
        self.assets = assets
        self.stats = {}
        root = local_asset(assets, "stats_root", directory=True)
        for modality in modalities:
            values = []
            for filename in STATS[modality]:
                path = root / filename
                value = torch.load(path, map_location=self.device, weights_only=True).float()
                if value.shape not in ((768,), (1, 768)) or not torch.isfinite(value).all():
                    raise ValueError(f"{modality}: invalid normalization tensor: {path}")
                values.append(value.reshape(768))
            if not (values[1] > 0).all():
                raise ValueError(f"{modality}: normalization standard deviations must be positive: {root}")
            self.stats[modality] = values
        self.text_model = self.image_model = self.audio_model = None
        # Check every requested asset before constructing any encoder.
        for modality in modalities:
            for name in {"text": ("text_model",), "image": ("clip_checkpoint",), "audio": ("clap_checkpoint", "audio_bridge")}[modality]:
                local_asset(assets, name, directory=name == "text_model")

    def standardize(self, modality: str, values: torch.Tensor) -> np.ndarray:
        mean, std = self.stats[modality]
        result = ((values.float() - mean) / std).detach().cpu().numpy()
        if result.ndim != 2 or result.shape[1] != 768 or not np.isfinite(result).all():
            raise ValueError(f"{modality}: encoder produced invalid/nonfinite 768-dimensional features")
        return result

    def encode_text(self, captions: list[str]) -> np.ndarray:
        if self.text_model is None:
            from sentence_transformers import SentenceTransformer
            self.text_model = SentenceTransformer(
                str(local_asset(self.assets, "text_model", directory=True)),
                device=str(self.device), local_files_only=True,
                model_kwargs={"dtype": torch.bfloat16},
            ).eval()
        with torch.inference_mode():
            values = self.text_model.encode(captions, convert_to_tensor=True, normalize_embeddings=False)
        return self.standardize("text", values)

    def encode_images(self, paths: list[Path]) -> np.ndarray:
        if self.image_model is None:
            import clip
            candidate = clip.load(str(local_asset(self.assets, "clip_checkpoint")), device=self.device, jit=False)
            candidate[0].eval()
            visual = candidate[0].visual
            if visual.input_resolution != 224 or visual.conv1.weight.shape != (1024, 3, 14, 14) or len(visual.transformer.resblocks) != 24:
                raise ValueError("assets.clip_checkpoint: expected OpenAI CLIP ViT-L/14")
            self.image_model = candidate
        model, preprocess = self.image_model
        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(preprocess(image.convert("RGB")))
        with torch.inference_mode():
            values = model.encode_image(torch.stack(images).to(self.device))
        return self.standardize("image", values)

    def load_audio(self) -> None:
        # Import the installed standalone clap_module, not laion_clap.__init__:
        # the latter imports unused training tokenizers with Hub side effects.
        package_root = Path(importlib.metadata.distribution("laion-clap").locate_file("laion_clap"))
        sys.path.insert(0, str(package_root))
        try:
            htsat = importlib.import_module("clap_module.htsat")
            model_module = importlib.import_module("clap_module.model")
        finally:
            sys.path.remove(str(package_root))
        with (package_root / "clap_module/model_configs/HTSAT-base.json").open() as handle:
            config = json.load(handle)
        branch = htsat.create_htsat_model(model_module.CLAPAudioCfp(**config["audio_cfg"]), enable_fusion=False)
        projection = torch.nn.Sequential(torch.nn.Linear(config["embed_dim"], 512), torch.nn.ReLU(), torch.nn.Linear(512, 512))
        state = checkpoint(local_asset(self.assets, "clap_checkpoint"))
        for module, prefix in ((branch, "audio_branch."), (projection, "audio_projection.")):
            module.load_state_dict({key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}, strict=True)
            module.to(self.device).eval()
        bridge = torch.nn.Sequential(torch.nn.Linear(512, 768), torch.nn.ReLU(), torch.nn.Linear(768, 768))
        state = checkpoint(local_asset(self.assets, "audio_bridge"))
        bridge.load_state_dict({key[len("encoder."):]: value for key, value in state.items() if key.startswith("encoder.")}, strict=True)
        bridge.to(self.device).eval()
        self.audio_model = branch, projection, bridge

    def encode_audio(self, paths: list[Path]) -> np.ndarray:
        from torchaudio.functional import resample
        waveforms = []
        for path in paths:
            samples, rate = read_audio(path)
            waveform = torch.from_numpy(samples).mean(dim=1)
            if rate != 48000:
                waveform = resample(waveform, rate, 48000)
            waveform = waveform[:480000]
            if waveform.numel() < 480000:
                waveform = torch.nn.functional.pad(waveform, (0, 480000 - waveform.numel()))
            waveforms.append(waveform)
        if self.audio_model is None:
            self.load_audio()
        batch = torch.stack(waveforms).to(self.device)
        branch, projection, bridge = self.audio_model
        # Exact nonfusion CLAP input after upstream ten-second crop/pad:
        # get_audio_features adds only waveform and longer=False at this length.
        data = {"waveform": batch, "longer": torch.zeros((len(paths), 1), dtype=torch.bool, device=self.device)}
        with torch.inference_mode():
            embedded = branch(data, mixup_lambda=None, device=self.device)["embedding"]
            values = bridge(torch.nn.functional.normalize(projection(embedded), dim=-1))
        return self.standardize("audio", values)
