"""Task-independent pluggable evaluation metrics."""

import hashlib
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, Dataset, Subset, TensorDataset

from muni.evaluation.metrics.base import BaseMetric
from muni.evaluation.metrics.utils import frechet_distance, per_sample_class_correct
from muni.util.misc import preserve_rng_state


class ImagePSNR(BaseMetric):
    """Calculates Peak Signal to Noise Ratio for Normalized Float Images."""

    def __init__(self, **kwargs):
        self.mse_sum = 0.0
        self.total = 0

    def update(
        self, preds: torch.Tensor, targets: torch.Tensor | None = None, **kwargs
    ) -> None:
        if targets is None:
            return
        mse = F.mse_loss(preds.detach(), targets.detach(), reduction="none").mean(
            dim=[1, 2, 3]
        )
        self.mse_sum += mse.sum().item()
        self.total += preds.shape[0]

    def compute(self):
        if self.total == 0:
            return {"PSNR": 0.0}
        mse = self.mse_sum / self.total
        psnr = 10 * math.log10(
            4.0 / max(mse, 1e-8)
        )  # Note: signal range is [-1, 1], squared difference is 4.0

        self.mse_sum = 0.0
        self.total = 0
        return {"PSNR": psnr}


class CategoricalAccuracy(BaseMetric):
    """General accuracy evaluation for 1D scalar/index classification pairs."""

    def __init__(self, **kwargs):
        self.correct = 0
        self.total = 0

    def update(
        self, preds: torch.Tensor, targets: torch.Tensor | None = None, **kwargs
    ) -> None:
        if targets is None:
            return
        pred_cls = preds.flatten()
        tgt_cls = targets.flatten()
        self.correct += per_sample_class_correct(pred_cls, tgt_cls).sum().item()
        self.total += pred_cls.shape[0]

    def compute(self):
        acc = self.correct / max(self.total, 1)
        self.correct = 0
        self.total = 0
        return {"Accuracy": acc}


class ImageFID(BaseMetric):
    """FID for streamed or in-memory generated images."""

    _FEATURE_DIM = 2048
    _EXTRACTOR_ID = "pytorch_fid_inception_v3_2048_v1"
    _IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    def __init__(
        self,
        num_samples: int = 256,
        batch_size: int = 64,
        fid_statistics_file: str | None = None,
        reference_dir: str | None = None,
        value_range: str = "minus_one_one",
        device: str | torch.device | None = None,
        feature_extractor_factory=None,
        **kwargs,
    ):
        self.num_samples = int(num_samples)
        self.batch_size = int(batch_size)
        self.fid_statistics_file = fid_statistics_file
        self.reference_dir = reference_dir
        self.value_range = value_range
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._feature_extractor_factory = feature_extractor_factory
        self._feature_extractor: torch.nn.Module | None = None
        self._prediction_parts: list[torch.Tensor] = []

    def update(
        self, preds: torch.Tensor, targets: torch.Tensor | None = None, **kwargs
    ) -> None:
        del targets
        value_range = kwargs.pop("value_range", self.value_range)
        remaining = self.num_samples - sum(len(part) for part in self._prediction_parts)
        if remaining <= 0:
            return
        self._prediction_parts.append(
            self._normalize_images(preds[:remaining], value_range=value_range).cpu()
        )

    def compute_for(
        self,
        generated_images: torch.Tensor,
        *,
        reference_dir: str | None = None,
        fid_statistics_file: str | None = None,
        value_range: str = "zero_one",
    ) -> float:
        """Compute one FID value from an in-memory generated image tensor."""
        self.reference_dir = reference_dir or self.reference_dir
        self.fid_statistics_file = fid_statistics_file or self.fid_statistics_file
        self._prediction_parts = [
            self._normalize_images(
                generated_images[: self.num_samples], value_range=value_range
            ).cpu()
        ]
        return float(self.compute()["FID"])

    @staticmethod
    def _normalize_images(images: torch.Tensor, *, value_range: str) -> torch.Tensor:
        images = images.detach().float()
        if value_range == "uint8":
            return images.div(255.0).clamp(0.0, 1.0)
        if value_range == "zero_one":
            return images.clamp(0.0, 1.0)
        if value_range == "minus_one_one":
            return images.clamp(-1.0, 1.0).mul(0.5).add(0.5)
        raise ValueError(
            "ImageFID value_range must be one of 'uint8', 'zero_one', or "
            f"'minus_one_one', got {value_range!r}"
        )

    def _build_feature_extractor(self) -> torch.nn.Module:
        if self._feature_extractor_factory is not None:
            return self._feature_extractor_factory(self.device)
        from pytorch_fid.inception import InceptionV3

        block_index = InceptionV3.BLOCK_INDEX_BY_DIM[self._FEATURE_DIM]
        return InceptionV3([block_index]).to(self.device).eval()

    def _get_feature_extractor(self) -> torch.nn.Module:
        if self._feature_extractor is None:
            with preserve_rng_state(torch.device(self.device)):
                self._feature_extractor = self._build_feature_extractor()
        return self._feature_extractor

    @staticmethod
    def _pool_features(features: torch.Tensor) -> torch.Tensor:
        if features.shape[-2:] != (1, 1):
            features = F.adaptive_avg_pool2d(features, output_size=(1, 1))
        return features.flatten(1)

    def _extract_features(self, model: torch.nn.Module, dataset: Dataset) -> np.ndarray:
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            generator=torch.Generator(),
        )
        parts: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                needs_uint8_scale = batch.numel() > 0 and bool(batch.max() > 1.0)
                images = batch.to(self.device).float()
                if needs_uint8_scale:
                    images = images.div(255.0)
                parts.append(self._pool_features(model(images)[0]).cpu())
        if not parts:
            return np.empty((0, self._FEATURE_DIM), dtype=np.float32)
        return torch.cat(parts, dim=0).numpy()

    @staticmethod
    def _statistics_from_features(
        features: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(features) < 2:
            raise ValueError("FID statistics require at least two samples")
        return (
            np.mean(features, axis=0).astype(np.float32),
            np.cov(features, rowvar=False).astype(np.float64),
        )

    def compute_statistics(
        self,
        images: Dataset | str | os.PathLike[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return Inception feature mean and covariance for a dataset."""
        dataset = (
            self._directory_dataset(os.fspath(images))
            if isinstance(images, (str, os.PathLike))
            else images
        )
        model = self._get_feature_extractor()
        return self._statistics_from_features(self._extract_features(model, dataset))

    def extract_features(
        self,
        images: torch.Tensor,
        *,
        value_range: str = "zero_one",
    ) -> np.ndarray:
        """Extract CPU Inception features from one decoded image chunk."""
        normalized = self._normalize_images(images, value_range=value_range).cpu()
        dataset: Dataset = torch.utils.data.TensorDataset(normalized)
        return self._extract_features(self._get_feature_extractor(), dataset)

    def compute_from_features(
        self,
        generated_features: np.ndarray | torch.Tensor,
        *,
        reference_dir: str | None = None,
        fid_statistics_file: str | None = None,
    ) -> float:
        """Compute FID from streamed ``(N, 2048)`` Inception features."""
        self.reference_dir = reference_dir or self.reference_dir
        self.fid_statistics_file = fid_statistics_file or self.fid_statistics_file
        if isinstance(generated_features, torch.Tensor):
            generated_features = generated_features.detach().cpu().numpy()

        generated_mean, generated_covariance = self._statistics_from_features(
            generated_features
        )
        reference_mean, reference_covariance = self._reference_statistics()
        return frechet_distance(
            generated_mean,
            generated_covariance,
            reference_mean,
            reference_covariance,
        )

    def save_statistics(
        self,
        images: Dataset | str | os.PathLike[str],
        output_path: str | os.PathLike[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute and persist canonical FID reference statistics."""
        mean, covariance = self.compute_statistics(images)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output,
            mu=mean,
            sigma=covariance,
            extractor=np.asarray(self._EXTRACTOR_ID),
        )
        return mean, covariance

    @classmethod
    def _directory_dataset(cls, directory: str) -> Dataset:
        from PIL import Image
        import torchvision.transforms.functional as TF

        files = sorted(
            path
            for path in Path(directory).rglob("*")
            if path.suffix.lower() in cls._IMAGE_EXTENSIONS
        )

        class ImageFileDataset(Dataset):
            def __len__(self) -> int:
                return len(files)

            def __getitem__(self, index: int) -> torch.Tensor:
                with Image.open(files[index]) as source:
                    return TF.to_tensor(source.convert("RGB"))

        return ImageFileDataset()

    @classmethod
    def _stats_cache_path(cls, reference_dir: str) -> Path:
        resolved = str(Path(reference_dir).resolve())
        cache_key = f"{cls._EXTRACTOR_ID}:{resolved}"
        digest = hashlib.sha256(cache_key.encode()).hexdigest()[:16]
        root = Path(__file__).resolve().parents[2] / "data" / "evaluation_stats"
        return root / f"fid_ref__{digest}.npz"

    @classmethod
    def _load_statistics_file(cls, path: str | os.PathLike[str]):
        with np.load(path) as data:
            if "extractor" in data.files:
                extractor = str(data["extractor"].item())
                if extractor != cls._EXTRACTOR_ID:
                    raise ValueError(
                        f"FID statistics at {path} use extractor {extractor!r}; "
                        f"expected {cls._EXTRACTOR_ID!r}"
                    )
            return data["mu"], data["sigma"]

    def _reference_statistics(
        self, model: torch.nn.Module | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.fid_statistics_file:
            return self._load_statistics_file(self.fid_statistics_file)
        if not self.reference_dir or not Path(self.reference_dir).is_dir():
            raise FileNotFoundError("FID requires a statistics file or a reference image directory")

        cache_path = self._stats_cache_path(self.reference_dir)
        if cache_path.is_file():
            return self._load_statistics_file(cache_path)

        features = self._extract_features(
            model if model is not None else self._get_feature_extractor(),
            self._directory_dataset(self.reference_dir),
        )
        mean, covariance = self._statistics_from_features(features)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_path,
            mu=mean,
            sigma=covariance,
            extractor=np.asarray(self._EXTRACTOR_ID),
        )
        return mean, covariance

    def compute(self):
        if not self._prediction_parts:
            return {"FID": float("nan")}
        torch.cat([part[:0] for part in self._prediction_parts], dim=0)
        parts = ConcatDataset([TensorDataset(part) for part in self._prediction_parts])
        dataset: Dataset = Subset(parts, range(len(parts))[: self.num_samples])

        model = self._get_feature_extractor()
        features = self._extract_features(model, dataset)
        self._prediction_parts = []
        if len(features) == 0:
            return {"FID": float("nan")}

        return {"FID": self.compute_from_features(features)}


class ClassHistogramDistance(BaseMetric):
    """L1 distance between the generated class distribution and a uniform distribution."""

    def __init__(
        self, num_samples: int = 256, num_classes: int | None = None, **kwargs
    ):
        self.num_classes = num_classes
        self._preds: list[torch.Tensor] = []

    def update(
        self, preds: torch.Tensor, targets: torch.Tensor | None = None, **kwargs
    ) -> None:
        self._preds.append(preds.flatten().long().cpu())

    def compute(self):
        if not self._preds:
            return {"HistDist": float("nan")}
        all_preds = torch.cat(self._preds)
        self._preds = []

        nc = self.num_classes or int(all_preds.max().item()) + 1
        if nc <= 0:
            return {"HistDist": float("nan")}

        valid = all_preds.clamp(0, nc - 1)
        counts = torch.bincount(valid, minlength=nc)[:nc].float()
        total = counts.sum().clamp_min(1)
        hist = counts / total
        uniform = torch.ones(nc) / nc

        return {"HistDist": (hist - uniform).abs().sum().item()}
