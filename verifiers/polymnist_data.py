"""PolyMNIST dataset for the held-out digit/quadrant verifier."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from muni.data.sources.image_cache import load_image_tensor_cache


DEFAULT_MODALITIES = ("m0", "m1", "m2")


class PolyMNISTVerifierDataset(Dataset):
    """Sample one image modality and its shared digit/quadrant labels."""

    def __init__(self, cfg: dict, split: str = "train"):
        data_cfg = cfg["data"]
        self.train = split == "train"
        cache = load_image_tensor_cache(
            data_cfg["cache_path"], labels_csv=data_cfg["csv_path"]
        )
        self.modalities = tuple(data_cfg.get("modalities", DEFAULT_MODALITIES))
        self.images = cache["images"]
        self.modality_index = {
            name: cache["modalities"].index(name) for name in self.modalities
        }
        self.key_index = {key: index for index, key in enumerate(cache["keys"])}
        self.key_format = data_cfg.get("key_format", "{:06d}")

        stem_range = data_cfg[f"{split}_stems"]
        lo, hi = map(int, stem_range)
        with Path(data_cfg["csv_path"]).open(newline="") as handle:
            self.entries = []
            for row in csv.DictReader(handle):
                sample_id = int(row["sample_id"])
                if lo <= sample_id < hi:
                    self.entries.append(
                        (sample_id, int(row["digit"]), int(row["quadrant"]))
                    )

        self.transform = (
            self._build_transform(cfg.get("augment", {})) if split == "train" else None
        )

    @staticmethod
    def _build_transform(cfg: dict) -> A.Compose:
        scale = tuple(cfg.get("scale_range", (0.95, 1.05)))
        translate = float(cfg.get("translate_frac", 0.08))
        rotation = float(cfg.get("rotation_deg", 2.0))
        return A.Compose(
            [
                A.Affine(
                    scale=scale,
                    translate_percent={
                        "x": (-translate, translate),
                        "y": (-translate, translate),
                    },
                    rotate=(-rotation, rotation),
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=0,
                    fill_mask=0,
                    p=float(cfg.get("affine_prob", 0.5)),
                ),
                A.ColorJitter(
                    brightness=float(cfg.get("brightness", 0.1)),
                    contrast=float(cfg.get("contrast", 0.1)),
                    saturation=float(cfg.get("saturation", 0.1)),
                    hue=float(cfg.get("hue", 0.05)),
                    p=0.5,
                ),
                A.GaussNoise(
                    std_range=(0.0, float(cfg.get("noise_std", 0.01))),
                    per_channel=True,
                    p=0.3,
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample_id, digit, quadrant = self.entries[index]
        modality = (
            random.choice(self.modalities)
            if self.train
            else self.modalities[index % len(self.modalities)]
        )
        image = self.images[
            self.modality_index[modality],
            self.key_index[self.key_format.format(sample_id)],
        ]
        if self.transform is None:
            normalized = torch.from_numpy(
                image.numpy().astype(np.float32) / 127.5 - 1.0
            ).contiguous()
        else:
            image = image.permute(1, 2, 0).contiguous().numpy()
            image = self.transform(image=image)["image"]
            normalized = (
                torch.from_numpy(image.astype(np.float32) / 127.5 - 1.0)
                .permute(2, 0, 1)
                .contiguous()
            )

        return {
            "img": normalized,
            "digit": torch.tensor(digit),
            "quadrant": torch.tensor(quadrant),
            "sample_id": torch.tensor(sample_id),
        }
