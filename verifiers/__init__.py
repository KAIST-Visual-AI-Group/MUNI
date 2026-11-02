"""Training and inference utilities for visual verifiers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.utils.data import Dataset


SUPPORTED_DATASETS = ("polymnist",)


def build_verifier_dataset(cfg: dict, split: str = "train") -> Dataset:
    dataset_type = cfg.get("data", {}).get("type", "polymnist")
    if dataset_type == "polymnist":
        from .polymnist_data import PolyMNISTVerifierDataset

        return PolyMNISTVerifierDataset(cfg, split)
    raise ValueError(
        f"unknown data.type {dataset_type!r}; expected one of {SUPPORTED_DATASETS}"
    )


__all__ = ["SUPPORTED_DATASETS", "build_verifier_dataset"]
