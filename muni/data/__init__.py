"""Public construction API for MUNI datasets and loaders."""

from muni.data.composition import (
    CompositeDataset,
    PairGroupBatchSampler,
    PairGroupDataset,
    build_dataset,
    build_eval_loader,
    build_train_loader,
)
from muni.data.core import BaseSubdataset, SUBDATASET_REGISTRY, register_subdataset
from muni.training.batching import resolve_eval_batch_size


def build_configured_eval_loader(cfg, *, num_replicas: int, rank: int):
    """Evaluate dataset_eval when configured, otherwise the training dataset."""
    dataset_cfg = cfg.get("dataset_eval", None)
    if dataset_cfg is None:
        dataset_cfg = cfg.dataset
    return build_eval_loader(
        build_dataset(dataset_cfg),
        batch_size=resolve_eval_batch_size(cfg.train, cfg.eval, num_replicas),
        num_workers=int(cfg.eval.get("num_workers", cfg.train.num_workers)),
        pin_memory=bool(cfg.eval.get("pin_mem", cfg.train.pin_mem)),
        num_replicas=num_replicas,
        rank=rank,
    )


__all__ = [
    "BaseSubdataset",
    "CompositeDataset",
    "PairGroupBatchSampler",
    "PairGroupDataset",
    "SUBDATASET_REGISTRY",
    "build_dataset",
    "build_configured_eval_loader",
    "build_eval_loader",
    "build_train_loader",
    "register_subdataset",
]
