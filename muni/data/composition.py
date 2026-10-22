"""Dataset composition, pair-group sampling, and DataLoader factories."""

import torch
from torch.utils.data import ConcatDataset, Dataset

from muni.data.core import BaseSubdataset, SUBDATASET_REGISTRY
import muni.data.sources as _registered_data_sources  # noqa: F401
from muni.util.misc import print_on_main as print


class CompositeDataset(Dataset):
    """Align subdatasets by key and share augmentation state across modalities."""

    def __init__(
        self, subdatasets: dict[str, BaseSubdataset], flip_prob: float = 0.5,
        alignment: str = "strict",
        *, key_sets: dict[str, set[str]] | None = None,
    ):
        if not 0.0 <= float(flip_prob) <= 1.0:
            raise ValueError(f"flip_prob must lie in [0, 1], got {flip_prob}.")
        self.subdatasets = subdatasets
        self.flip_prob = float(flip_prob)

        if alignment not in {"strict", "intersection"}:
            raise ValueError(f"alignment must be 'strict' or 'intersection', got {alignment!r}.")
        if key_sets is None:
            key_sets = {name: set(sd.keys()) for name, sd in subdatasets.items()}
        common = set.intersection(*key_sets.values())
        dropped = {name: len(keys - common) for name, keys in key_sets.items()}
        if alignment == "strict" and any(dropped.values()):
            raise ValueError(
                f"Strict dataset alignment requires equal key sets; unmatched counts: {dropped}. "
                "Correct the sources or explicitly select alignment='intersection'."
            )
        self.common_keys = sorted(common)
        if not self.common_keys:
            raise ValueError(
                "No common keys across subdatasets. Check that key formats and data files are aligned."
            )
        print(
            f"[CompositeDataset] {len(self.common_keys)} samples, modalities: {list(subdatasets)}, "
            f"alignment={alignment}, dropped={dropped}"
        )

    def __len__(self) -> int:
        return len(self.common_keys)

    def __getitem__(self, index: int):
        key = self.common_keys[index]
        hflip = self.flip_prob > 0.0 and bool(torch.rand(1).item() < self.flip_prob)
        augment_state = {"hflip": hflip}

        dict_data = {
            name: sd.get(key, augment_state) for name, sd in self.subdatasets.items()
        }
        cache_key = f"{index}:{int(hflip)}"
        return dict_data, 0, cache_key


class PairGroupDataset(ConcatDataset):
    """Concatenate modality pair-groups into one index space."""

    def __init__(self, groups: dict[str, CompositeDataset]):
        if not groups:
            raise ValueError(
                "PairGroupDataset requires at least one enabled pair group."
            )
        self.groups = groups
        super().__init__(list(groups.values()))
        self.group_ranges = {
            name: range(start, end)
            for name, start, end in zip(
                groups, [0, *self.cumulative_sizes[:-1]], self.cumulative_sizes, strict=True
            )
        }
        print(
            f"[PairGroupDataset] {len(groups)} groups, {len(self)} total samples: "
            + ", ".join(f"{name}={len(indices)}" for name, indices in self.group_ranges.items())
        )



class PairGroupBatchSampler:
    """Yield same-group batches from disjoint DDP shards."""

    def __init__(
        self,
        dataset: PairGroupDataset,
        batch_size: int,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 0,
    ):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        if not 0 <= rank < num_replicas:
            raise ValueError(f"rank must lie in [0, {num_replicas}), got {rank}.")
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        self._group_indices: list[list[int]] = []
        self._batches_per_group: list[int] = []
        for gname, indices in dataset.group_ranges.items():
            full = list(indices)
            total = len(full)
            base = total // num_replicas
            remainder = total % num_replicas
            self._group_indices.append(full)
            if drop_last:
                if base < batch_size:
                    raise ValueError(
                        f"Training group {gname!r} has {total} samples, but needs at least "
                        f"{batch_size * num_replicas} for one full DDP batch "
                        f"(batch_size={batch_size}, num_replicas={num_replicas}). "
                        "Increase its size, reduce batch_size/num_replicas, or disable the group."
                    )
                # All ranks need the same group and optimizer-step schedule.
                self._batches_per_group.append(base // batch_size)
            else:
                per_rank = base + (1 if rank < remainder else 0)
                self._batches_per_group.append(
                    (per_rank + batch_size - 1) // batch_size
                )

        self._total_batches = sum(self._batches_per_group)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self) -> int:
        return self._total_batches

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        all_batches = []
        for group_index, full_indices in enumerate(self._group_indices):
            if self.shuffle:
                perm = torch.randperm(len(full_indices), generator=g).tolist()
                indices = [full_indices[i] for i in perm]
            else:
                indices = full_indices
            # Match DistributedSampler's remainder-aware disjoint partition.
            total = len(indices)
            base = total // self.num_replicas
            remainder = total % self.num_replicas
            start = self.rank * base + min(self.rank, remainder)
            end = start + base + (1 if self.rank < remainder else 0)
            indices = indices[start:end]
            if self.drop_last:
                shared_samples = self._batches_per_group[group_index] * self.batch_size
                indices = indices[:shared_samples]
            for b_start in range(0, len(indices), self.batch_size):
                batch = indices[b_start : b_start + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                if batch:
                    all_batches.append(batch)

        if self.shuffle:
            order = torch.randperm(len(all_batches), generator=g).tolist()
            all_batches = [all_batches[i] for i in order]

        yield from all_batches


def _apply_stem_range(
    keys: list[str],
    stem_range: tuple[int, int] | list[int] | None,
) -> list[str]:
    """Filter numeric stems to ``[lo, hi)``; reject opaque keys."""
    if stem_range is None:
        return list(keys)
    lo, hi = map(int, stem_range)
    if lo > hi:
        raise ValueError(
            f"stem_range lower bound must not exceed upper bound, got [{lo}, {hi}]."
        )
    out: list[str] = []
    for k in keys:
        s = k
        if "_" in k:
            head, tail = k.split("_", 1)
            if head.isdigit():
                s = tail
        if s.isdigit():
            n = int(s)
            if lo <= n < hi:
                out.append(k)
        else:
            raise ValueError(
                f"stem_range requires numeric stems (optionally source-prefixed); got {k!r}."
            )
    return out


def _build_composite(
    subdatasets_cfg,
    flip_prob: float = 0.0,
    *,
    stem_range: tuple[int, int] | list[int] | None = None,
    alignment: str = "strict",
) -> CompositeDataset:
    subdatasets: dict[str, BaseSubdataset] = {}
    key_sets: dict[str, set[str]] | None = {} if stem_range is not None else None
    for name, sd_cfg in subdatasets_cfg.items():
        # OmegaConf inheritance cannot delete a key; null disables it.
        if sd_cfg is None:
            continue
        cls = SUBDATASET_REGISTRY[sd_cfg.type]
        inner = cls(**sd_cfg.get("params", {}))
        subdatasets[name] = inner
        if key_sets is not None:
            key_sets[name] = set(_apply_stem_range(inner.keys(), stem_range))
    return CompositeDataset(
        subdatasets, flip_prob=flip_prob, alignment=alignment, key_sets=key_sets
    )


def build_dataset(
    dataset_cfg,
    *,
    flip_prob: float | None = None,
) -> PairGroupDataset:
    """Build pair-group data, including the single-group config form."""
    configured_flip_prob = dataset_cfg.get("flip_prob", 0.5)
    effective_flip_prob = flip_prob if flip_prob is not None else configured_flip_prob
    if effective_flip_prob is None:
        effective_flip_prob = 0.5
    effective_flip_prob = float(effective_flip_prob)
    dataset_stem_range = dataset_cfg.get("stem_range", None)
    dataset_alignment = dataset_cfg.get("alignment", "strict")

    group_configs = (
        dataset_cfg.pair_groups if "pair_groups" in dataset_cfg else {"main": dataset_cfg}
    )
    groups: dict[str, CompositeDataset] = {}
    for group_name, group_cfg in group_configs.items():
        # OmegaConf inheritance cannot delete a key; null disables it.
        if group_cfg is None:
            continue
        group_stem_range = group_cfg.get("stem_range")
        group_alignment = group_cfg.get("alignment")
        groups[group_name] = _build_composite(
            group_cfg.subdatasets,
            effective_flip_prob,
            stem_range=dataset_stem_range if group_stem_range is None else group_stem_range,
            alignment=dataset_alignment if group_alignment is None else group_alignment,
        )
    return PairGroupDataset(groups)


def build_train_loader(
    dataset: PairGroupDataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    num_replicas: int,
    rank: int,
    seed: int,
) -> tuple[torch.utils.data.DataLoader, PairGroupBatchSampler]:
    sampler = PairGroupBatchSampler(
        dataset,
        batch_size=batch_size,
        num_replicas=num_replicas,
        rank=rank,
        shuffle=True,
        drop_last=True,
        seed=seed,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    return loader, sampler


def build_eval_loader(
    dataset: PairGroupDataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    num_replicas: int,
    rank: int,
) -> torch.utils.data.DataLoader:
    sampler = PairGroupBatchSampler(
        dataset,
        batch_size=batch_size,
        num_replicas=num_replicas,
        rank=rank,
        shuffle=False,
        drop_last=False,
        seed=0,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=2 if num_workers > 0 else None,
    )
