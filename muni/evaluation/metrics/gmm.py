"""Metrics for vector modalities and the Gaussian-mixture benchmark."""

import itertools
from collections.abc import Sequence

import numpy as np
import torch

from muni.evaluation.metrics.base import BaseMetric
from muni.evaluation.metrics.utils import squared_error_sum_and_count


# Vector-modality metrics
# These metrics operate on flat feature tensors of shape ``(B, D)``.  They are
# domain-agnostic — used by the Gaussian-mixture toy experiment but also
# applicable to any vector modality (e.g. pre-extracted features).


def _venn_shared_dimensions(
    modality_names: Sequence[str],
    n_modalities: int,
    region_dim: int,
    region_dim_by_size: dict | None,
) -> dict[tuple[str, str], list[int]]:
    """Map ``(source, target)`` to target-local shared Venn dimensions."""
    subsets = [
        subset
        for size in range(1, n_modalities + 1)
        for subset in itertools.combinations(range(n_modalities), size)
    ]
    subsets.sort(key=lambda subset: (-len(subset), list(subset)))
    widths_by_size = (
        {int(key): int(value) for key, value in dict(region_dim_by_size).items()}
        if region_dim_by_size is not None
        else {}
    )
    widths = [widths_by_size.get(len(subset), region_dim) for subset in subsets]

    shared: dict[tuple[str, str], list[int]] = {}
    for source_index, source in enumerate(modality_names):
        for target_index, target in enumerate(modality_names):
            local_indices: list[int] = []
            target_offset = 0
            for subset, width in zip(subsets, widths):
                if source_index in subset and target_index in subset:
                    local_indices.extend(range(target_offset, target_offset + width))
                if target_index in subset:
                    target_offset += width
            shared[(source, target)] = local_indices
    return shared


class AlignmentMSE(BaseMetric):
    """Paired MSE, optionally restricted to source-recoverable Venn dimensions."""

    def __init__(
        self,
        shared_dims_only: bool = False,
        n_modalities: int = 3,
        region_dim: int = 2,
        modality_names: Sequence[str] | None = None,
        name: str = "AlignMSE",
        region_dim_by_size: dict | None = None,
        **kwargs,
    ):
        self.sse = 0.0
        self.count = 0
        self.shared_dims_only = bool(shared_dims_only)
        self.name = str(name)
        self._shared_in_tgt: dict[tuple[str, str], list[int]] = {}

        if not self.shared_dims_only:
            return

        names = (
            list(modality_names)
            if modality_names is not None
            else [chr(ord("A") + i) for i in range(n_modalities)]
        )
        if len(names) != n_modalities:
            raise ValueError(
                f"modality_names length {len(names)} != n_modalities {n_modalities}"
            )
        self.modality_names = names
        self.n_modalities = int(n_modalities)
        self.region_dim = int(region_dim)
        self.region_dim_by_size = dict(region_dim_by_size or {})
        self._shared_in_tgt = _venn_shared_dimensions(
            names,
            self.n_modalities,
            self.region_dim,
            self.region_dim_by_size,
        )

    def _select_shared(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        src_modalities: list[str],
        tgt_modality: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select the union of target dimensions shared with any source."""
        idx_set: set[int] = set()
        for s in src_modalities:
            idx_set.update(self._shared_in_tgt.get((s, tgt_modality), []))
        idx = sorted(idx_set)
        if not idx:
            return preds, targets
        return preds[:, idx], targets[:, idx]

    def update(
        self, preds: torch.Tensor, targets: torch.Tensor | None = None, **kwargs
    ) -> None:
        if targets is None:
            return
        if self.shared_dims_only:
            srcs = kwargs.get("src_modalities") or [kwargs.get("src_modality")]
            srcs = [s for s in srcs if s]
            tgt = kwargs.get("tgt_modality")
            if srcs and tgt:
                preds, targets = self._select_shared(preds, targets, srcs, tgt)
        squared_error, count = squared_error_sum_and_count(preds, targets)
        self.sse += squared_error
        self.count += count

    def compute(self):
        if self.count == 0:
            self.sse = 0.0
            return {self.name: float("nan")}
        out = self.sse / float(self.count)
        self.sse = 0.0
        self.count = 0
        return {self.name: out}


class SlicedWassersteinMetric(BaseMetric):
    """Sliced W2 against an NPZ reference marginal.

    Both pools are deterministically subsampled to equal size before sorting
    projected samples, avoiding interpolation between unequal empirical CDFs.
    """

    def __init__(
        self,
        reference_file: str,
        num_projections: int = 128,
        seed: int = 0,
        name: str = "SWD",
        **kwargs,
    ):
        self.reference_file = reference_file
        self.num_projections = int(num_projections)
        self.seed = int(seed)
        self.name = str(name)
        self._preds: list[torch.Tensor] = []
        # Defer reference load to compute() so construction is cheap and
        # workers/replicas don't all hit disk simultaneously.
        self._refs: torch.Tensor | None = None

    def _load_refs(self) -> torch.Tensor:
        if self._refs is None:
            with np.load(self.reference_file) as data:
                arr = (
                    data["samples"] if "samples" in data.files else data[data.files[0]]
                )
                self._refs = torch.from_numpy(np.asarray(arr, dtype=np.float32))
            if self._refs.ndim != 2:
                raise ValueError(
                    f"SlicedWassersteinMetric: expected (M, D) reference, got "
                    f"shape {tuple(self._refs.shape)} from {self.reference_file}"
                )
        return self._refs

    def update(
        self, preds: torch.Tensor, targets: torch.Tensor | None = None, **kwargs
    ) -> None:
        if preds.numel() == 0:
            return
        if preds.ndim != 2:
            preds = preds.reshape(preds.shape[0], -1)
        self._preds.append(preds.detach().float().cpu())

    def compute(self):
        if not self._preds:
            return {self.name: float("nan")}
        all_preds = torch.cat(self._preds, dim=0)
        self._preds = []

        refs = self._load_refs()

        n = min(all_preds.shape[0], refs.shape[0])
        if n == 0:
            return {self.name: float("nan")}

        gen = torch.Generator().manual_seed(self.seed)
        # Deterministic subsample of the larger pool to common size n.
        if all_preds.shape[0] > n:
            idx = torch.randperm(all_preds.shape[0], generator=gen)[:n]
            all_preds = all_preds[idx]
        if refs.shape[0] > n:
            idx = torch.randperm(refs.shape[0], generator=gen)[:n]
            refs = refs[idx]

        d = all_preds.shape[1]
        theta = torch.randn(self.num_projections, d, generator=gen)
        theta = theta / theta.norm(dim=1, keepdim=True).clamp_min(1e-12)

        proj_p = all_preds @ theta.T  # (n, P)
        proj_r = refs @ theta.T  # (n, P)
        sorted_p, _ = torch.sort(proj_p, dim=0)
        sorted_r, _ = torch.sort(proj_r, dim=0)
        per_dir_w2 = torch.sqrt(((sorted_p - sorted_r) ** 2).mean(dim=0).clamp_min(0))
        return {self.name: float(per_dir_w2.mean().item())}


class PairwiseSharedDimsMSE(BaseMetric):
    """MSE between corresponding Venn-shared dimensions of generated modalities."""

    def __init__(
        self,
        n_modalities: int = 3,
        region_dim: int = 2,
        modality_names: Sequence[str] | None = None,
        region_dim_by_size: dict | None = None,
        **kwargs,
    ):
        if n_modalities < 2:
            raise ValueError("n_modalities must be >= 2")
        names = (
            list(modality_names)
            if modality_names is not None
            else [chr(ord("A") + i) for i in range(n_modalities)]
        )
        if len(names) != n_modalities:
            raise ValueError(
                f"modality_names length {len(names)} != n_modalities {n_modalities}"
            )
        self.n_modalities = int(n_modalities)
        self.region_dim = int(region_dim)
        self.modality_names = names
        self.region_dim_by_size = dict(region_dim_by_size or {})
        shared = _venn_shared_dimensions(
            names,
            self.n_modalities,
            self.region_dim,
            self.region_dim_by_size,
        )
        self._pair_indices = {
            (names[i], names[j]): (
                shared[(names[j], names[i])],
                shared[(names[i], names[j])],
            )
            for i in range(self.n_modalities)
            for j in range(i + 1, self.n_modalities)
        }

        self._sse: dict[tuple[str, str], float] = {
            pair: 0.0 for pair in self._pair_indices
        }
        self._count: dict[tuple[str, str], int] = {
            pair: 0 for pair in self._pair_indices
        }

    def update(
        self,
        preds: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        sources = kwargs.get("sources")
        if not isinstance(sources, dict):
            return
        for (mi, mj), (idx_i, idx_j) in self._pair_indices.items():
            if mi not in sources or mj not in sources:
                continue
            xi = sources[mi]
            xj = sources[mj]
            if not isinstance(xi, torch.Tensor) or not isinstance(xj, torch.Tensor):
                continue
            if xi.numel() == 0 or xj.numel() == 0:
                continue
            n = min(int(xi.shape[0]), int(xj.shape[0]))
            if n == 0:
                continue
            si = xi[:n].detach().float()[:, idx_i]
            sj = xj[:n].detach().float()[:, idx_j]
            squared_error, count = squared_error_sum_and_count(si, sj)
            self._sse[(mi, mj)] += squared_error
            self._count[(mi, mj)] += count

    def compute(self):
        out: dict[str, float] = {}
        per_pair: list[float] = []
        for mi, mj in self._pair_indices:
            c = self._count[(mi, mj)]
            v = self._sse[(mi, mj)] / float(c) if c > 0 else float("nan")
            out[f"PairMSE/{mi}_{mj}"] = v
            if c > 0:
                per_pair.append(v)
            self._sse[(mi, mj)] = 0.0
            self._count[(mi, mj)] = 0
        out["PairMSE/avg"] = (
            float(sum(per_pair) / len(per_pair)) if per_pair else float("nan")
        )
        return out
