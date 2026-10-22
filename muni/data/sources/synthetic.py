"""Synthetic Gaussian-mixture data source for toy experiments."""

import torch

from muni.data.core import BaseSubdataset, register_subdataset

@register_subdataset
class GaussianMixtureSubdataset(BaseSubdataset):
    """A modality-specific view of a shared Venn-structured Gaussian mixture.

    Each non-empty modality subset owns one independent mixture region. A
    modality observes the regions containing its index. Pools are materialized
    once per parameter set so all modality instances share the same world.
    """

    registry_name = "gaussian_mixture"

    _POOL_CACHE: dict[tuple, torch.Tensor] = {}
    _PROJECTION_CACHE: dict[tuple, torch.Tensor] = {}

    def __init__(
        self,
        n_modalities: int = 3,
        region_dim: int = 2,
        K: int = 16,
        n_samples: int = 100_000,
        modality_idx: int = 0,
        mixture_seed: int = 42,
        stem_pad: int = 8,
        region_dim_by_size: dict | None = None,
        observation_noise_std: float = 0.0,
        observation_noise_seed: int = 314159,
    ):
        import itertools

        if n_modalities < 2:
            raise ValueError("n_modalities must be >= 2")
        if not (0 <= modality_idx < n_modalities):
            raise ValueError(f"modality_idx={modality_idx} not in [0, {n_modalities})")
        if K < 1 or n_samples < 1:
            raise ValueError("K/n_samples must be positive")

        self.n_modalities = int(n_modalities)
        self.region_dim = int(region_dim)
        self.K = int(K)
        self.n_samples = int(n_samples)
        self.modality_idx = int(modality_idx)
        self.mixture_seed = int(mixture_seed)
        self.stem_pad = int(stem_pad)
        self.observation_noise_std = float(observation_noise_std)
        self.observation_noise_seed = int(observation_noise_seed)
        if self.observation_noise_std < 0.0:
            raise ValueError("observation_noise_std must be non-negative")
        if region_dim_by_size is not None:
            self.region_dim_by_size = {int(k): int(v) for k, v in dict(region_dim_by_size).items()}
        else:
            self.region_dim_by_size = {}

        # PairwiseSharedDimsMSE relies on this historical region order.
        subsets = [
            combo
            for size in range(1, self.n_modalities + 1)
            for combo in itertools.combinations(range(self.n_modalities), size)
        ]
        subsets.sort(key=lambda s: (-len(s), list(s)))
        self._region_subsets: tuple[tuple[int, ...], ...] = tuple(subsets)

        self._region_widths: tuple[int, ...] = tuple(
            self.region_dim_by_size.get(len(s), self.region_dim) for s in self._region_subsets
        )
        if any(w < 1 for w in self._region_widths):
            raise ValueError(f"all region widths must be >= 1, got {self._region_widths}")
        self._region_offsets = (0, *itertools.accumulate(self._region_widths))
        self._hidden_dim = self._region_offsets[-1]

        mask = [self.modality_idx in s for s in self._region_subsets]
        self._region_mask = mask
        self._modality_dim = sum(w for w, m in zip(self._region_widths, mask) if m)

        cache_key = (
            self.n_modalities,
            self.region_dim,
            self._region_widths,
            self.K,
            self.n_samples,
            self.mixture_seed,
        )
        cls = type(self)
        if cache_key not in cls._POOL_CACHE:
            cls._POOL_CACHE[cache_key] = self._materialize_pool()
        self._h_pool: torch.Tensor = cls._POOL_CACHE[cache_key]

        projection_key = (
            cache_key, self.modality_idx,
            self.observation_noise_std, self.observation_noise_seed,
        )
        if projection_key not in cls._PROJECTION_CACHE:
            slice_dims: list[int] = []
            for r, in_mod in enumerate(self._region_mask):
                if in_mod:
                    slice_dims.extend(range(self._region_offsets[r], self._region_offsets[r + 1]))
            projection = self._h_pool[:, slice_dims].contiguous()
            if self.observation_noise_std > 0.0:
                # Fixed modality-specific streams preserve deterministic access.
                import numpy as np

                noise_seed = (self.observation_noise_seed + 104729 * self.modality_idx) % (2**32)
                noise_rng = np.random.RandomState(noise_seed)
                noise = noise_rng.standard_normal(size=projection.shape).astype(np.float32)
                projection.add_(torch.from_numpy(noise), alpha=self.observation_noise_std)
            cls._PROJECTION_CACHE[projection_key] = projection
        self._mod_full = cls._PROJECTION_CACHE[projection_key]

        self._keys = [str(i).zfill(self.stem_pad) for i in range(self.n_samples)]

    def _materialize_pool(self) -> torch.Tensor:
        """Generate the pool with the legacy RandomState draw order."""
        import numpy as np

        rng = np.random.RandomState(self.mixture_seed)
        means: list[np.ndarray] = []
        stds: list[np.ndarray] = []
        for w in self._region_widths:
            means.append(rng.randn(self.K, w).astype(np.float32) * 3.0)
            stds.append(rng.uniform(0.5, 1.5, (self.K, w)).astype(np.float32))
        N = self.n_samples
        h = np.empty((N, self._hidden_dim), dtype=np.float32)
        for r, w in enumerate(self._region_widths):
            comp = rng.randint(0, self.K, size=N)
            mu = means[r][comp]
            sd = stds[r][comp]
            noise = rng.standard_normal(size=(N, w)).astype(np.float32)
            h[:, self._region_offsets[r] : self._region_offsets[r + 1]] = mu + sd * noise
        return torch.from_numpy(h)

    @property
    def modality_dim(self) -> int:
        return self._modality_dim

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def region_widths(self) -> tuple[int, ...]:
        return self._region_widths

    def keys(self) -> list[str]:
        return self._keys

    def get(self, key: str, augment_state: dict | None = None) -> torch.Tensor:
        return self._mod_full[int(key)]

    def get_batch(
        self,
        keys: list[str],
        augment_state: dict | None = None,
    ) -> torch.Tensor:
        """Fetch rows with one tensor index operation."""
        idx = torch.as_tensor([int(k) for k in keys], dtype=torch.long)
        return self._mod_full[idx]
