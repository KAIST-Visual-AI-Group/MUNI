"""Build deterministic Gaussian-mixture marginals for GMM SWD evaluation.

Geometry, seed, and output directory follow the training config.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from omegaconf.errors import InterpolationResolutionError

from muni.config_utils import MissingConfigEnvironmentError, load_config
from muni.data.sources.image_cache import create_temporary_file, publish_file
from muni.data.sources.synthetic import GaussianMixtureSubdataset


def _build_modality_samples(
    n_modalities: int,
    region_dim: int,
    K: int,
    mixture_seed: int,
    num_samples: int,
    start_key: int,
    modality_idx: int,
    stem_pad: int,
    region_dim_by_size: dict | None = None,
    observation_noise_std: float = 0.0,
    observation_noise_seed: int = 314159,
) -> np.ndarray:
    sd = GaussianMixtureSubdataset(
        n_modalities=n_modalities,
        region_dim=region_dim,
        K=K,
        n_samples=start_key + num_samples,
        modality_idx=modality_idx,
        mixture_seed=mixture_seed,
        stem_pad=stem_pad,
        region_dim_by_size=region_dim_by_size,
        observation_noise_std=observation_noise_std,
        observation_noise_seed=observation_noise_seed,
    )
    keys = sd.keys()[start_key : start_key + num_samples]
    return sd.get_batch(keys).numpy().astype(np.float32, copy=False)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/gmm/muni.yaml")
    p.add_argument("--num-samples", type=int, default=50_000)
    p.add_argument("--start-key", type=int, default=None, help="Defaults to toy.n_train.")
    p.add_argument("--stem-pad", type=int, default=8)
    p.add_argument("--out-dir", default=None, help="Defaults to toy.marginal_dir.")
    args = p.parse_args()
    try:
        cfg = load_config(args.config, resolve=False)
        toy = cfg.toy
        region_dim_by_size = (
            OmegaConf.to_container(toy.region_dim_by_size, resolve=True)
            if toy.get("region_dim_by_size") is not None else None
        )
        names = list(toy.modalities)
        if len(names) != toy.n_modalities:
            p.error("toy.modalities must contain one name for each toy.n_modalities entry")
        start_key = toy.n_train if args.start_key is None else args.start_key
        out_dir = Path(args.out_dir or str(toy.marginal_dir)).expanduser()
    except (MissingConfigEnvironmentError, InterpolationResolutionError) as exc:
        p.error(str(exc))

    output_paths = [out_dir / f"marginal_{name}.npz" for name in names]
    for path in output_paths:
        if path.exists():
            p.error(f"Reference already exists: {path}. Choose a fresh --out-dir.")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[marginals] n={toy.n_modalities} d={toy.region_dim} K={toy.K} "
        f"seed={toy.mixture_seed} N={args.num_samples} start={start_key} "
        f"sigma_obs={toy.observation_noise_std}"
    )
    for idx, out_path in enumerate(output_paths):
        samples = _build_modality_samples(
            n_modalities=toy.n_modalities,
            region_dim=toy.region_dim,
            K=toy.K,
            mixture_seed=toy.mixture_seed,
            num_samples=args.num_samples,
            start_key=start_key,
            modality_idx=idx,
            stem_pad=args.stem_pad,
            region_dim_by_size=region_dim_by_size,
            observation_noise_std=toy.observation_noise_std,
            observation_noise_seed=toy.observation_noise_seed,
        )
        temporary = create_temporary_file(out_path)
        try:
            with temporary.open("wb") as handle:
                np.savez(handle, samples=samples)
            publish_file(temporary, out_path, overwrite=False)
        finally:
            temporary.unlink(missing_ok=True)
        print(
            f"  → {out_path}  shape={samples.shape}  mean={samples.mean():+.3f} std={samples.std():.3f}"
        )


if __name__ == "__main__":
    main()
