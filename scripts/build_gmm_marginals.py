"""Build deterministic Gaussian-mixture marginals for GMM SWD evaluation.

Set ``--start-key`` beyond the training key range to keep references disjoint.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

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
    p.add_argument("--n-modalities", type=int, default=3)
    p.add_argument("--region-dim", type=int, default=10)
    p.add_argument("--K", type=int, default=64)
    p.add_argument("--mixture-seed", type=int, default=42)
    p.add_argument("--num-samples", type=int, default=50_000)
    p.add_argument("--start-key", type=int, default=1_000_000)
    p.add_argument("--stem-pad", type=int, default=8)
    p.add_argument("--observation-noise-std", type=float, default=0.0)
    p.add_argument("--observation-noise-seed", type=int, default=314159)
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--region-dim-by-size",
        type=str,
        default='{"1":20,"2":10,"3":10}',
        help='JSON dict mapping subset size to dim, e.g. \'{"1":20,"2":10,"3":10}\'',
    )
    p.add_argument(
        "--modality-names",
        nargs="*",
        default=None,
        help="Optional modality labels; default ABC...",
    )
    args = p.parse_args()
    region_dim_by_size = None
    if args.region_dim_by_size:
        import json

        region_dim_by_size = {
            int(k): int(v) for k, v in json.loads(args.region_dim_by_size).items()
        }

    names = (
        args.modality_names
        if args.modality_names is not None
        else [chr(ord("A") + i) for i in range(args.n_modalities)]
    )
    if len(names) != args.n_modalities:
        raise SystemExit(
            f"--modality-names length {len(names)} != --n-modalities {args.n_modalities}"
        )

    os.makedirs(args.out_dir, exist_ok=True)
    print(
        f"[marginals] n={args.n_modalities} d={args.region_dim} K={args.K} "
        f"seed={args.mixture_seed} N={args.num_samples} start={args.start_key} "
        f"sigma_obs={args.observation_noise_std}"
    )
    for idx, name in enumerate(names):
        samples = _build_modality_samples(
            n_modalities=args.n_modalities,
            region_dim=args.region_dim,
            K=args.K,
            mixture_seed=args.mixture_seed,
            num_samples=args.num_samples,
            start_key=args.start_key,
            modality_idx=idx,
            stem_pad=args.stem_pad,
            region_dim_by_size=region_dim_by_size,
            observation_noise_std=args.observation_noise_std,
            observation_noise_seed=args.observation_noise_seed,
        )
        out_path = os.path.join(args.out_dir, f"marginal_{name}.npz")
        np.savez(out_path, samples=samples)
        print(
            f"  → {out_path}  shape={samples.shape}  mean={samples.mean():+.3f} std={samples.std():.3f}"
        )


if __name__ == "__main__":
    main()
