"""Build the PolyMNIST-Quadrant-Labels cache directly, without PNG files."""

from __future__ import annotations

import argparse
import csv
import math
import fcntl
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import datasets
from tqdm import tqdm


from muni.data.sources.image_cache import (
    create_temporary_file,
    publish_file,
    save_image_tensor_cache,
    sha256_file,
)


CANVAS_SIZE = 64
DIGIT_SIZE = 32
THRESHOLD = 128
NUM_QUADRANTS = 4
RNG_STREAM = {
    "digit_pool": 20_003,
    "background_crop": 40_009,
    "synthetic_background": 50_021,
}
QUADRANT_OFFSETS = {
    0: (0, 0),
    1: (32, 0),
    2: (0, 32),
    3: (32, 32),
}
ORIGINAL_BACKGROUNDS = [
    (
        "0_fractal_tree.jpg",
        "https://people.sc.fsu.edu/~jburkardt/data/jpg/fractal_tree.jpg",
        "c3aea824f3a92ab5692e480d02fe868bafa338de3ed9993d540274374a3c31b8",
    ),
    (
        "1_the_scream.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/f/f4/The_Scream.jpg",
        "a9bd171fe0a8da87f90a467e30542be195fb2e2a46d650f5287421869e57d390",
    ),
    (
        "2_lena3.tif",
        "https://links.uwaterloo.ca/Repository/TIF/lena3.tif",
        "ecabbfc5c57cac78f90cf150b1491ae1a8b96a9dec5083a3a5cd3ba500edd605",
    ),
    (
        "3_star_field.jpg",
        "https://people.sc.fsu.edu/~jburkardt/data/jpg/star_field.jpg",
        "22317171efc167c6a5b4a783d44289f8f9b3c987e1e363acddabc81d2227c27c",
    ),
    (
        "4_shingles.jpg",
        "https://people.sc.fsu.edu/~jburkardt/data/jpg/shingles.jpg",
        "22e05161aa0b9963cd0c2089f17e06807e163b36696b70b305c130396c5c448e",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Persistent source root for MNIST and original background images.",
    )
    parser.add_argument("--mnist-root", type=Path, default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--num-samples", type=int, default=230000)
    parser.add_argument("--num-modalities", type=int, default=5)
    parser.add_argument("--cache-modalities", nargs="+", default=["m0", "m1", "m2"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--background-dir", type=Path, default=None)
    parser.add_argument(
        "--background-preset",
        choices=["none", "original-polymnist"],
        default="original-polymnist",
    )
    parser.add_argument("--synthetic-background-size", type=int, default=512)
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def rng_for(seed: int, stream: str, *keys: int) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence([seed, RNG_STREAM[stream], *keys])
    )


def download_file(url: str, path: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        response = urllib.request.urlopen(request, timeout=60)
    except urllib.error.HTTPError as exc:
        if exc.code not in {301, 302, 303, 307, 308} or "Location" not in exc.headers:
            raise
        request = urllib.request.Request(
            exc.headers["Location"], headers={"User-Agent": "Mozilla/5.0"}
        )
        response = urllib.request.urlopen(request, timeout=60)
    with response:
        path.write_bytes(response.read())


def ensure_original_backgrounds(source_dir: Path) -> Path:
    background_dir = source_dir / "original_polymnist_backgrounds"
    background_dir.mkdir(parents=True, exist_ok=True)
    for filename, url, expected_hash in ORIGINAL_BACKGROUNDS:
        path = background_dir / filename
        if not path.exists() or path.stat().st_size == 0:
            print(f"Downloading {url} -> {path}")
            download_file(url, path)
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Background hash mismatch for {path}: expected {expected_hash}, got {actual_hash}"
            )
    return background_dir


def make_synthetic_background(size: int, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    xx /= max(size - 1, 1)
    yy /= max(size - 1, 1)
    base_color = rng.uniform(0.15, 0.85, size=(3,))
    accent = rng.uniform(0.0, 1.0, size=(3,))
    phase = rng.uniform(0.0, 2.0 * math.pi, size=(3,))
    freq = rng.uniform(2.0, 7.0, size=(3,))
    texture = np.zeros((size, size, 3), dtype=np.float32)
    for channel in range(3):
        waves = 0.22 * np.sin(
            (xx * freq[channel] + yy * (freq[channel] * 0.7)) * 2.0 * math.pi
            + phase[channel]
        ) + 0.14 * np.cos(
            (yy * (freq[channel] + 1.3)) * 2.0 * math.pi + phase[channel] * 0.5
        )
        texture[..., channel] = (
            base_color[channel] + waves + 0.25 * accent[channel] * xx
        )
    noise_small = rng.normal(0.0, 0.09, size=(size // 8 + 1, size // 8 + 1, 3))
    noise = np.repeat(np.repeat(noise_small, 8, axis=0), 8, axis=1)[:size, :size]
    return Image.fromarray(
        (np.clip(texture + noise, 0.0, 1.0) * 255.0).round().astype(np.uint8),
        mode="RGB",
    )


def load_backgrounds(args: argparse.Namespace) -> list[Image.Image]:
    background_dir = args.background_dir
    if background_dir is None and args.background_preset == "original-polymnist":
        background_dir = ensure_original_backgrounds(args.source_dir)
    if background_dir is not None:
        suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        files = sorted(
            path for path in background_dir.iterdir() if path.suffix.lower() in suffixes
        )
        if len(files) < args.num_modalities:
            raise ValueError(
                f"Need {args.num_modalities} backgrounds in {background_dir}, found {len(files)}"
            )
        return [
            Image.open(path).convert("RGB") for path in files[: args.num_modalities]
        ]
    return [
        make_synthetic_background(
            args.synthetic_background_size,
            int(
                rng_for(args.seed, "synthetic_background", modality).integers(0, 2**32)
            ),
        )
        for modality in range(args.num_modalities)
    ]


def load_mnist(root: Path, download: bool) -> tuple[torch.Tensor, torch.Tensor]:
    parts = [
        datasets.MNIST(str(root), train=train, download=download)
        for train in (True, False)
    ]
    return torch.cat([part.data for part in parts]), torch.cat(
        [part.targets.long() for part in parts]
    )


class DigitPool:
    def __init__(self, indices: np.ndarray, seed: int, digit: int, modality: int):
        self.indices = indices
        self.seed = seed
        self.digit = digit
        self.modality = modality
        self.cycle = 0
        self.position = 0
        self.order = self._shuffle()

    def _shuffle(self) -> np.ndarray:
        order = self.indices.copy()
        rng_for(self.seed, "digit_pool", self.digit, self.modality, self.cycle).shuffle(
            order
        )
        return order

    def next(self) -> int:
        if self.position == len(self.order):
            self.cycle += 1
            self.position = 0
            self.order = self._shuffle()
        index = int(self.order[self.position])
        self.position += 1
        return index


def build_digit_pools(
    labels: torch.Tensor, num_modalities: int, seed: int
) -> dict[int, list[DigitPool]]:
    by_digit = {
        digit: np.flatnonzero(labels.numpy() == digit).astype(np.int64)
        for digit in range(10)
    }
    return {
        digit: [
            DigitPool(by_digit[digit], seed, digit, modality)
            for modality in range(num_modalities)
        ]
        for digit in range(10)
    }


def draw_distinct(pools: list[DigitPool]) -> list[int]:
    selected: list[int] = []
    used: set[int] = set()
    for pool in pools:
        index = pool.next()
        attempts = 1
        while index in used and attempts <= len(pool.indices):
            index = pool.next()
            attempts += 1
        selected.append(index)
        used.add(index)
    return selected


def crop_background(
    background: Image.Image, seed: int, sample: int, modality: int
) -> np.ndarray:
    rng = rng_for(seed, "background_crop", sample, modality)
    if background.width < CANVAS_SIZE or background.height < CANVAS_SIZE:
        scale = max(CANVAS_SIZE / background.width, CANVAS_SIZE / background.height)
        background = background.resize(
            (math.ceil(background.width * scale), math.ceil(background.height * scale)),
            Image.Resampling.BICUBIC,
        )
    max_x = background.width - CANVAS_SIZE
    max_y = background.height - CANVAS_SIZE
    x = int(rng.integers(0, max_x + 1)) if max_x else 0
    y = int(rng.integers(0, max_y + 1)) if max_y else 0
    return np.asarray(
        background.crop((x, y, x + CANVAS_SIZE, y + CANVAS_SIZE)), dtype=np.uint8
    )


def digit_mask(image: torch.Tensor) -> np.ndarray:
    source = Image.fromarray(image.numpy().astype(np.uint8), mode="L")
    resized = source.resize((DIGIT_SIZE, DIGIT_SIZE), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.uint8) > THRESHOLD


def compose_image(
    background: np.ndarray, mask: np.ndarray, quadrant: int
) -> torch.Tensor:
    result = background.copy()
    x0, y0 = QUADRANT_OFFSETS[quadrant]
    patch = result[y0 : y0 + DIGIT_SIZE, x0 : x0 + DIGIT_SIZE]
    patch[mask] = 255 - patch[mask]
    return torch.from_numpy(result).permute(2, 0, 1)


def build_cache(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Keep the inode: unlinking a lock file allows two builders to own different locks.
    with (args.out_dir / ".image_cache_build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _build_cache(args)


def _build_cache(args: argparse.Namespace) -> None:
    if args.num_samples <= 0 or args.num_modalities <= 0:
        raise ValueError("--num-samples and --num-modalities must be positive.")
    if len(set(args.cache_modalities)) != len(args.cache_modalities):
        raise ValueError(f"Duplicate cache modalities: {args.cache_modalities}")
    modality_indices = []
    for name in args.cache_modalities:
        if not name.startswith("m") or not name[1:].isdigit():
            raise ValueError(f"Invalid modality {name!r}; expected m0, m1, ...")
        modality = int(name[1:])
        if not 0 <= modality < args.num_modalities:
            raise ValueError(f"{name} lies outside m0..m{args.num_modalities - 1}")
        modality_indices.append(modality)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.cache_path or (
        args.out_dir / f"images_{'-'.join(args.cache_modalities)}_{CANVAS_SIZE}_u8.pt"
    )
    labels_path = args.out_dir / "labels.csv"
    protected = [cache_path, labels_path]
    existing = [path for path in protected if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {existing}. Pass --overwrite to replace it."
        )

    mnist_root = args.mnist_root or args.source_dir
    mnist_images, mnist_labels = load_mnist(mnist_root, args.download)
    pools = build_digit_pools(mnist_labels, args.num_modalities, args.seed)
    backgrounds = load_backgrounds(args)
    images = torch.empty(
        (len(args.cache_modalities), args.num_samples, 3, CANVAS_SIZE, CANVAS_SIZE),
        dtype=torch.uint8,
    )
    mask_cache: dict[int, np.ndarray] = {}
    labels_temporary = create_temporary_file(labels_path)
    try:
        with labels_temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["sample_id", "digit", "quadrant"]
            )
            writer.writeheader()
            for sample in tqdm(
                range(args.num_samples), desc="Building PolyMNIST cache"
            ):
                case = sample % (10 * NUM_QUADRANTS)
                digit, quadrant = case // NUM_QUADRANTS, case % NUM_QUADRANTS
                source_indices = draw_distinct(pools[digit])
                for cache_index, modality in enumerate(modality_indices):
                    source_index = source_indices[modality]
                    mask = mask_cache.get(source_index)
                    if mask is None:
                        mask = digit_mask(mnist_images[source_index])
                        mask_cache[source_index] = mask
                    background = crop_background(
                        backgrounds[modality], args.seed, sample, modality
                    )
                    images[cache_index, sample].copy_(
                        compose_image(background, mask, quadrant)
                    )
                writer.writerow(
                    {"sample_id": f"{sample:06d}", "digit": digit, "quadrant": quadrant}
                )

        labels_hash = sha256_file(labels_temporary)
        save_image_tensor_cache(
            cache_path,
            modalities=args.cache_modalities,
            keys=[f"{sample:06d}" for sample in range(args.num_samples)],
            images=images,
            img_size=CANVAS_SIZE,
            metadata={
                "builder": "scripts/data_prep/build_polymnist_cache.py",
                "labels_sha256": labels_hash,
                "seed": args.seed,
                "num_samples": args.num_samples,
                "num_source_images": len(mnist_labels),
                "num_modalities": args.num_modalities,
                "canvas_size": CANVAS_SIZE,
                "digit_size": DIGIT_SIZE,
                "threshold": THRESHOLD,
                "background_preset": args.background_preset,
                "background_dir": str(args.background_dir)
                if args.background_dir
                else None,
                "source_dir": str(args.source_dir),
                "mnist_root": str(mnist_root),
                "mnist_part": "all",
                "source_dataset": "mnist",
                "quadrant_policy": "cycle",
            },
            overwrite=args.overwrite,
        )
        publish_file(labels_temporary, labels_path, overwrite=args.overwrite)
    finally:
        labels_temporary.unlink(missing_ok=True)

    print(f"Saved {images.nbytes / 1024**3:.3f} GiB cache to {cache_path}")


if __name__ == "__main__":
    build_cache(parse_args())
