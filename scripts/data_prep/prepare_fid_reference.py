"""Prepare reference images and compute FID statistics."""

import argparse
import glob
import os

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from muni.evaluation.metrics.common import ImageFID
from muni.util.crop import center_crop_arr


# Image preprocessing


class CroppedImageFolder(Dataset):
    """ImageFolder-like dataset that center-crops to img_size and returns (PIL, index)."""

    EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"}

    def __init__(self, root: str, img_size: int):
        self.img_size = img_size
        self.paths = sorted(
            p
            for p in glob.glob(os.path.join(root, "**", "*"), recursive=True)
            if os.path.splitext(p)[1].lower() in self.EXTS
        )
        assert len(self.paths) > 0, f"No images found in {root}"

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        img = center_crop_arr(img, self.img_size)
        return transforms.functional.pil_to_tensor(img), i


def prepare_images(
    data_path: str,
    output_path: str,
    img_size: int,
    batch_size: int = 256,
    num_workers: int = 16,
):
    """Center-crop source images and save as PNGs. Skips already-written files."""
    os.makedirs(output_path, exist_ok=True)
    dataset = CroppedImageFolder(data_path, img_size)

    # Check how many are already done for resume
    existing = set(os.listdir(output_path))
    total = len(dataset)
    done = sum(1 for i in range(total) if f"{i:08d}.png" in existing)
    if done >= total:
        print(f"All {total} images already exist in {output_path}, skipping resize.")
        return
    print(f"Resuming image prep: {done}/{total} already done.")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=False,
    )
    to_pil = transforms.ToPILImage()
    saved = done

    for batch_imgs, batch_idxs in tqdm(loader, desc="Saving images"):
        for img_t, idx in zip(batch_imgs, batch_idxs):
            fname = f"{idx.item():08d}.png"
            if fname in existing:
                continue
            to_pil(img_t).save(
                os.path.join(output_path, fname), format="PNG", compress_level=1
            )
            saved += 1
    print(f"Done: {saved} images in {output_path}")


# FID statistics


def compute_stats(
    image_dir: str,
    stats_path: str,
    batch_size: int = 256,
):
    """Compute and save stats through the canonical ImageFID implementation."""
    print(f"Computing FID stats from {image_dir} ...")
    metric = ImageFID(batch_size=batch_size)
    mean, covariance = metric.save_statistics(image_dir, stats_path)
    print(
        f"Saved FID stats to {stats_path}  "
        f"(mu: {mean.shape}, sigma: {covariance.shape})"
    )


# Main


def main():
    parser = argparse.ArgumentParser(
        description="Prepare reference images and/or compute FID statistics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="",
        help="Source image directory (e.g. ImageNet train split)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="ref_images",
        help="Directory of processed (center-cropped) images",
    )
    parser.add_argument(
        "--stats_path",
        type=str,
        default="fid_stats/ref_stats.npz",
        help="Output path for the .npz FID statistics file",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=256,
        help="Resolution for center-crop (default: 256)",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--stats_only",
        action="store_true",
        help="Skip image preprocessing; compute stats from output_path directly",
    )
    args = parser.parse_args()

    # Phase 1: image preprocessing
    if not args.stats_only:
        assert args.data_path, "--data_path is required unless --stats_only is set"
        prepare_images(
            args.data_path, args.output_path, args.img_size, batch_size=args.batch_size
        )

    # Phase 2: compute FID stats
    image_dir = args.output_path
    compute_stats(image_dir, args.stats_path, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
