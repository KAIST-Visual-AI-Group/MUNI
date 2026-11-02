"""Regenerate FID reference statistics from evaluation images."""
from __future__ import annotations

import io

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from muni.evaluation.metrics.common import ImageFID
from muni.util.crop import center_crop_arr
from scripts.data_prep.image_text_audio.references import run


class ReferenceImages(Dataset):
    """Decode source images directly, without cropped-image copies on disk."""

    def __init__(self, cohort):
        self.cohort = cohort

    def __len__(self):
        return len(self.cohort)

    def __getitem__(self, index):
        with Image.open(io.BytesIO(self.cohort.encoded(index))) as image:
            cropped = center_crop_arr(image.convert("RGB"), 256)
        return torch.from_numpy(np.array(cropped)).permute(2, 0, 1).float().div_(255.0)


def prepare(args):
    metric = ImageFID(device=args.device, batch_size=args.batch_size, value_range="zero_one")

    def compute(cohort):
        return metric.compute_statistics(ReferenceImages(cohort))

    return compute


def main():
    run("fid", prepare)


if __name__ == "__main__":
    main()
