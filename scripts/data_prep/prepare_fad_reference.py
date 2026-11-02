"""Regenerate FAD reference statistics from evaluation audio."""
from __future__ import annotations

import io

import numpy as np
import resampy
import soundfile as sf

from muni.evaluation.metrics.image_text_audio import AudioFAD
from muni.evaluation.metrics.common import ImageFID
from scripts.data_prep.image_text_audio.references import run


def waveform(cohort, index):
    samples, rate = sf.read(io.BytesIO(cohort.encoded(index)), dtype="float32")
    if not samples.size or rate <= 0 or not np.isfinite(samples).all():
        raise ValueError(f"{cohort.name}: empty/nonfinite audio or invalid rate for input {cohort.keys[index]!r}")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if rate != 16000:
        samples = resampy.resample(samples, rate, 16000, filter="kaiser_best")
    return samples


def prepare(args):
    metric = AudioFAD(device=args.device, batch_size=args.batch_size)
    metric._ensure_loaded()

    def compute(cohort):
        features = []
        for start in range(0, len(cohort), args.batch_size):
            waveforms = [waveform(cohort, index) for index in
                         range(start, min(start + args.batch_size, len(cohort)))]
            features.append(metric._embed_waveforms(waveforms, 16000))
        # The canonical shared moments preserve float32 means and float64 covariance.
        return ImageFID._statistics_from_features(np.concatenate(features, axis=0))

    return compute


def main():
    run("fad", prepare)


if __name__ == "__main__":
    main()
