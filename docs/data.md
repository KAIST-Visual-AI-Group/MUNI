# Image-text-audio data preparation

Set `DATA_ROOT` and `CKPT_ROOT` as in the [README](../README.md#installation).
Acquire the datasets from their original providers and follow their usage terms.

## Training inputs

Set the metadata and media paths in
[`configs/data/image_text_audio.yaml`](../configs/data/image_text_audio.yaml).

| Dataset | Metadata | Media |
|---|---|---|
| [LAION-COCO](https://huggingface.co/datasets/guangyil/laion-coco-aesthetic) | CSV or Parquet with `sha256` and `caption` | Images named by `sha256` |
| [Flickr30k](https://huggingface.co/datasets/nlphuji/flickr30k) | Annotation CSV with `filename` and `raw` | Original images |
| [AudioCaps](https://github.com/cdjkim/audiocaps) | `train.csv` | WAV clips named `<youtube_id>_<start_time>.wav` |
| [VGGSound](https://github.com/hche11/VGGSound) | Original CSV | Frames and WAV clips named `<youtube_id>_<start_time:06d>` |

For LAION, use metadata for the images you downloaded.
Images may be PNG, JPG, or JPEG.

## Evaluation inputs

Prepare inputs for your selected [MS-COCO 2014 validation](https://cocodataset.org/#download) subset,
the AudioCaps test split, and the VGGSound test split. AudioCaps multimodal evaluation also
requires images aligned with the selected audio clips and captions.

```text
$DATA_ROOT/image_text_audio_raw/
├── coco_val2014/
│   ├── captions.json
│   └── images/<key>.jpg
├── audiocaps_test/
│   ├── captions.json
│   ├── images/<key>.jpg
│   └── audio/<key>.wav
└── vggsound_test/
    ├── images/<key>.png
    └── audio/<key>.wav
```

For MS-COCO 2014 validation and the AudioCaps test split, map each filename stem to all its
reference captions in `captions.json`:

```json
{"sample_0001": ["First caption.", "Second caption."]}
```

Use matching sample keys for paired images, audio, and captions.

## Feature assets

Feature extraction uses these assets from the
[README downloads](../README.md#checkpoints):

```text
$CKPT_ROOT/flowbind/
├── embeddinggemma-300m/
├── clap.pt
└── lightweight_audio_ae.pth
```

OpenAI CLIP ViT-L/14 downloads on first use. Feature normalization uses the
statistics in `assets/image_text_audio/stats`. CLIP-based evaluation metrics use
a separate ViT-B/32 encoder, which OpenCLIP also downloads on first use.

## Build the caches

```bash
python -m scripts.data_prep.build_image_text_audio_cache
```

To build only the evaluation caches:

```bash
python -m scripts.data_prep.build_image_text_audio_cache \
  --datasets coco_val2014 audiocaps_test vggsound_test
```

Caches are written to `$DATA_ROOT/image_text_audio/{train,eval}/<dataset>`.
Each contains `feats.parquet` and, where available, `captions.json`.
Evaluation media are stored in `images.parquet` and `audio.parquet`.

## Reference statistics

Download the released FID/FAD statistics:

```bash
hf download 32V/MUNI --include "stats/*" --local-dir "$DATA_ROOT/image_text_audio"
```

Or generate them from your evaluation data:

```bash
python -m scripts.data_prep.prepare_fid_reference
python -m scripts.data_prep.prepare_fad_reference
```

For a different evaluation split, set the paths and sample counts in
[`configs/data/image_text_audio_references.yaml`](../configs/data/image_text_audio_references.yaml).

