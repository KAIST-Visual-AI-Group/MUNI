# Image-text-audio data preparation

Set `DATA_ROOT` and `CKPT_ROOT` as in the [README](../README.md#installation).
Place the datasets below under `$DATA_ROOT/image_text_audio_raw`, or edit their
paths in [`configs/data/image_text_audio.yaml`](../configs/data/image_text_audio.yaml).

## Training data

Sources: [LAION-COCO](https://huggingface.co/datasets/guangyil/laion-coco-aesthetic),
[Flickr30k](https://huggingface.co/datasets/nlphuji/flickr30k),
[AudioCaps](https://github.com/cdjkim/audiocaps), and
[VGGSound](https://huggingface.co/datasets/Loie/VGGSound).

```text
$DATA_ROOT/image_text_audio_raw/
├── laion_coco_aesthetic/
│   ├── metadata.csv                 # columns: image_id,text
│   └── images/<image_id>.jpg
├── flickr30k/
│   ├── annotations.csv              # columns: filename,raw (JSON caption list)
│   └── images/<filename>
├── audiocaps/
│   ├── metadata/
│   │   ├── train.csv                # columns: audiocap_id,youtube_id,start_time,caption
│   │   ├── val.csv
│   │   └── test.csv
│   └── audio/<youtube_id>_<start_time>.wav
└── vggsound/
    ├── metadata.csv                 # columns: youtube_id,start_time,label,split
    ├── images/<youtube_id>_<start_time:06d>.png
    └── audio/<youtube_id>_<start_time:06d>.wav
```

Use the selected LAION caption rows. In AudioCaps filenames, write `start_time`
as an unpadded integer; in VGGSound filenames, pad it to six digits.

## Evaluation inputs

Use [COCO val2014](https://cocodataset.org/#download), AudioCaps test, and VGGSound
test. AudioCaps evaluation requires paired images as well as audio and captions.

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

For COCO and AudioCaps, map each filename stem to its captions in `captions.json`:

```json
{"sample_0001": ["First caption.", "Second caption."]}
```

Use the same keys for paired images and audio, with all reference captions in
`captions.json`.

## Metric references

Build FID references from the evaluation images:

```bash
for dataset in coco_val2014 vggsound_test; do
  python -m scripts.data_prep.prepare_fid_reference \
    --data_path "$DATA_ROOT/image_text_audio_raw/$dataset/images" \
    --output_path "$DATA_ROOT/image_text_audio_references/$dataset/images" \
    --stats_path "$DATA_ROOT/image_text_audio_references/$dataset/fid_stats.npz" \
    --img_size 256
done
```

FAD reference statistics for AudioCaps test and VGGSound test are included in
`assets/image_text_audio/reference_stats`. The data config copies the FID and FAD
files into their evaluation caches.

## Feature assets

Download the models from the [FlowBind release](https://huggingface.co/yeonwoo378/flowbind)
and the OpenAI [ViT-L/14 checkpoint](https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt),
and place them here:

```text
$CKPT_ROOT/
├── clip/ViT-L-14.pt
└── flowbind/
    ├── embeddinggemma-300m/
    ├── clap.pt
    └── lightweight_audio_ae.pth
```

## Build the caches

Run from the repository root, first checking the inputs and then extracting features:

```bash
python -m scripts.data_prep.build_image_text_audio_cache \
  --config configs/data/image_text_audio.yaml --dry-run
python -m scripts.data_prep.build_image_text_audio_cache \
  --config configs/data/image_text_audio.yaml --device cuda:0 --batch-size 64
```

The caches are saved under `$DATA_ROOT/image_text_audio/train/<dataset>` and
`$DATA_ROOT/image_text_audio/eval/<dataset>`.
