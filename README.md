# MUNI: Multimodal Unified Latent Diffusion for Coherent Any-to-Any Generation

[Kyeongmin Yeo](https://32v.github.io/)\*,
[Yunhong Min](https://myh4832.github.io/)\*,
[Minhyuk Sung](https://mhsung.github.io/)  
KAIST  
\* Equal contribution

NeurIPS 2026

[Paper](https://arxiv.org/abs/2606.16408) |
[Project Page](https://muni-proj.github.io/)

Official PyTorch implementation of MUNI. Modality-specific Gaussian posteriors
are combined through product-of-experts or Hellinger aggregation, with encoders
and decoders trained jointly with a flow-based latent prior.

Experiments cover image-text-audio, PolyMNIST-Quadrant-Labels, and GMM.


<table>
  <tr>
    <td align="center" width="50%">
      <img src="https://muni-proj.github.io/muni/qualitative/qualitative_ita/uncond_ours/ours_hell_200_078.png" width="62%" alt="Generated scene with people and bicycles"><br>
      <em>“A group of people with bicycles in the grass.”</em><br>
      <a href="https://muni-proj.github.io/muni/supplementary/Figure3/row0_col0.wav">
        <img src="https://muni-proj.github.io/muni/waveforms/Figure3/row0_col0.png" width="92%" alt="Generated audio waveform for the bicycle scene"><br>
        <sub>▶ Play generated audio</sub>
      </a>
    </td>
    <td align="center" width="50%">
      <img src="https://muni-proj.github.io/muni/qualitative/qualitative_ita/uncond_ours/ours_poe_200_255.png" width="62%" alt="Generated beach scene"><br>
      <em>“The beach is empty and clear with palm trees.”</em><br>
      <a href="https://muni-proj.github.io/muni/supplementary/Figure3/row0_col1.wav">
        <img src="https://muni-proj.github.io/muni/waveforms/Figure3/row0_col1.png" width="92%" alt="Generated audio waveform for the beach scene"><br>
        <sub>▶ Play generated audio</sub>
      </a>
    </td>
  </tr>
</table>
<p align="center"><em>Unconditional image-text-audio co-generation: each triplet is sampled jointly from MUNI’s learned prior. See the <a href="https://muni-proj.github.io/">project page</a> for more results.</em></p>

## Installation

Use Python 3.10 and PyTorch 2.9.1. The example below uses CUDA 12.8; choose
another build from the [PyTorch installation guide](https://docs.pytorch.org/get-started/previous-versions/)
if needed.

```bash
conda create -n muni python=3.10
conda activate muni

# Example: PyTorch 2.9.1 with CUDA 12.8
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
  --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
```

```bash
export DATA_ROOT=/path/to/data
export CKPT_ROOT=/path/to/checkpoints
```

`DATA_ROOT` contains prepared datasets and runtime caches; `CKPT_ROOT` contains
local foundation checkpoints. Run the commands from the repository root.

## Checkpoints

Image-text-audio uses the foundation models and adapters from
[FlowBind](https://github.com/yeonwoo378/flowbind) and its
[model release](https://huggingface.co/yeonwoo378/flowbind). Arrange them as follows:

```text
$CKPT_ROOT/flowbind/
├── gemmae_3.pt
├── clap.pt
├── embeddinggemma-300m/
├── gemma-3-1b-pt/
├── stable-diffusion-2-1-unclip-small/
├── audioldm-m-full/
└── lightweight_audio_ae.pth
```

Feature extraction also needs OpenAI CLIP ViT-L/14; see the
[data preparation guide](docs/data.md#feature-assets).

## Data preparation

Prepare data before training. PolyMNIST-Quadrant-Labels also requires a verifier.

<details>
<summary>PolyMNIST-Quadrant-Labels</summary>

Download MNIST and build the aligned image cache and labels under
`$DATA_ROOT/poly_mnist`:

```bash
bash scripts/build_polymnist_cache.sh
```

Train the verifier and copy its final checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python -m verifiers.train \
  --config verifiers/configs/polymnist.yaml
mkdir -p "$CKPT_ROOT"
cp verifiers/checkpoints/polymnist/last.pth "$CKPT_ROOT/polymnist64_verifier.pth"
```

</details>

<details>
<summary>Image-text-audio</summary>

Follow the [data preparation guide](docs/data.md) to prepare the inputs and
feature models. Build feature and media caches under
`$DATA_ROOT/image_text_audio/train/<name>` and
`$DATA_ROOT/image_text_audio/eval/<name>`:

```bash
python -m scripts.data_prep.build_image_text_audio_cache \
  --config configs/data/image_text_audio.yaml --device cuda:0 --batch-size 64
```

Each group contains `feats.parquet`; text groups also have `captions.json`, and
evaluation groups include `images.parquet` and/or `audio.parquet`.

</details>

<details>
<summary>Controlled Gaussian mixture</summary>

GMM samples are generated procedurally. Build the reference marginals under
`output/gmm/marginals` before training:

```bash
python -m scripts.build_gmm_marginals --out-dir output/gmm/marginals
```

</details>

## Training

```bash
# Image-text-audio
bash scripts/train.sh configs/image_text_audio/muni_poe.yaml --gpus 2 train.batch_size=512

# PolyMNIST-Quadrant-Labels
bash scripts/train.sh configs/poly_mnist/muni.yaml --gpus 2

# Controlled Gaussian mixture
bash scripts/train.sh configs/gmm/muni.yaml --gpus 1
```

`train.batch_size` is per GPU. Keep the effective batch size unchanged when
adjusting `--gpus`. Resume a run with `--resume output/path`; override YAML
settings with `key=value`.

For Hellinger aggregation in image-text-audio, use
`configs/image_text_audio/muni_helvae.yaml`.

## Evaluation

Use the benchmark commands for the task-specific evaluation protocols:

```bash
# Image-text-audio
python -m scripts.benchmark image-text-audio --gpu 0 \
  --config configs/image_text_audio/muni_poe.yaml \
  --checkpoint output/image_text_audio/muni_poe/checkpoint-200.pth \
  --eval-config configs/image_text_audio/eval_benchmark.yaml \
  --tasks all --method MUNI

# PolyMNIST-Quadrant-Labels
python -m scripts.benchmark polymnist --gpu 0 \
  --method muni \
  --config configs/poly_mnist/muni.yaml \
  --checkpoint output/poly_mnist/muni/checkpoint-300.pth \
  --verifier-ckpt "$CKPT_ROOT/polymnist64_verifier.pth"

# Controlled Gaussian mixture
python -m scripts.benchmark gmm --gpu 0 \
  --method muni \
  --config configs/gmm/muni.yaml \
  --checkpoint output/gmm/muni/checkpoint-last.pth
```

Image-text-audio fidelity metrics need the reference statistics and captions
listed in the [data guide](docs/data.md#evaluation-inputs).

Generate decoded image-text-audio samples with:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.qualitative \
  --config configs/image_text_audio/muni_poe.yaml \
  --checkpoint output/image_text_audio/muni_poe/checkpoint-200.pth \
  --eval-config configs/image_text_audio/eval_benchmark.yaml \
  --output-dir output/image_text_audio/muni_poe/qualitative
```

The default image renderer is Stable UnCLIP. To use Stable Cascade, select
`configs/image_text_audio/muni_poe_cascade_large.yaml` and provide the models
under `$CKPT_ROOT/stable-cascade-prior` and `$CKPT_ROOT/stable-cascade`.

## Citation

```bibtex
@inproceedings{yeo2026muni,
  title={MUNI: Multimodal Unified Latent Diffusion for Coherent Any-to-Any Generation},
  author={Yeo, Kyeongmin and Min, Yunhong and Sung, Minhyuk},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2026}
}
```

## Acknowledgements

This codebase builds on these open-source projects:

- [FlowBind](https://github.com/yeonwoo378/flowbind)
- [score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch)
- [JiT (Just image Transformer)](https://github.com/LTH14/JiT)
