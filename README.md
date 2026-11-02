# MUNI: Multimodal Unified Latent Diffusion for Coherent Any-to-Any Generation

[Kyeongmin Yeo](https://32v.github.io/)\*,
[Yunhong Min](https://myh4832.github.io/)\*,
[Minhyuk Sung](https://mhsung.github.io/)  
KAIST  
\* Equal contribution

NeurIPS 2026

[Paper](https://arxiv.org/abs/2606.16408) |
[Project Page](https://muni-proj.github.io/) |
[Checkpoints](https://huggingface.co/32V/MUNI)

Official PyTorch implementation of MUNI.

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

Tested with Python 3.10 and PyTorch 2.9.1 (CUDA 12.8). For other builds, see the
[PyTorch installation guide](https://docs.pytorch.org/get-started/previous-versions/).

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

`DATA_ROOT` contains raw datasets, prepared caches, and evaluation references.
`CKPT_ROOT` contains downloaded model checkpoints.

## Checkpoints

Download the MUNI-PoE EMA weights:

```bash
hf download 32V/MUNI muni_poe.pth --local-dir "$CKPT_ROOT/muni"
```

Image-text-audio uses the foundation models and adapters from
[FlowBind](https://github.com/yeonwoo378/flowbind). Its
[model release](https://huggingface.co/yeonwoo378/flowbind) provides the adapters, so download the foundation models separately:

```bash
hf download yeonwoo378/flowbind gemmae_3.pt clap.pt lightweight_audio_ae.pth \
  --local-dir "$CKPT_ROOT/flowbind"
# Accept the Google model licenses on Hugging Face, then authenticate.
hf auth login
hf download google/embeddinggemma-300m --local-dir "$CKPT_ROOT/flowbind/embeddinggemma-300m"
hf download google/gemma-3-1b-pt --local-dir "$CKPT_ROOT/flowbind/gemma-3-1b-pt"
hf download sd2-community/stable-diffusion-2-1-unclip-small \
  --local-dir "$CKPT_ROOT/flowbind/stable-diffusion-2-1-unclip-small"
hf download cvssp/audioldm-m-full --local-dir "$CKPT_ROOT/flowbind/audioldm-m-full"
```

The resulting layout is:

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

## Demo

[demo.ipynb](demo.ipynb) generates text and audio from your own image without a
dataset cache.

## Data preparation

PolyMNIST-Quadrant-Labels evaluation needs a trained verifier.

<details>
<summary>PolyMNIST-Quadrant-Labels</summary>

Download MNIST and build the aligned image cache and labels under
`$DATA_ROOT/poly_mnist`:

```bash
bash scripts/build_polymnist_cache.sh
```

Train the verifier:

```bash
CUDA_VISIBLE_DEVICES=0 python -m verifiers.train \
  --config verifiers/configs/polymnist.yaml
```

</details>

<details>
<summary>Image-text-audio</summary>

Prepare the original datasets and feature models following the
[data guide](docs/data.md), then build the feature and media caches:

```bash
python -m scripts.data_prep.build_image_text_audio_cache
```

Download or generate the FID/FAD [reference statistics](docs/data.md#reference-statistics).

</details>

<details>
<summary>Controlled Gaussian mixture</summary>

GMM training and evaluation samples are generated in memory. Build the reference
marginals under `$DATA_ROOT/gmm/marginals` before training:

```bash
python -m scripts.build_gmm_marginals --config configs/gmm/muni.yaml
```

Use the same config for reference generation and training.

</details>

## Training

```bash
# Image-text-audio
bash scripts/train.sh configs/image_text_audio/muni_poe.yaml --gpus 2

# PolyMNIST-Quadrant-Labels
bash scripts/train.sh configs/poly_mnist/muni.yaml --gpus 2

# Controlled Gaussian mixture
bash scripts/train.sh configs/gmm/muni.yaml --gpus 1
```

Use `--init_from "$CKPT_ROOT/muni/muni_poe.pth"` to start from the released weights,
or `--resume output/path` to continue a training run.
Override YAML settings with `key=value`.

For Hellinger aggregation in image-text-audio, use `configs/image_text_audio/muni_helvae.yaml`.

## Evaluation

Use the benchmark commands for the task-specific evaluation protocols:

```bash
# Image-text-audio
python -m scripts.benchmark image-text-audio --gpu 0 \
  --config configs/image_text_audio/muni_poe.yaml \
  --checkpoint "$CKPT_ROOT/muni/muni_poe.pth" \
  --eval-config configs/image_text_audio/eval_benchmark.yaml \
  --tasks all --method MUNI

# PolyMNIST-Quadrant-Labels
python -m scripts.benchmark polymnist --gpu 0 \
  --method muni \
  --config configs/poly_mnist/muni.yaml \
  --checkpoint output/poly_mnist/muni/checkpoint-300.pth

# Controlled Gaussian mixture
python -m scripts.benchmark gmm --gpu 0 \
  --method muni \
  --config configs/gmm/muni.yaml \
  --checkpoint output/gmm/muni/checkpoint-last.pth
```

Image-text-audio fidelity metrics need the reference statistics and captions
listed in the [data guide](docs/data.md#evaluation-inputs).

Generate unconditional image-text-audio samples:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.qualitative \
  --config configs/image_text_audio/muni_poe.yaml \
  --checkpoint "$CKPT_ROOT/muni/muni_poe.pth" \
  --eval-config configs/image_text_audio/eval_benchmark.yaml \
  --output-dir output/image_text_audio/muni_poe/qualitative \
  --skip-translation --num-unconditional 4 --no-compile
```

The default image renderer is Stable UnCLIP. To use Stable Cascade, select
`configs/image_text_audio/muni_poe_cascade_large.yaml` and provide the models
under `$CKPT_ROOT/stable-cascade-prior` and `$CKPT_ROOT/stable-cascade`.

## Citation

```bibtex
@inproceedings{yeo2026muni,
  title={{MUNI}: Multimodal Unified Latent Diffusion for Coherent Any-to-Any Generation},
  author={Yeo, Kyeongmin and Min, Yunhong and Sung, Minhyuk},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}
```

## Acknowledgements

This codebase builds on these open-source projects:

- [FlowBind](https://github.com/yeonwoo378/flowbind)
- [score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch)
- [JiT (Just image Transformer)](https://github.com/LTH14/JiT)

## License

The source code is [MIT-licensed](LICENSE).
