<h1 align="center">LatentSync</h1>

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2412.09262)
[![arXiv](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow)](https://huggingface.co/ByteDance/LatentSync-1.6)
[![arXiv](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Space-yellow)](https://huggingface.co/spaces/fffiloni/LatentSync)
<a href="https://replicate.com/lucataco/latentsync"><img src="https://replicate.com/lucataco/latentsync/badge" alt="Replicate"></a>

</div>

> [!NOTE]
> **This is a fork of [bytedance/LatentSync](https://github.com/bytedance/LatentSync) that adds support for macOS / Apple Silicon (MPS).**
> The model, training code and checkpoints are unchanged and remain the work of the original authors (see [Acknowledgement](#-acknowledgement) and [Citation](#-citation)).
> 
> Jump to [🍎 macOS / Apple Silicon](#-macos--apple-silicon-fork) for the setup guide.

## 🔥 Updates

- `2025/06/11`: We released **LatentSync 1.6**, which is trained on 512 $\times$ 512 resolution videos to mitigate the blurriness problem. Watch the demo [here](docs/changelog_v1.6.md).

- `2025/03/14`: We released **LatentSync 1.5**, which **(1)** improves temporal consistency via adding temporal layer, **(2)** improves performance on Chinese videos and **(3)** reduces the VRAM requirement of the stage2 training to **20 GB** through a series of optimizations. Learn more details [here](docs/changelog_v1.5.md).

## 📖 Introduction

We present *LatentSync*, an end-to-end lip-sync method based on audio-conditioned latent diffusion models without any intermediate motion representation, diverging from previous diffusion-based lip-sync methods based on pixel-space diffusion or two-stage generation. Our framework can leverage the powerful capabilities of Stable Diffusion to directly model complex audio-visual correlations.

## 🏗️ Framework

<p align="center">
<img src="docs/framework.png" width=100%>
<p>

LatentSync uses the [Whisper](https://github.com/openai/whisper) to convert melspectrogram into audio embeddings, which are then integrated into the U-Net via cross-attention layers. The reference and masked frames are channel-wise concatenated with noised latents as the input of U-Net. In the training process, we use a one-step method to get estimated clean latents from predicted noises, which are then decoded to obtain the estimated clean frames. The TREPA, [LPIPS](https://arxiv.org/abs/1801.03924) and [SyncNet](https://www.robots.ox.ac.uk/~vgg/publications/2016/Chung16a/chung16a.pdf) losses are added in the pixel space.

## 🎬 Demo

<table class="center">
  <tr style="font-weight: bolder;text-align:center;">
        <td width="50%"><b>Original video</b></td>
        <td width="50%"><b>Lip-synced video</b></td>
  </tr>
  <tr>
    <td>
      <video src=https://github.com/user-attachments/assets/b778e3c3-ba25-455d-bdf3-d89db0aa75f4 controls preload></video>
    </td>
    <td>
      <video src=https://github.com/user-attachments/assets/ac791682-1541-4e6a-aa11-edd9427b977e controls preload></video>
    </td>
  </tr>
  <tr>
    <td>
      <video src=https://github.com/user-attachments/assets/6d4f4afd-6547-428d-8484-09dc53a19ecf controls preload></video>
    </td>
    <td>
      <video src=https://github.com/user-attachments/assets/b4723d08-c1d4-4237-8251-09c43eb77a6a controls preload></video>
    </td>
  </tr>
  <tr>
    <td>
      <video src=https://github.com/user-attachments/assets/fb4dc4c1-cc98-43dd-a211-1ff8f843fcfa controls preload></video>
    </td>
    <td>
      <video src=https://github.com/user-attachments/assets/7c6ca513-d068-4aa9-8a82-4dfd9063ac4e controls preload></video>
    </td>
  </tr>
  <tr>
    <td width=300px>
      <video src=https://github.com/user-attachments/assets/0756acef-2f43-4b66-90ba-6dc1d1216904 controls preload></video>
    </td>
    <td width=300px>
      <video src=https://github.com/user-attachments/assets/663ff13d-d716-4a35-8faa-9dcfe955e6a5 controls preload></video>
    </td>
  </tr>
  <tr>
    <td>
      <video src=https://github.com/user-attachments/assets/0f7f9845-68b2-4165-bd08-c7bbe01a0e52 controls preload></video>
    </td>
    <td>
      <video src=https://github.com/user-attachments/assets/c34fe89d-0c09-4de3-8601-3d01229a69e3 controls preload></video>
    </td>
  </tr>
</table>

(Photorealistic videos are filmed by contracted models, and anime videos are from [VASA-1](https://www.microsoft.com/en-us/research/project/vasa-1/))

## 📑 Open-source Plan

- [x] Inference code and checkpoints
- [x] Data processing pipeline
- [x] Training code

## 🍎 macOS / Apple Silicon (fork)

This fork makes LatentSync run on Apple Silicon GPUs through PyTorch MPS, with CPU fallback. The original CUDA path is kept: the device is picked automatically (`cuda` > `mps` > `cpu`).

Tested on a **MacBook M3 with 18 GB of unified memory**:

| Model | Resolution | Status on M3 18 GB |
| --- | --- | --- |
| LatentSync 1.5 | 256 × 256 | Runs efficiently with the default settings |
| LatentSync 1.6 | 512 × 512 | Runs, but slower, with the memory variables below |

### What was changed

- Automatic device / dtype selection (`latentsync/utils/util.py`) instead of hard-coded `"cuda"`; float32 by default on MPS, since float16 degrades the output.
- Chunked attention on MPS to stay under the Metal buffer size limit at 512 × 512 (same result, lower peak memory).
- VAE encoding/decoding by batches of frames to lower peak memory.
- Face detection through ONNX Runtime CoreML instead of CUDA.
- Whisper runs in float32 outside CUDA.
- Python 3.13 environment managed with [`uv`](https://docs.astral.sh/uv/), and refreshed dependencies (see `requirements.txt`).

### Setup

Requires [`uv`](https://docs.astral.sh/uv/) and, if missing, Homebrew to install `ffmpeg`.

```bash
source setup_env.sh
```

This downloads the checkpoints for both LatentSync 1.6 (`checkpoints/`) and 1.5 (`checkpoints/v1.5/`).

### Run LatentSync 1.5 at 256 × 256 (recommended on 18 GB)

This is the default configuration of `inference.sh`:

```bash
./inference.sh
```

which runs `configs/unet/stage2.yaml` with `checkpoints/v1.5/latentsync_unet.pt`.

### Run at 512 × 512 (LatentSync 1.6, slower)

In `inference.sh`, use the 512 config and the 1.6 checkpoint:

```bash
--unet_config_path "configs/unet/stage2_512.yaml" \
--inference_ckpt_path "checkpoints/latentsync_unet.pt" \
```

then lower the memory usage with the environment variables below:

```bash
LATENTSYNC_DTYPE=bfloat16 MPS_ATTENTION_BUDGET_MB=256 VAE_CHUNK_SIZE=2 ./inference.sh
```

### Memory environment variables

Set them when the process is killed or the machine swaps heavily. All are optional.

| Variable | Default | Effect |
| --- | --- | --- |
| `LATENTSYNC_DTYPE` | `float32` on MPS | `float32`, `bfloat16` or `float16`. `bfloat16` halves the memory of models and activations. |
| `MPS_ATTENTION_BUDGET_MB` | `1024` | Memory budget of the attention scores. Lower = less memory, more chunks. |
| `VAE_CHUNK_SIZE` | `16` | Number of frames per VAE batch. Lower = less memory. |

`PYTORCH_ENABLE_MPS_FALLBACK=1` (already set in `inference.sh`) falls back to the CPU for operations not yet implemented on MPS.

> [!NOTE]
> Forcing `LATENTSYNC_DTYPE=bfloat16` is also worthwhile at 256 resolution: it speeds up inference by almost 20% with minimal quality degradation (measured on the 256 model, not a guarantee for every video).

> [!TIP]
> The values above (`bfloat16`, `256`, `2`) are a conservative starting point for 18 GB, not tuned optima. Raise `MPS_ATTENTION_BUDGET_MB` and `VAE_CHUNK_SIZE` if you have more memory to go faster. Lowering `--inference_steps` (e.g. 5–8 with `--enable_deepcache`) also shortens the run at some cost in quality.

## 🔧 Setting up the Environment

> On macOS, follow the [macOS / Apple Silicon](#-macos--apple-silicon-fork) section above.

Install the required packages and download the checkpoints via:

```bash
source setup_env.sh
```

If the download is successful, the checkpoints should appear as follows:

```
./checkpoints/
|-- latentsync_unet.pt
|-- whisper
|   `-- tiny.pt
```

Or you can download `latentsync_unet.pt` and `tiny.pt` manually from our [HuggingFace repo](https://huggingface.co/ByteDance/LatentSync-1.6)

## 🚀 Inference

Minimum VRAM for inference:

- **8 GB** with LatentSync 1.5
- **18 GB** with LatentSync 1.6

There are two ways to perform inference:

### 1. Gradio App

Run the Gradio app for inference:

```bash
python gradio_app.py
```

### 2. Command Line Interface

Run the script for inference:

```bash
./inference.sh
```

You can try adjusting the following inference parameters to achieve better results:

- `inference_steps` [20-50]: A higher value improves visual quality but slows down the generation speed.
- `guidance_scale` [1.0-3.0]: A higher value improves lip-sync accuracy but may cause the video distortion or jitter.

## 🔄 Data Processing Pipeline

The complete data processing pipeline includes the following steps:

1. Remove the broken video files.
2. Resample the video FPS to 25, and resample the audio to 16000 Hz.
3. Scene detect via [PySceneDetect](https://github.com/Breakthrough/PySceneDetect).
4. Split each video into 5-10 second segments.
5. Affine transform the faces according to the landmarks detected by [InsightFace](https://github.com/deepinsight/insightface), then resize to 256 $\times$ 256.
6. Remove videos with [sync confidence score](https://www.robots.ox.ac.uk/~vgg/publications/2016/Chung16a/chung16a.pdf) lower than 3, and adjust the audio-visual offset to 0.
7. Calculate [hyperIQA](https://openaccess.thecvf.com/content_CVPR_2020/papers/Su_Blindly_Assess_Image_Quality_in_the_Wild_Guided_by_a_CVPR_2020_paper.pdf) score, and remove videos with scores lower than 40.

Run the script to execute the data processing pipeline:

```bash
./data_processing_pipeline.sh
```

You should change the parameter `input_dir` in the script to specify the data directory to be processed. The processed videos will be saved in the `high_visual_quality` directory. Each step will generate a new directory to prevent the need to redo the entire pipeline in case the process is interrupted by an unexpected error.

## 🏋️‍♂️ Training U-Net

Before training, you should process the data as described above. We released a pretrained SyncNet with 94% accuracy on both VoxCeleb2 and HDTF datasets for the supervision of U-Net training. You can execute the following command to download this SyncNet checkpoint:

```bash
huggingface-cli download ByteDance/LatentSync-1.6 stable_syncnet.pt --local-dir checkpoints
```

If all the preparations are complete, you can train the U-Net with the following script:

```bash
./train_unet.sh
```

We prepared several UNet configuration files in the ``configs/unet`` directory, each corresponding to a specific training setup:

- `stage1.yaml`: Stage1 training, requires **23 GB** VRAM.
- `stage2.yaml`: Stage2 training with optimal performance, requires **30 GB** VRAM.
- `stage2_efficient.yaml`: Efficient Stage 2 training, requires **20 GB** VRAM. It may lead to slight degradation in visual quality and temporal consistency compared with `stage2.yaml`, suitable for users with consumer-grade GPUs, such as the RTX 3090.
- `stage1_512.yaml`: Stage1 training on 512 $\times$ 512 resolution videos, requires **30 GB** VRAM.
- `stage2_512.yaml`: Stage2 training on 512 $\times$ 512 resolution videos, requires **55 GB** VRAM.

Also remember to change the parameters in U-Net config file to specify the data directory, checkpoint save path, and other training hyperparameters. For convenience, we prepared a script for writing a data files list. Run the following command:

```bash
python -m tools.write_fileslist
```

## 🏋️‍♂️ Training SyncNet

In case you want to train SyncNet on your own datasets, you can run the following script. The data processing pipeline for SyncNet is the same as U-Net. 

```bash
./train_syncnet.sh
```

After `validations_steps` training, the loss charts will be saved in `train_output_dir`. They contain both the training and validation loss. If you want to customize the architecture of SyncNet for different image resolutions and input frame lengths, please follow the [guide](docs/syncnet_arch.md).

## 📊 Evaluation

You can evaluate the [sync confidence score](https://www.robots.ox.ac.uk/~vgg/publications/2016/Chung16a/chung16a.pdf) of a generated video by running the following script:

```bash
./eval/eval_sync_conf.sh
```

You can evaluate the accuracy of SyncNet on a dataset by running the following script:

```bash
./eval/eval_syncnet_acc.sh
```

Note that our released SyncNet is trained on data processed through our data processing pipeline, which includes special operations such as affine transformation and audio-visual adjustment. Therefore, before evaluation, the test data must first be processed using the provided pipeline.

## 🙏 Acknowledgement

- Our code is built on [AnimateDiff](https://github.com/guoyww/AnimateDiff). 
- Some code are borrowed from [MuseTalk](https://github.com/TMElyralab/MuseTalk), [StyleSync](https://github.com/guanjz20/StyleSync), [SyncNet](https://github.com/joonson/syncnet_python), [Wav2Lip](https://github.com/Rudrabha/Wav2Lip).

Thanks for their generous contributions to the open-source community!

## 📖 Citation

If you find our repo useful for your research, please consider citing our paper:

```bibtex
@article{li2024latentsync,
  title={LatentSync: Taming Audio-Conditioned Latent Diffusion Models for Lip Sync with SyncNet Supervision},
  author={Li, Chunyu and Zhang, Chao and Xu, Weikai and Lin, Jingyu and Xie, Jinghui and Feng, Weiguo and Peng, Bingyue and Chen, Cunjian and Xing, Weiwei},
  journal={arXiv preprint arXiv:2412.09262},
  year={2024}
}
```
