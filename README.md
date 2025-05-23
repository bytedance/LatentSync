<h1 align="center">LatentSync</h1>

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2412.09262)
[![arXiv](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow)](https://huggingface.co/ByteDance/LatentSync-1.5)
[![arXiv](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Space-yellow)](https://huggingface.co/spaces/fffiloni/LatentSync)
<a href="https://replicate.com/lucataco/latentsync"><img src="https://replicate.com/lucataco/latentsync/badge" alt="Replicate"></a>

</div>

## 🔥 Updates   d

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
      <video src=https://github.com/user-attachments/assets/ff3a84da-dc9b-498a-950f-5c54f58dd5c5 controls preload></video>
    </td>
    <td>
      <video src=https://github.com/user-attachments/assets/150e00fd-381e-4421-a478-a9ea3d1212a8 controls preload></video>
    </td>
  </tr>
  <tr>
    <td>
      <video src=https://github.com/user-attachments/assets/32c830a9-4d7d-4044-9b33-b184d8e11010 controls preload></video>
    </td>
    <td>
      <video src=https://github.com/user-attachments/assets/84e4fe9d-b108-44a4-8712-13a012348145 controls preload></video>
    </td>
  </tr>
  <tr>
    <td>
      <video src=https://github.com/user-attachments/assets/7510a448-255a-44ee-b093-a1b98bd3961d controls preload></video>
    </td>
    <td>
      <video src=https://github.com/user-attachments/assets/6150c453-c559-4ae0-bb00-c565f135ff41 controls preload></video>
    </td>
  </tr>
  <tr>
    <td width=300px>
      <video src=https://github.com/user-attachments/assets/0f7f9845-68b2-4165-bd08-c7bbe01a0e52 controls preload></video>
    </td>
    <td width=300px>
      <video src=https://github.com/user-attachments/assets/c34fe89d-0c09-4de3-8601-3d01229a69e3 controls preload></video>
    </td>
  </tr>
  <tr>
    <td>
      <video src=https://github.com/user-attachments/assets/7ce04d50-d39f-4154-932a-ec3a590a8f64 controls preload></video>
    </td>
    <td>
      <video src=https://github.com/user-attachments/assets/70bde520-42fa-4a0e-b66c-d3040ae5e065 controls preload></video>
    </td>
  </tr>
</table>

(Photorealistic videos are filmed by contracted models, and anime videos are from [VASA-1](https://www.microsoft.com/en-us/research/project/vasa-1/) and [EMO](https://humanaigc.github.io/emote-portrait-alive/))

## 📑 Open-source Plan

- [x] Inference code and checkpoints
- [x] Data processing pipeline
- [x] Training code

## 🔧 Setting up the Environment

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

Or you can download `latentsync_unet.pt` and `tiny.pt` manually from our [HuggingFace repo](https://huggingface.co/ByteDance/LatentSync-1.5)

## 🚀 Inference

There are two ways to perform inference, and both require **7.8 GB** of VRAM.

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
huggingface-cli download ByteDance/LatentSync-1.5 stable_syncnet.pt --local-dir checkpoints
```

If all the preparations are complete, you can train the U-Net with the following script:

```bash
./train_unet.sh
```

We prepared three UNet configuration files in the ``configs/unet`` directory, each corresponding to a different training setup:

- `stage1.yaml`: Stage1 training, requires **23 GB** VRAM.
- `stage2.yaml`: Stage2 training with optimal performance, requires **30 GB** VRAM.
- `stage2_efficient.yaml`: Efficient Stage 2 training, requires **20 GB** VRAM. It may lead to slight degradation in visual quality and temporal consistency compared with `stage2.yaml`, suitable for users with consumer-grade GPUs, such as the RTX 3090.

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
