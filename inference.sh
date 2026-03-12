#!/bin/bash

python -m scripts.inference \
    --unet_config_path "configs/unet/stage2_512.yaml" \
    --inference_ckpt_path "checkpoints/latentsync_unet.pt" \
    --inference_steps 43 \
    --guidance_scale 2.5 \
    --seed 42 \
    --video_path "assets/demo1_video.mp4" \
    --audio_path "assets/demo1_audio.wav" \
    --video_out_path "video_out.mp4"
