#!/bin/bash

# Optional environment variables (Apple Silicon), to set when memory runs out (process killed / heavy swap):
#   LATENTSYNC_DTYPE=bfloat16      halves the memory of the models and activations (default: float32 on mps)
#   MPS_ATTENTION_BUDGET_MB=1024   memory budget of the attention scores in MB (default: 1024), lower = less memory
#   VAE_CHUNK_SIZE=4               frames per VAE batch (default: 4), lower = less memory
#
# PYTORCH_ENABLE_MPS_FALLBACK=1: fall back to CPU for ops not yet implemented on MPS

PYTORCH_ENABLE_MPS_FALLBACK=1 python -m scripts.inference \
    --unet_config_path "configs/unet/stage2.yaml" \
    --inference_ckpt_path "checkpoints/v1.5/latentsync_unet.pt" \
    --inference_steps 8 \
    --guidance_scale 1.5 \
    --enable_deepcache \
    --video_path "assets/demo1_video.mp4" \
    --audio_path "assets/demo1_audio.wav" \
    --video_out_path "video_out.mp4"