# TensorRT-RTX Optimized Inference

This document summarizes an experimental optimized inference path for LatentSync 1.6.

The exact mode keeps 20 denoising steps, guidance scale 1.5, 512x512 face processing, and 16-frame temporal blocks. It replaces the UNet execution path with a static FP16 TensorRT-RTX engine and uses persistent staging buffers plus CUDA Graph replay.

## Files

- `scripts/inference_optimized.py`
- `scripts/batch_inference_optimized.py`

## Engine

The TensorRT engine is not included. Build it locally for the target GPU/software stack.

The tested engine was built for the 512x16 LatentSync UNet shape and validated numerically against PyTorch. Compiled TensorRT engines should not be treated as universally portable artifacts.

## Example

```bash
export LATENTSYNC_ROOT="$(pwd)"
python -m scripts.inference_optimized \
  --unet_config_path configs/unet/stage2_512.yaml \
  --inference_ckpt_path checkpoints/latentsync_unet.pt \
  --video_path assets/demo1_video.mp4 \
  --audio_path assets/demo1_audio.wav \
  --video_out_path video_out_optimized20.mp4 \
  --inference_steps 20 \
  --guidance_scale 1.5 \
  --seed 1247 \
  --backend tensorrt-rtx \
  --tensorrt_engine_path engines/latentsync_unet_512x16_cfg_fp16.trt_rtx_1_5.plan
```

## Batch Runner

The batch runner loads the pipeline and TensorRT runner once, then processes a JSON job list. This is useful when multiple clips are processed in one session because engine load is paid once.

## Reduced Steps

Fewer-step modes are supported for experiments, but they are approximate. Do not present `optimized12`, `optimized8`, or `optimized4` as identical to exact 20-step output.

## Known Limitations

- Validated mainly on RTX 5070 Ti / Blackwell.
- Requires TensorRT-RTX runtime support.
- Linux should be possible, but the validation described here was Windows-based.
- More public-fixture validation is needed before broad production claims.

