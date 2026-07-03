# Technical Summary

## Scope

This package shares an exact 20-step TensorRT-RTX inference path for ByteDance LatentSync 1.6, tested against the official 512x512 checkpoint and the upstream `main` branch at commit `a229c3948406bc2cf6eaf4873e662e70c6a04746`.

The clean public branch excludes private media, local GUI tooling, local launchers, databases, generated engines, model weights, caches, and machine-specific paths.

## Exact Contribution

The strongest upstream contribution is `optimized20`, an exact 20-step path that keeps:

- 512x512 face processing
- 25 fps output
- 16-frame temporal blocks
- 20 denoising steps
- guidance scale 1.5
- FP16 TensorRT-RTX UNet engine
- PyTorch fallback/reference path

The implementation adds:

- `scripts/inference_optimized.py`
- `scripts/batch_inference_optimized.py`

Important exact optimizations:

- Static TensorRT-RTX UNet engine execution.
- Persistent input/output staging buffers.
- CUDA Graph capture/replay for repeated UNet calls.
- Cleanup moved out of hot per-block loops.
- TensorRT runner released before VAE decode/restore.
- Aligned-face reuse during restore.
- Batch runner that preloads the TensorRT runner once across jobs.

## Approximate Modes

The branch also exposes reduced-step experiments separately:

- `optimized12`
- `optimized8`
- `optimized4`
- DPM-Solver multistep as an opt-in scheduler experiment

These modes change denoising step count. They are approximate, not identical to `exact20`, and should not be presented as the same exact-quality speedup.

## What Is Not Included

- Private validation media.
- Compiled TensorRT engines.
- Model weights.
- GUI/job queue tooling.
- Local Windows launcher scripts.
- Benchmark media.
- Machine-specific caches.

