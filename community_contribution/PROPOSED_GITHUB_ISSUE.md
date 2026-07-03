# Proposed GitHub Issue

Title:

`TensorRT-RTX exact 20-step speedup and reduced-step validation results for LatentSync 1.6`

Body:

Hi maintainers,

I would like to share a tested LatentSync 1.6 optimization path and ask how you would prefer it to be contributed upstream.

This work was directed, tested, and validated by Petrus Vermaak. OpenAI Codex served as the implementation and engineering agent under my direction: it implemented the code, ran profiling, prepared validation scripts, and organized the evidence while I drove the requirements, testing decisions, hardware validation, and human video review.

## Related existing threads

This appears related to:

- #94 Can the inference speed be faster?
- #214 DeepCache acceleration question
- #226 How to speed up inference?
- #265 Slow inference
- #290 DPM-Solver++ / reduced-step scheduler discussion
- #313 TensorRT consumer GPU speed question
- #326 Longer video / memory growth
- #285 / #288 memory or output failure reports where repeated jobs and cleanup behavior matter

## Exact 20-step TensorRT-RTX path

The strongest result is an exact 20-step TensorRT-RTX path for the UNet. It keeps:

- official LatentSync 1.6 checkpoint
- 512x512 face processing
- 25 fps
- 16-frame temporal blocks
- 20 denoising steps
- guidance scale 1.5
- FP16 TensorRT-RTX engine
- RTX 5070 Ti 16 GB test system

Key implementation ideas:

- static TensorRT-RTX UNet engine
- persistent input/output staging buffers
- CUDA Graph reuse
- no `gc.collect()` / `torch.cuda.empty_cache()` inside hot per-block loops
- cleanup retained at phase boundaries
- release TensorRT runner before VAE decode/restore
- aligned-face reuse during restore
- optional batch runner that reuses a preloaded TensorRT runner across jobs

## Canonical benchmark

Fixture: official demo video/audio, 9.68s output, 242 frames, 512 face processing, 20 steps, guidance 1.5.

| Backend | Wall seconds | Sec/output second | Change vs exact20 | Peak total VRAM |
| --- | ---: | ---: | ---: | ---: |
| exact20 PyTorch eager | 397.512 | 41.065 | baseline | 15807 MB |
| optimized20 TensorRT-RTX exact | 268.160 | 27.702 | 32.54% faster | 12551 MB |

Quality vs exact20:

| Metric | Result |
| --- | ---: |
| Engine cosine vs PyTorch | 0.9999949036 |
| Full-frame SSIM mean | 0.981501 |
| Mouth SSIM mean | 0.972206 |
| LPIPS Alex mean | 0.000739 |
| SyncNet confidence | 8.344 |
| AV offset | 0 |

The exact20 reference SyncNet confidence was 8.578 with AV offset 0.

## Real-footage validation

I also tested private real footage locally. The media cannot be shared, but the validation was useful for checking that the result was not only a demo-fixture artifact.

30.08s validation clip:

| Mode | Wall seconds | Sec/output second | Faster than exact20 |
| --- | ---: | ---: | ---: |
| exact20 | 1779.10 | 59.15 | baseline |
| optimized20 | 779.70 | 25.92 | 56.17% |
| optimized12 | 609.50 | 20.26 | 65.74% |
| optimized8 | 502.50 | 16.71 | 71.76% |
| optimized4 | 384.70 | 12.79 | 78.38% |

All candidates had SyncNet AV offset 0. `optimized12` and `optimized8` stayed close in SSIM/LPIPS/DISTS/landmark checks. `optimized4` was visually plausible in this test but showed the largest objective drift, so I would not present it as equivalent quality.

## Reduced-step modes are separate

The reduced-step modes are useful, but they are approximate:

- `optimized12`
- `optimized8`
- `optimized4`
- DPM-Solver multistep experiments

They change the number of denoising steps and should not be mixed with the exact TensorRT-RTX speedup claim.

## Negative findings

These may save others time:

- 24 frames is the hard temporal-position maximum for the tested checkpoint/config.
- 25 through 31 frames fail.
- Valid 22-frame and 24-frame engines were slower than the 16-frame engine.
- Audio-only cacheable work was too small to prioritize.
- TensorRT builder `opt5` and full tiling were slower.
- Direct input binding and skip-zero staging were slower.
- ONNX Runtime Stable Diffusion fusions produced contrib ops unsuitable for TensorRT-RTX.
- GPU erosion had too little gain for its memory cost.
- Per-block cache flushing was a major full-pipeline bottleneck.
- Batch engine reuse helps repeated jobs by avoiding repeated engine load cost.

## Code branch

I have a clean public branch prepared with no private media, no model weights, no compiled engines, and no machine-specific paths.

I will post the pull request link here once the branch is available on GitHub. The intended PR is a focused contribution containing the TensorRT-RTX optimized inference path, batch runner, benchmark summary, compatibility notes, and privacy/license audit.
