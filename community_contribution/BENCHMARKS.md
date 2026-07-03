# Benchmarks

## Hardware and Software

- GPU: NVIDIA GeForce RTX 5070 Ti, 16 GB
- Compute capability: 12.0
- Driver: 610.47
- Python: 3.10.13
- PyTorch: 2.7.1+cu128
- Torch CUDA: 12.8
- TorchVision: 0.22.1+cu128
- ONNX Runtime: 1.23.2
- TensorRT-RTX: 1.5.0.114
- FFmpeg: 7.1.1

Absolute runtime depends on GPU, driver, startup state, clip length, preprocessing, thermals, and background system load.

## Canonical Exact 20-Step Benchmark

Fixture: official demo video/audio, 9.68s output, 242 frames, 512 face processing, 20 inference steps, guidance scale 1.5, seed 1247.

| Backend | Wall seconds | Seconds/output second | Change vs exact20 | Peak total VRAM |
| --- | ---: | ---: | ---: | ---: |
| exact20 PyTorch eager | 397.512 | 41.065 | baseline | 15807 MB |
| optimized20 TensorRT-RTX exact | 268.160 | 27.702 | 32.54% faster | 12551 MB |
| official20 DeepCache | 272.606 | 28.162 | 31.42% faster | 15754 MB |

DeepCache is listed for context only; it is approximate and not the exact TensorRT-RTX contribution.

## Canonical Exact Quality

`optimized20` vs `exact20`:

| Metric | Result |
| --- | ---: |
| Engine cosine vs PyTorch | 0.9999949036 |
| NaN/Inf check | PASS |
| Full-frame SSIM mean | 0.981501 |
| Mouth SSIM mean | 0.972206 |
| LPIPS Alex mean | 0.000739 |
| SyncNet confidence | 8.344 |
| AV offset | 0 |

Reference exact20 SyncNet confidence was 8.578 with AV offset 0.

## Real 30.08s Validation

Private validation footage was used locally and is not included. All outputs had 752 frames, 30.08s duration, 25 fps, and audio present.

| Mode | Wall seconds | Sec/output second | Faster than exact20 | Faster than optimized20 |
| --- | ---: | ---: | ---: | ---: |
| exact20 | 1779.10 | 59.15 | baseline | n/a |
| optimized20 | 779.70 | 25.92 | 56.17% | baseline |
| optimized12 | 609.50 | 20.26 | 65.74% | 21.83% |
| optimized8 | 502.50 | 16.71 | 71.76% | 35.55% |
| optimized4 | 384.70 | 12.79 | 78.38% | 50.66% |

Quality summary vs exact20:

| Candidate | Full SSIM | Mouth SSIM | LPIPS | SyncNet confidence | AV offset | Mouth DISTS mean | Mouth landmark RMS mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| optimized20 | 0.979932 | 0.976455 | 0.000541 | 7.189 | 0 | 0.009087834 | 0.003097012 |
| optimized12 | 0.979921 | 0.976451 | 0.000534 | 7.244 | 0 | 0.009069781 | 0.003176913 |
| optimized8 | 0.979896 | 0.976468 | 0.000546 | 7.258 | 0 | 0.009120805 | 0.003247818 |
| optimized4 | 0.979694 | 0.976446 | 0.000665 | 7.382 | 0 | 0.009525675 | 0.003772854 |

Human review found no obvious naked-eye failure in the tested clips, but automated metrics show `optimized4` has the largest drift and should remain experimental.

## Hard Holdout 15.08s Validation

Private holdout footage was used locally and is not included.

| Mode | Wall seconds | Sec/output second | Faster than optimized20 |
| --- | ---: | ---: | ---: |
| optimized20 | 405.871 | 26.915 | baseline |
| optimized12 | 306.449 | 20.322 | 24.49% |
| optimized8 | 254.351 | 16.867 | 37.34% |
| optimized4 | 237.176 | 15.728 | 41.57% |

Hard holdout quality vs optimized20:

| Mode | Full SSIM | LPIPS | Mouth SSIM | DISTS mouth | Landmark mouth RMS |
| --- | ---: | ---: | ---: | ---: | ---: |
| optimized12 | 0.983261 | 0.000270 | 0.978221 | 0.007678 | 0.001593 |
| optimized8 | 0.983588 | 0.000335 | 0.978348 | 0.007785 | 0.001615 |
| optimized4 | 0.980455 | 0.003710 | 0.977751 | 0.010873 | 0.006625 |

