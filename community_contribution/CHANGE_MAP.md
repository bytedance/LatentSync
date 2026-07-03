# Change Map

## Base

- Upstream: `https://github.com/bytedance/LatentSync`
- Base branch: `main`
- Base commit: `a229c3948406bc2cf6eaf4873e662e70c6a04746`
- Local development commits mapped from the validation branch:
  - `cc791ae` Add LatentSync optimization runner
  - `93ae664` Add TensorRT-RTX UNet backend
  - `72c5b5c` Improve TensorRT-RTX execution path
  - `e2f93dd` Avoid per-block cache flushes in optimized inference
  - `761e8b9` Add optimized batch runner
  - `ed430e5` Reuse aligned faces in optimized restore
  - `e59e136` Add scheduler experiment option

## Public Files

- `scripts/inference_optimized.py`
- `scripts/batch_inference_optimized.py`
- `community_contribution/*`
- `docs/tensorrt_rtx_optimized_inference.md`

## Exact 20-Step Path

| Optimization | Source area | Evidence |
| --- | --- | --- |
| TensorRT-RTX UNet runner | `TensorRTRTXUNetRunner` | Engine cosine 0.9999949036 vs PyTorch |
| Persistent staging buffers | TensorRT runner input/output buffers | Reduced per-call overhead and enabled graph replay |
| CUDA Graph replay | TensorRT runner graph capture | Direct engine execution stabilized around 0.5s/call |
| Removed per-block `gc.collect()` / `torch.cuda.empty_cache()` | denoise/decode hot loops | Decode/restore phase fell from 141.219s to 26.699s in canonical profiling |
| Phase-boundary cleanup | phase transitions | Preserves cleanup without per-block hot-loop penalty |
| Release TensorRT runner before VAE decode | before decode/restore | Reduced VRAM pressure before phase C |
| Aligned-face reuse | restore path | Small exact cleanup; quality gates passed |
| Batch runner | `batch_inference_optimized.py` | Engine load paid once for a batch |

## Approximate Reduced-Step Modes

`optimized12`, `optimized8`, and `optimized4` reuse the same checkpoint and backend but reduce denoising steps. They are useful research/production-trial modes, not exact replacements for `exact20`.

## Local-Only Work Excluded

- PySide6 production Studio GUI.
- SQLite queue and cleanup tooling.
- Windows local launchers.
- Private validation media and generated reports.
- TensorRT engine binaries.

