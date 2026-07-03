# Proposed Pull Request

Title:

`Add TensorRT-RTX optimized inference path`

Body:

## Summary

Adds a TensorRT-RTX optimized inference path for LatentSync 1.6:

- `scripts/inference_optimized.py`
- `scripts/batch_inference_optimized.py`
- public technical notes and benchmark summary

This PR is intentionally scoped to code and documentation that can be shared publicly. It excludes local production GUI tooling, private validation media, generated benchmark media, model weights, compiled TensorRT engines, caches, and machine-specific paths.

## Attribution

This optimization effort was directed, tested, and validated by Petrus Vermaak. OpenAI Codex served as the implementation and engineering agent under his direction.

## Exact 20-step result

On an RTX 5070 Ti 16 GB test system, using the official LatentSync 1.6 checkpoint, 512 face processing, 16-frame temporal blocks, 20 denoising steps, and guidance scale 1.5:

| Backend | Wall seconds | Sec/output second | Change vs exact20 | Peak total VRAM |
| --- | ---: | ---: | ---: | ---: |
| exact20 PyTorch eager | 397.512 | 41.065 | baseline | 15807 MB |
| optimized20 TensorRT-RTX exact | 268.160 | 27.702 | 32.54% faster | 12551 MB |

Quality vs exact20:

- engine cosine vs PyTorch: 0.9999949036
- full-frame SSIM mean: 0.981501
- mouth SSIM mean: 0.972206
- LPIPS Alex mean: 0.000739
- SyncNet confidence: 8.344
- AV offset: 0

## Approximate modes

Reduced-step modes (`optimized12`, `optimized8`, `optimized4`) are included as experiments. They reduce denoising step count and are not exact replacements for `exact20`.

## Notes for maintainers

This is a focused contribution for the TensorRT-RTX optimized inference path, batch runner, benchmark summary, compatibility notes, and privacy/license audit. The reduced-step modes are included as experimental speed presets and are documented separately from the exact 20-step path.

## Privacy / licensing

- No private footage.
- No model weights.
- No compiled TensorRT engines.
- No local caches.
- No absolute local machine paths.
- Upstream license is Apache 2.0; this contribution is intended for upstream inclusion under the same terms.
