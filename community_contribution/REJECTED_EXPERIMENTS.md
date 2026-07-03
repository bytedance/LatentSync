# Rejected Experiments

These findings are included to save other developers from repeating low-ceiling branches.

## Larger Temporal Blocks

The tested checkpoint/config has a hard temporal-position limit of 24 frames. Frame counts 25 through 31 failed with the same temporal-position mismatch. Valid 22-frame and 24-frame TensorRT-RTX engines were built and numerically validated, but projected UNet time was slower than the current 16-frame production engine.

Decision: keep the 16-frame engine.

## Audio K/V Caching

Static dependency analysis found pure audio-only work at only 0.129% of timed direct engine work.

Decision: not a priority unless a larger graph split makes it free.

## TensorRT Builder `opt5` and Full Tiling

Both builder variants were slower than the production engine in direct profile:

- `opt5_fulltile`: about 1.18% slower
- `opt5`: about 1.02% slower

Decision: reject.

## Direct Input Binding

Directly rebinding input addresses produced a valid smoke render but slower phase-B UNet timing. It likely interfered with stable CUDA Graph reuse.

Decision: keep staging buffers.

## Skip-Zero Staging

Produced a valid smoke render but slower timing than the no-flush path.

Decision: reject for now.

## ONNX Runtime Fused Graph

ONNX Runtime's Stable Diffusion optimizer produced graph fusions such as `NhwcConv` and `SkipLayerNormalization`, but those are ONNX Runtime contrib ops and were not parseable by TensorRT-RTX. Running the fused graph directly with ONNX Runtime CUDA was numerically close but far too slow for this workload.

Decision: useful directionally, not a drop-in backend.

## GPU Erosion for Restore

Kornia GPU erosion was only marginally faster than CPU erosion in a micro-benchmark and used about 4.8 GiB peak allocation.

Decision: not a safe default.

## Per-Block Cache Flushing

Per-block `gc.collect()` and `torch.cuda.empty_cache()` were a severe performance penalty. Removing them while keeping cleanup at phase boundaries produced the largest exact full-pipeline gain.

Decision: accepted.

