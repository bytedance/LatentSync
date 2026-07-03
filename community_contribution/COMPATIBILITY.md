# Compatibility

## Tested

- Windows 10
- NVIDIA GeForce RTX 5070 Ti, 16 GB
- Driver 610.47
- CUDA 12.8 PyTorch stack
- PyTorch 2.7.1+cu128
- TorchVision 0.22.1+cu128
- ONNX Runtime 1.23.2
- TensorRT-RTX 1.5.0.114
- Official ByteDance LatentSync 1.6 checkpoint

## Expected Requirements

- NVIDIA RTX GPU supported by TensorRT-RTX.
- Fixed 512x512 LatentSync UNet config.
- Exported static ONNX UNet graph.
- Built local TensorRT-RTX engine matching the current GPU/software stack.

Compiled TensorRT engines are not portable across all machines and are intentionally not included.

## Python Packages

The optimized script uses the existing LatentSync dependency stack plus:

- `tensorrt-rtx`
- `pynvml` for optional telemetry
- `kornia` for the optional GPU restore helper

## Fallback

The optimized script keeps a staged/PyTorch-compatible path for comparison and fallback. Upstream maintainers may prefer integrating the TensorRT-RTX runner into the existing inference path instead of accepting a parallel script.

## Known Limits

- Tested mainly on RTX 5070 Ti / Blackwell.
- Linux should be possible, but this validation was Windows-based.
- Reduced-step modes are approximate.
- More validation is needed on diverse public footage before broad quality claims.

