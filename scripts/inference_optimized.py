import argparse
import gc
import json
import math
import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import cv2
import numpy as np
import psutil
import soundfile as sf
import torch
import torchvision
import tqdm
import kornia
from diffusers import AutoencoderKL, DDIMScheduler, DPMSolverMultistepScheduler, PNDMScheduler
from einops import rearrange
from omegaconf import OmegaConf
from torchvision import transforms

from accelerate.utils import set_seed
from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from latentsync.utils.image_processor import ImageProcessor, load_fixed_mask
from latentsync.utils.util import check_ffmpeg_installed, read_audio, read_video, write_video
from latentsync.whisper.audio2feature import Audio2Feature

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def repo_root():
    return Path(__file__).resolve().parents[1]


class Telemetry:
    def __init__(self):
        self.proc = psutil.Process(os.getpid())
        self.nvml = None
        self.handle = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self.nvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
        except Exception:
            self.nvml = None
            self.handle = None

    def sample(self):
        rss = self.proc.memory_info().rss / 1024**2
        private = getattr(self.proc.memory_full_info(), "private", 0) / 1024**2
        gpu = {}
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            gpu.update(
                {
                    "torch_allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                    "torch_reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                    "torch_max_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
                    "torch_max_reserved_mb": torch.cuda.max_memory_reserved() / 1024**2,
                    "cuda_free_mb": free / 1024**2,
                    "cuda_total_mb": total / 1024**2,
                }
            )
        if self.nvml and self.handle:
            try:
                mem = self.nvml.nvmlDeviceGetMemoryInfo(self.handle)
                util = self.nvml.nvmlDeviceGetUtilizationRates(self.handle)
                power = self.nvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
                clock = self.nvml.nvmlDeviceGetClockInfo(self.handle, self.nvml.NVML_CLOCK_GRAPHICS)
                gpu.update(
                    {
                        "nvml_used_mb": mem.used / 1024**2,
                        "nvml_free_mb": mem.free / 1024**2,
                        "gpu_util_percent": util.gpu,
                        "gpu_mem_util_percent": util.memory,
                        "gpu_power_w": power,
                        "gpu_graphics_clock_mhz": clock,
                    }
                )
            except Exception as exc:
                gpu["nvml_error"] = repr(exc)
        return {"rss_mb": rss, "private_mb": private, **gpu}


class Profile:
    def __init__(self):
        self.telemetry = Telemetry()
        self.events = []
        self.peaks = {}
        self.started_at = time.perf_counter()

    @contextmanager
    def section(self, name, cuda=False):
        if cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            start_event = end_event = None
        start = time.perf_counter()
        start_sample = self.telemetry.sample()
        try:
            yield
        finally:
            if end_event is not None:
                end_event.record()
                torch.cuda.synchronize()
                cuda_ms = start_event.elapsed_time(end_event)
            else:
                cuda_ms = None
                if cuda and torch.cuda.is_available():
                    torch.cuda.synchronize()
            end = time.perf_counter()
            end_sample = self.telemetry.sample()
            event = {
                "name": name,
                "wall_seconds": end - start,
                "cuda_ms": cuda_ms,
                "start": start_sample,
                "end": end_sample,
            }
            self.events.append(event)
            self._merge_peak(start_sample)
            self._merge_peak(end_sample)

    def mark(self, name):
        sample = self.telemetry.sample()
        self.events.append({"name": name, "wall_seconds": 0.0, "sample": sample})
        self._merge_peak(sample)

    def _merge_peak(self, sample):
        for key, value in sample.items():
            if isinstance(value, (int, float)):
                self.peaks[key] = max(self.peaks.get(key, value), value)

    def summary(self):
        total = time.perf_counter() - self.started_at
        by_name = {}
        for event in self.events:
            by_name[event["name"]] = by_name.get(event["name"], 0.0) + event.get("wall_seconds", 0.0)
        return {"total_wall_seconds": total, "sections": self.events, "section_totals": by_name, "peaks": self.peaks}


class TensorOnlyUNet(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, latent_input, timestep, audio_conditioning):
        return self.unet(
            latent_input,
            timestep,
            encoder_hidden_states=audio_conditioning,
            return_dict=False,
        )[0]


def add_tensorrt_rtx_runtime_paths(root=None):
    root = Path(root or os.environ.get("LATENTSYNC_ROOT") or repo_root())
    paths = [
        root / "env" / "Lib" / "site-packages" / "tensorrt_rtx_libs",
        root / "env" / "Lib" / "site-packages" / "torch" / "lib",
        root / "env" / "Library" / "bin",
        root / "env" / "Scripts",
    ]
    for path in paths:
        if path.exists():
            try:
                os.add_dll_directory(str(path))
            except (AttributeError, OSError):
                pass
    os.environ["PATH"] = os.pathsep.join(str(p) for p in paths if p.exists()) + os.pathsep + os.environ.get("PATH", "")


class TensorRTRTXUNetRunner:
    def __init__(self, engine_path, root=None):
        add_tensorrt_rtx_runtime_paths(root)
        import tensorrt_rtx as trt

        self.trt = trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine_path = str(engine_path)
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Could not deserialize TensorRT-RTX engine: {engine_path}")
        runtime_config = self.engine.create_runtime_config()
        runtime_config.cuda_graph_strategy = trt.CudaGraphStrategy.WHOLE_GRAPH_CAPTURE
        self.context = self.engine.create_execution_context(runtime_config)
        self.stream = torch.cuda.Stream()
        self.input = torch.empty((2, 13, 16, 64, 64), device="cuda", dtype=torch.float16)
        self.audio = torch.empty((32, 50, 384), device="cuda", dtype=torch.float16)
        self.output = torch.empty((2, 4, 16, 64, 64), device="cuda", dtype=torch.float16)
        self.timestep = torch.empty((), device="cuda", dtype=torch.int64)

    def _prepare_inputs(self, latent_input, audio_conditioning):
        frame_count = latent_input.shape[2]
        if latent_input.shape[0] != 2 or latent_input.shape[1] != 13 or frame_count > 16:
            raise ValueError(f"TensorRT-RTX UNet expects latent_input shape [2,13,<=16,64,64], got {list(latent_input.shape)}")
        if audio_conditioning.shape[0] > 32:
            raise ValueError(f"TensorRT-RTX UNet expects <=32 audio conditioning rows, got {list(audio_conditioning.shape)}")
        self.input.zero_()
        self.audio.zero_()
        self.input[:, :, :frame_count].copy_(latent_input)
        self.audio[: audio_conditioning.shape[0]].copy_(audio_conditioning)
        return self.input, self.audio, frame_count

    def __call__(self, latent_input, timestep, audio_conditioning):
        latent_input, audio_conditioning, frame_count = self._prepare_inputs(latent_input, audio_conditioning)
        if torch.is_tensor(timestep):
            self.timestep.copy_(timestep.to(device="cuda", dtype=torch.int64))
        else:
            self.timestep.fill_(int(timestep))
        bindings = {
            "latent_input": latent_input,
            "timestep": self.timestep,
            "audio_conditioning": audio_conditioning,
            "noise_pred": self.output,
        }
        for name, tensor in bindings.items():
            ok = self.context.set_tensor_address(name, int(tensor.data_ptr()))
            if ok is False:
                raise RuntimeError(f"Failed to bind TensorRT-RTX tensor address for {name}")
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            ok = self.context.execute_async_v3(self.stream.cuda_stream)
        if ok is False:
            raise RuntimeError("TensorRT-RTX execute_async_v3 returned False")
        self.stream.synchronize()
        return self.output[:, :, :frame_count]


def ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def run_ffprobe(path):
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-show_streams",
        "-of",
        "json",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return {"error": result.stderr}
    return json.loads(result.stdout)


def build_scheduler(args):
    if args.scheduler_type == "ddim":
        return DDIMScheduler.from_pretrained("configs")
    if args.scheduler_type == "dpm_solver_multistep":
        return DPMSolverMultistepScheduler.from_pretrained("configs")
    if args.scheduler_type == "pndm":
        return PNDMScheduler.from_pretrained("configs")
    raise ValueError(f"Unsupported scheduler_type: {args.scheduler_type}")


def build_pipeline(config, args, dtype):
    scheduler = build_scheduler(args)
    if config.model.cross_attention_dim == 768:
        whisper_model_path = "checkpoints/whisper/small.pt"
    elif config.model.cross_attention_dim == 384:
        whisper_model_path = "checkpoints/whisper/tiny.pt"
    else:
        raise NotImplementedError("cross_attention_dim must be 768 or 384")

    audio_encoder = Audio2Feature(
        model_path=whisper_model_path,
        device="cuda",
        num_frames=config.data.num_frames,
        audio_feat_length=config.data.audio_feat_length,
    )
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0
    unet, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(config.model),
        args.inference_ckpt_path,
        device="cpu",
    )
    unet = unet.to(dtype=dtype)
    return LipsyncPipeline(vae=vae, audio_encoder=audio_encoder, unet=unet, scheduler=scheduler).to("cuda")


def make_audio_embeds(whisper_chunks, start, count, device, dtype, do_cfg):
    audio_embeds = torch.stack(whisper_chunks[start : start + count]).to(device, dtype=dtype)
    if do_cfg:
        null_audio_embeds = torch.zeros_like(audio_embeds)
        audio_embeds = torch.cat([null_audio_embeds, audio_embeds])
    return audio_embeds


def prepare_aligned_masks_and_masked_images(pipe, aligned_faces):
    if isinstance(aligned_faces, np.ndarray):
        aligned_faces = torch.from_numpy(aligned_faces)
    if aligned_faces.shape[-1] == 3:
        aligned_faces = rearrange(aligned_faces, "f h w c -> f c h w")
    if aligned_faces.shape[-2:] != (pipe.image_processor.resolution, pipe.image_processor.resolution):
        return pipe.image_processor.prepare_masks_and_masked_images(aligned_faces, affine_transform=False)

    pixel_values = aligned_faces.to(dtype=torch.float32) / 255.0
    pixel_values = pipe.image_processor.normalize(pixel_values)
    mask_image = pipe.image_processor.mask_image
    masked_pixel_values = pixel_values * mask_image
    masks = mask_image[0:1].expand(pixel_values.shape[0], -1, -1, -1)
    return pixel_values, masked_pixel_values, masks


def prepare_block_latents(pipe, faces, start, count, height, width, dtype, device, generator, do_cfg, profile):
    inference_faces = faces[start : start + count]
    with torch.inference_mode():
        with profile.section("mask_and_reference_pixel_prep"):
            ref_pixel_values, masked_pixel_values, masks = prepare_aligned_masks_and_masked_images(pipe, inference_faces)
        with profile.section("vae_masked_frame_encoding", cuda=True):
            mask_latents, masked_image_latents = pipe.prepare_mask_latents(
                masks, masked_pixel_values, height, width, dtype, device, generator, do_cfg
            )
        with profile.section("vae_reference_encoding", cuda=True):
            ref_latents = pipe.prepare_image_latents(ref_pixel_values, device, dtype, generator, do_cfg)
    block = {
        "mask_latents": mask_latents.detach().cpu(),
        "masked_image_latents": masked_image_latents.detach().cpu(),
        "ref_latents": ref_latents.detach().cpu(),
        "frame_count": count,
    }
    del mask_latents, masked_image_latents, ref_latents, ref_pixel_values, masked_pixel_values, masks
    gc.collect()
    torch.cuda.empty_cache()
    return block


def align_faces_from_matrices(pipe, video_frames, affine_matrices, start, count, dtype):
    faces = []
    restorer = pipe.image_processor.restorer
    for frame, affine_matrix in zip(video_frames[start : start + count], affine_matrices[start : start + count]):
        face_tensor = torch.from_numpy(frame).to(device=restorer.device, dtype=dtype)
        face_tensor = rearrange(face_tensor, "h w c -> c h w").unsqueeze(0)
        cropped = kornia.geometry.transform.warp_affine(
            face_tensor,
            affine_matrix.to(device=restorer.device, dtype=dtype),
            (restorer.face_size[1], restorer.face_size[0]),
            mode="bilinear",
            padding_mode="fill",
            fill_value=restorer.fill_value,
        )
        cropped = torchvision.transforms.functional.resize(
            cropped.squeeze(0),
            size=(pipe.image_processor.resolution, pipe.image_processor.resolution),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True,
        )
        faces.append(cropped.cpu().to(torch.uint8))
    return torch.stack(faces)


def preallocate_static_inputs(latents, mask_latents, masked_image_latents, ref_latents):
    batch, channels, frames, height, width = mask_latents.shape
    cfg_latents = torch.empty((batch, latents.shape[1], frames, height, width), device=latents.device, dtype=latents.dtype)
    unet_input = torch.empty((batch, 13, frames, height, width), device=latents.device, dtype=latents.dtype)
    unet_input[:, 4:5].copy_(mask_latents)
    unet_input[:, 5:9].copy_(masked_image_latents)
    unet_input[:, 9:13].copy_(ref_latents)
    return cfg_latents, unet_input


def make_repeated_base_latents(pipe, frame_count, height, width, dtype, device, base_latent=None):
    if base_latent is None:
        shape = (1, pipe.vae.config.latent_channels, 1, height // pipe.vae_scale_factor, width // pipe.vae_scale_factor)
        base_latent = torch.randn(shape, device=device, dtype=dtype) * pipe.scheduler.init_noise_sigma
    return base_latent.repeat(1, 1, frame_count, 1, 1), base_latent


def denoise_block(
    pipe,
    latents,
    timesteps,
    audio_embeds,
    mask_latents,
    masked_image_latents,
    ref_latents,
    guidance_scale,
    extra_step_kwargs,
    do_cfg,
    profile,
    backend,
    compiled_wrapper=None,
    tensorrt_runner=None,
    capture=None,
):
    unet_step_latencies = []
    unet_diffs = []
    static_buffers = backend in {"static", "compile-default", "compile-autotune", "compile-aot-eager", "tensorrt-rtx"}
    if static_buffers:
        cfg_latents, unet_input = preallocate_static_inputs(latents, mask_latents, masked_image_latents, ref_latents)

    with torch.inference_mode():
        for j, t in enumerate(tqdm.tqdm(timesteps, desc=f"{backend} denoise")):
            if static_buffers:
                cfg_latents[0].copy_(latents[0])
                if do_cfg:
                    cfg_latents[1].copy_(latents[0])
                scaled = pipe.scheduler.scale_model_input(cfg_latents, t)
                unet_input[:, 0:4].copy_(scaled)
            else:
                unet_input = torch.cat([latents] * 2) if do_cfg else latents
                unet_input = pipe.scheduler.scale_model_input(unet_input, t)
                unet_input = torch.cat([unet_input, mask_latents, masked_image_latents, ref_latents], dim=1)

            if capture is not None and j == 0:
                capture["unet_input"] = unet_input.detach().clone()
                capture["timestep"] = t.detach().clone() if torch.is_tensor(t) else torch.tensor(t, device=latents.device)
                capture["audio_embeds"] = audio_embeds.detach().clone()

            if static_buffers and capture is not None and j == 0:
                original_input = torch.cat([latents] * 2) if do_cfg else latents
                original_input = pipe.scheduler.scale_model_input(original_input, t)
                original_input = torch.cat([original_input, mask_latents, masked_image_latents, ref_latents], dim=1)
                diff = (original_input - unet_input).abs()
                unet_diffs.append(
                    {
                        "input_equivalence_max_abs": float(diff.max().item()),
                        "input_equivalence_mean_abs": float(diff.mean().item()),
                    }
                )

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            if tensorrt_runner is not None:
                noise_pred = tensorrt_runner(unet_input, t, audio_embeds)
            elif compiled_wrapper is not None:
                noise_pred = compiled_wrapper(unet_input, t, audio_embeds)
            else:
                noise_pred = pipe.unet(
                    unet_input,
                    t,
                    encoder_hidden_states=audio_embeds,
                    return_dict=False,
                )[0]
            end_event.record()
            torch.cuda.synchronize()
            unet_step_latencies.append(start_event.elapsed_time(end_event))

            if do_cfg:
                noise_pred_uncond, noise_pred_audio = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_audio - noise_pred_uncond)
            latents = pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
    profile.events.append(
        {
            "name": "unet_step_latency_ms",
            "values": unet_step_latencies,
            "average_ms": float(np.mean(unet_step_latencies)),
            "p50_ms": float(np.percentile(unet_step_latencies, 50)),
            "p95_ms": float(np.percentile(unet_step_latencies, 95)),
        }
    )
    if unet_diffs:
        profile.events.append({"name": "static_buffer_correctness", **unet_diffs[0]})
    return latents


def restore_and_write_streaming(pipe, decoded_block, video_frames, boxes, affine_matrices, start, temp_video_path, writer):
    print(f"Restoring block {start}..{start + len(decoded_block) - 1}")
    for offset, face in enumerate(decoded_block):
        idx = start + offset
        x1, y1, x2, y2 = boxes[idx]
        height = int(y2 - y1)
        width = int(x2 - x1)
        face = torchvision.transforms.functional.resize(
            face, size=(height, width), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True
        )
        out_frame = pipe.image_processor.restorer.restore_img(video_frames[idx], face, affine_matrices[idx])
        writer.append_data(out_frame)


def run_profiled(pipe, args, config, backend, profile, preloaded_tensorrt_runner=None):
    check_ffmpeg_installed()
    device = pipe._execution_device
    dtype = torch.float16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7 else torch.float32
    height = config.data.resolution
    width = config.data.resolution
    num_frames = config.data.num_frames
    do_cfg = args.guidance_scale > 1.0
    generator = None

    pipe.unet.eval()
    pipe.scheduler.set_timesteps(args.inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, 0.0)
    mask_image = load_fixed_mask(height, config.data.mask_image_path)
    pipe.image_processor = ImageProcessor(height, device="cuda", mask_image=mask_image)
    pipe.vae.to(device=device, dtype=dtype)
    if hasattr(pipe.audio_encoder, "model"):
        pipe.audio_encoder.model.to(device)
    if backend != "tensorrt-rtx":
        pipe.unet.to(device=device, dtype=dtype)

    with profile.section("whisper_feature_extraction", cuda=True):
        whisper_feature = pipe.audio_encoder.audio2feat(args.audio_path)
        whisper_chunks = pipe.audio_encoder.feature2chunks(feature_array=whisper_feature, fps=25)
    with profile.section("audio_loading_resampling"):
        audio_samples = read_audio(args.audio_path)
    with profile.section("input_video_conversion"):
        video_frames = read_video(args.video_path, use_decord=False)
    with profile.section("face_detection_landmarks_alignment", cuda=True):
        video_frames, faces, boxes, affine_matrices = pipe.loop_video(whisper_chunks, video_frames)

    if backend == "profile-original":
        return run_original_profiled(
            pipe,
            args,
            profile,
            video_frames,
            faces,
            boxes,
            affine_matrices,
            audio_samples,
            whisper_chunks,
            timesteps,
            extra_step_kwargs,
            height,
            width,
            dtype,
            device,
            generator,
            do_cfg,
        )

    block_records = []
    num_inferences = math.ceil(len(whisper_chunks) / num_frames)
    with profile.section("phase_a_preprocess_encode", cuda=True):
        for i in tqdm.tqdm(range(num_inferences), desc="Phase A encode"):
            start = i * num_frames
            count = min(num_frames, len(whisper_chunks) - start)
            block = prepare_block_latents(pipe, faces, start, count, height, width, dtype, device, generator, do_cfg, profile)
            block["audio_embeds"] = make_audio_embeds(whisper_chunks, start, count, "cpu", dtype, do_cfg)
            block["start"] = start
            block_records.append(block)

    pipe.vae.to("cpu")
    if hasattr(pipe.audio_encoder, "model"):
        pipe.audio_encoder.model.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()
    profile.mark("phase_a_boundary_after_unload")

    compile_info = None
    compiled_wrapper = None
    tensorrt_runner = None
    capture = {} if backend in {"static", "compile-default", "compile-autotune", "onnx", "tensorrt"} else None
    if backend in {"compile-default", "compile-autotune", "compile-aot-eager"}:
        wrapper = TensorOnlyUNet(pipe.unet).to(device).eval()
        mode = None if backend in {"compile-default", "compile-aot-eager"} else "max-autotune-no-cudagraphs"
        compile_started = time.perf_counter()
        try:
            if backend == "compile-aot-eager":
                compiled_wrapper = torch.compile(wrapper, backend="aot_eager", dynamic=False)
                compile_mode = "aot_eager"
            else:
                compiled_wrapper = torch.compile(wrapper, dynamic=False, mode=mode)
                compile_mode = mode or "default"
            compile_info = {
                "status": "compiled",
                "mode": compile_mode,
                "compile_call_seconds": time.perf_counter() - compile_started,
            }
        except Exception as exc:
            compile_info = {"status": "compile_failed", "mode": mode or "default", "error": repr(exc)}
            raise

    if backend in {"onnx", "tensorrt"}:
        return export_or_tensorrt_probe(pipe, args, profile, block_records[0], timesteps[0], device, dtype, backend)

    if backend == "tensorrt-rtx":
        pipe.unet.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
        if preloaded_tensorrt_runner is None:
            with profile.section("tensorrt_rtx_engine_load", cuda=True):
                tensorrt_runner = TensorRTRTXUNetRunner(args.tensorrt_engine_path)
        else:
            tensorrt_runner = preloaded_tensorrt_runner
            profile.mark("tensorrt_rtx_engine_reused")

    decoded_blocks = []
    latent_blocks = []
    base_latent = None
    with profile.section("phase_b_unet_only_inference", cuda=True):
        for block in block_records:
            count = block["frame_count"]
            latents, base_latent = make_repeated_base_latents(pipe, count, height, width, dtype, device, base_latent)
            audio_embeds = block["audio_embeds"].to(device=device, dtype=dtype)
            mask_latents = block["mask_latents"].to(device=device, dtype=dtype)
            masked_image_latents = block["masked_image_latents"].to(device=device, dtype=dtype)
            ref_latents = block["ref_latents"].to(device=device, dtype=dtype)
            pipe.scheduler.set_timesteps(args.inference_steps, device=device)
            block_timesteps = pipe.scheduler.timesteps
            latents = denoise_block(
                pipe,
                latents,
                block_timesteps,
                audio_embeds,
                mask_latents,
                masked_image_latents,
                ref_latents,
                args.guidance_scale,
                extra_step_kwargs,
                do_cfg,
                profile,
                backend,
                compiled_wrapper,
                tensorrt_runner,
                capture,
            )
            latent_blocks.append({"start": block["start"], "latents": latents.detach().cpu(), "frame_count": count})
            del latents, audio_embeds, mask_latents, masked_image_latents, ref_latents
            profile.mark(f"phase_b_after_block_{block['start']}")

    pipe.unet.to("cpu")
    if tensorrt_runner is not None and tensorrt_runner is not preloaded_tensorrt_runner:
        del tensorrt_runner
        tensorrt_runner = None
    gc.collect()
    torch.cuda.empty_cache()
    profile.mark("phase_b_boundary_after_unet_unload")

    pipe.vae.to(device=device, dtype=dtype)
    temp_dir = args.temp_dir
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    temp_video = os.path.join(temp_dir, "video.mp4")
    import imageio

    with imageio.get_writer(
        temp_video,
        fps=25,
        codec="libx264",
        macro_block_size=None,
        ffmpeg_params=["-crf", "13"],
        ffmpeg_log_level="error",
    ) as writer:
        with torch.inference_mode():
            with profile.section("phase_c_decode_restore_stream", cuda=True):
                for block in latent_blocks:
                    latents = block["latents"].to(device=device, dtype=dtype)
                    decoded = pipe.decode_latents(latents)
                    faces_for_paste = faces[block["start"] : block["start"] + block["frame_count"]]
                    ref_pixel_values, _, masks = prepare_aligned_masks_and_masked_images(pipe, faces_for_paste)
                    decoded = pipe.paste_surrounding_pixels_back(decoded, ref_pixel_values, 1 - masks, device, dtype)
                    restore_and_write_streaming(pipe, decoded, video_frames, boxes, affine_matrices, block["start"], temp_video, writer)
                    del latents, decoded, faces_for_paste, ref_pixel_values, masks

    with profile.section("audio_muxing"):
        frame_count = len(whisper_chunks)
        audio_samples_remain_length = int(frame_count / 25 * 16000)
        audio_out = os.path.join(temp_dir, "audio.wav")
        sf.write(audio_out, audio_samples[:audio_samples_remain_length].cpu().numpy(), 16000)
        ensure_parent(args.video_out_path)
        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            temp_video,
            "-i",
            audio_out,
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-q:v",
            "0",
            "-q:a",
            "0",
            args.video_out_path,
        ]
        subprocess.run(command, check=True)

    result = profile.summary()
    result.update(
        {
            "backend": backend,
            "scheduler_type": args.scheduler_type,
            "output": args.video_out_path,
            "ffprobe": run_ffprobe(args.video_out_path),
            "compile_info": compile_info,
        }
    )
    return result


@torch.inference_mode()
def run_original_profiled(
    pipe,
    args,
    profile,
    video_frames,
    faces,
    boxes,
    affine_matrices,
    audio_samples,
    whisper_chunks,
    timesteps,
    extra_step_kwargs,
    height,
    width,
    dtype,
    device,
    generator,
    do_cfg,
):
    synced_video_frames = []
    with profile.section("full_video_initial_noise", cuda=True):
        all_latents = pipe.prepare_latents(len(whisper_chunks), pipe.vae.config.latent_channels, height, width, dtype, device, generator)
    num_frames = 16
    num_inferences = math.ceil(len(whisper_chunks) / num_frames)
    for i in tqdm.tqdm(range(num_inferences), desc="profile original inference"):
        start = i * num_frames
        count = min(num_frames, len(whisper_chunks) - start)
        with profile.section("audio_conditioning"):
            audio_embeds = make_audio_embeds(whisper_chunks, start, count, device, dtype, do_cfg)
        latents = all_latents[:, :, start : start + count]
        block = prepare_block_latents(pipe, faces, start, count, height, width, dtype, device, generator, do_cfg, profile)
        mask_latents = block["mask_latents"].to(device=device, dtype=dtype)
        masked_image_latents = block["masked_image_latents"].to(device=device, dtype=dtype)
        ref_latents = block["ref_latents"].to(device=device, dtype=dtype)
        pipe.scheduler.set_timesteps(args.inference_steps, device=device)
        block_timesteps = pipe.scheduler.timesteps
        with profile.section("unet_denoising_total", cuda=True):
            latents = denoise_block(
                pipe,
                latents,
                block_timesteps,
                audio_embeds,
                mask_latents,
                masked_image_latents,
                ref_latents,
                args.guidance_scale,
                extra_step_kwargs,
                do_cfg,
                profile,
                "original",
            )
        with profile.section("vae_decoding", cuda=True):
            decoded_latents = pipe.decode_latents(latents)
        with profile.section("mouth_region_paste", cuda=True):
            ref_pixel_values, _, masks = pipe.image_processor.prepare_masks_and_masked_images(
                faces[start : start + count], affine_transform=False
            )
            decoded_latents = pipe.paste_surrounding_pixels_back(decoded_latents, ref_pixel_values, 1 - masks, device, dtype)
        synced_video_frames.append(decoded_latents)

    with profile.section("face_restoration", cuda=True):
        synced_video_frames = pipe.restore_video(torch.cat(synced_video_frames), video_frames, boxes, affine_matrices)
    with profile.section("audio_muxing"):
        audio_samples_remain_length = int(synced_video_frames.shape[0] / 25 * 16000)
        audio_samples = audio_samples[:audio_samples_remain_length].cpu().numpy()
        if os.path.exists(args.temp_dir):
            shutil.rmtree(args.temp_dir)
        os.makedirs(args.temp_dir, exist_ok=True)
        write_video(os.path.join(args.temp_dir, "video.mp4"), synced_video_frames, fps=25)
        sf.write(os.path.join(args.temp_dir, "audio.wav"), audio_samples, 16000)
        ensure_parent(args.video_out_path)
        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            os.path.join(args.temp_dir, "video.mp4"),
            "-i",
            os.path.join(args.temp_dir, "audio.wav"),
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-q:v",
            "0",
            "-q:a",
            "0",
            args.video_out_path,
        ]
        subprocess.run(command, check=True)
    result = profile.summary()
    result.update({"backend": "profile-original", "output": args.video_out_path, "ffprobe": run_ffprobe(args.video_out_path)})
    return result


def export_or_tensorrt_probe(pipe, args, profile, block, timestep, device, dtype, backend):
    wrapper = TensorOnlyUNet(pipe.unet).to(device).eval()
    latents = pipe.prepare_latents(block["frame_count"], pipe.vae.config.latent_channels, 512, 512, dtype, device, None)
    mask_latents = block["mask_latents"].to(device=device, dtype=dtype)
    masked_image_latents = block["masked_image_latents"].to(device=device, dtype=dtype)
    ref_latents = block["ref_latents"].to(device=device, dtype=dtype)
    audio_embeds = block["audio_embeds"].to(device=device, dtype=dtype)
    cfg_latents, unet_input = preallocate_static_inputs(latents, mask_latents, masked_image_latents, ref_latents)
    cfg_latents[0].copy_(latents[0])
    cfg_latents[1].copy_(latents[0])
    unet_input[:, 0:4].copy_(pipe.scheduler.scale_model_input(cfg_latents, timestep))
    timestep_tensor = timestep.detach().clone() if torch.is_tensor(timestep) else torch.tensor(timestep, device=device)

    out_dir = Path(os.environ.get("LATENTSYNC_ROOT", ".")) / "engines"
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "latentsync_unet_512x16_cfg_fp16.onnx"
    probe = {"backend": backend, "onnx_path": str(onnx_path)}
    with profile.section("onnx_export", cuda=True):
        try:
            torch.onnx.export(
                wrapper,
                (unet_input, timestep_tensor, audio_embeds),
                str(onnx_path),
                input_names=["latent_input", "timestep", "audio_conditioning"],
                output_names=["noise_pred"],
                opset_version=18,
                dynamo=True,
                external_data=True,
            )
            probe["onnx_export"] = "pass"
        except Exception as exc:
            probe["onnx_export"] = "fail"
            probe["onnx_error"] = repr(exc)
            return {**profile.summary(), **probe}

    try:
        import onnx

        onnx.checker.check_model(str(onnx_path))
        probe["onnx_check"] = "pass"
    except Exception as exc:
        probe["onnx_check"] = "fail"
        probe["onnx_check_error"] = repr(exc)

    if backend == "onnx":
        return {**profile.summary(), **probe}

    with profile.section("tensorrt_engine_probe"):
        try:
            import onnxruntime as ort

            sess = ort.InferenceSession(
                str(onnx_path),
                providers=[
                    (
                        "TensorrtExecutionProvider",
                        {
                            "trt_fp16_enable": True,
                            "trt_engine_cache_enable": True,
                            "trt_engine_cache_path": str(out_dir / "ort_trt_cache"),
                        },
                    ),
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider",
                ],
            )
            probe["ort_providers"] = sess.get_providers()
            probe["tensorrt_session"] = "pass" if "TensorrtExecutionProvider" in sess.get_providers() else "not_selected"
        except Exception as exc:
            probe["tensorrt_session"] = "fail"
            probe["tensorrt_error"] = repr(exc)
    return {**profile.summary(), **probe}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unet_config_path", required=True)
    parser.add_argument("--inference_ckpt_path", required=True)
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--audio_path", required=True)
    parser.add_argument("--video_out_path", required=True)
    parser.add_argument("--temp_dir", default="temp")
    parser.add_argument("--profile_json", required=True)
    parser.add_argument("--inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=1247)
    parser.add_argument(
        "--backend",
        choices=[
            "profile-original",
            "staged",
            "static",
            "compile-default",
            "compile-autotune",
            "compile-aot-eager",
            "onnx",
            "tensorrt",
            "tensorrt-rtx",
        ],
        default="profile-original",
    )
    parser.add_argument(
        "--scheduler_type",
        choices=["ddim", "dpm_solver_multistep", "pndm"],
        default="ddim",
    )
    parser.add_argument(
        "--tensorrt_engine_path",
        default=str(Path(os.environ.get("LATENTSYNC_ROOT", repo_root())) / "engines" / "latentsync_unet_512x16_cfg_fp16.trt_rtx_1_5.plan"),
    )
    args = parser.parse_args()

    if args.seed != -1:
        set_seed(args.seed)
    else:
        torch.seed()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.reset_peak_memory_stats()
    dtype = torch.float16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7 else torch.float32
    config = OmegaConf.load(args.unet_config_path)
    profile = Profile()
    print(f"Optimized inference backend={args.backend} seed={torch.initial_seed()} dtype={dtype}")
    pipe = build_pipeline(config, args, dtype)
    result = run_profiled(pipe, args, config, args.backend, profile)
    ensure_parent(args.profile_json)
    with open(args.profile_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({"backend": args.backend, "profile_json": args.profile_json, "output": args.video_out_path}, indent=2))


if __name__ == "__main__":
    main()
