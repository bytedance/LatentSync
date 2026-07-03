import argparse
import ctypes
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from accelerate.utils import set_seed
from omegaconf import OmegaConf

from scripts.inference_optimized import (
    Profile,
    TensorRTRTXUNetRunner,
    build_pipeline,
    ensure_parent,
    repo_root,
    run_profiled,
)


def _load_jobs(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("jobs")
    if not isinstance(data, list) or not data:
        raise ValueError("Batch manifest must be a JSON array or an object with a non-empty 'jobs' array.")
    for index, job in enumerate(data):
        if not isinstance(job, dict):
            raise ValueError(f"Job {index} must be a JSON object.")
        missing = [key for key in ("video_path", "audio_path", "video_out_path") if not job.get(key)]
        if missing:
            raise ValueError(f"Job {index} is missing: {', '.join(missing)}")
    return data


def _job_args(base_args, job, index, stamp):
    values = vars(base_args).copy()
    values.update(job)
    root = Path(values["root"])
    values["temp_dir"] = values.get("temp_dir") or str(root / "temp" / f"batch_{stamp}_job_{index:03d}")
    values["profile_json"] = values.get("profile_json") or str(root / "logs" / f"batch_{stamp}_job_{index:03d}_profile.json")
    for key in ("video_path", "audio_path", "video_out_path", "temp_dir"):
        values[key] = _short_path(values[key], create_parent=key in {"video_out_path", "temp_dir"})
    return SimpleNamespace(**values)


def _short_path(path, create_parent=False):
    path = str(path)
    if os.name != "nt":
        return path
    target = Path(path)
    if create_parent:
        parent = target if target.suffix == "" else target.parent
        parent.mkdir(parents=True, exist_ok=True)
    buffer = ctypes.create_unicode_buffer(4096)
    result = ctypes.windll.kernel32.GetShortPathNameW(path, buffer, len(buffer))
    if result:
        return buffer.value
    return path


def main():
    parser = argparse.ArgumentParser(description="Run multiple LatentSync optimized jobs in one warm process.")
    parser.add_argument("--jobs_json", required=True, help="JSON array, or object with jobs array.")
    parser.add_argument("--root", default=os.environ.get("LATENTSYNC_ROOT", str(repo_root())))
    parser.add_argument("--unet_config_path", default="configs/unet/stage2_512.yaml")
    parser.add_argument("--inference_ckpt_path", default="checkpoints/latentsync_unet.pt")
    parser.add_argument("--inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=1247)
    parser.add_argument("--backend", choices=["tensorrt-rtx", "staged"], default="tensorrt-rtx")
    parser.add_argument(
        "--tensorrt_engine_path",
        default=str(Path(os.environ.get("LATENTSYNC_ROOT", str(repo_root()))) / "engines" / "latentsync_unet_512x16_cfg_fp16.trt_rtx_1_5.plan"),
    )
    parser.add_argument("--summary_json", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    os.environ.setdefault("LATENTSYNC_ROOT", str(root))
    stamp = time.strftime("%Y%m%d_%H%M%S")
    if args.summary_json is None:
        args.summary_json = str(root / "logs" / f"batch_{stamp}_summary.json")

    jobs = _load_jobs(args.jobs_json)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    dtype = torch.float16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7 else torch.float32
    config = OmegaConf.load(args.unet_config_path)
    print(f"Batch optimized inference jobs={len(jobs)} backend={args.backend} dtype={dtype}")
    pipe = build_pipeline(config, args, dtype)

    preloaded_runner = None
    if args.backend == "tensorrt-rtx":
        started = time.perf_counter()
        preloaded_runner = TensorRTRTXUNetRunner(args.tensorrt_engine_path)
        engine_load_seconds = time.perf_counter() - started
    else:
        engine_load_seconds = None

    summary = {
        "status": "PASS",
        "backend": args.backend,
        "jobs_json": args.jobs_json,
        "summary_json": args.summary_json,
        "engine_load_seconds": engine_load_seconds,
        "jobs": [],
    }
    try:
        for index, job in enumerate(jobs):
            job_args = _job_args(args, job, index, stamp)
            if job_args.seed != -1:
                set_seed(job_args.seed)
            else:
                torch.seed()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            profile = Profile()
            print(f"Batch job {index + 1}/{len(jobs)} video={job_args.video_path} audio={job_args.audio_path}")
            result = run_profiled(
                pipe,
                job_args,
                config,
                job_args.backend,
                profile,
                preloaded_tensorrt_runner=preloaded_runner,
            )
            ensure_parent(job_args.profile_json)
            Path(job_args.profile_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
            summary["jobs"].append(
                {
                    "status": "PASS",
                    "index": index,
                    "video_path": job_args.video_path,
                    "audio_path": job_args.audio_path,
                    "video_out_path": job_args.video_out_path,
                    "profile_json": job_args.profile_json,
                    "wall_seconds": result.get("total_wall_seconds"),
                    "ffprobe": result.get("ffprobe"),
                }
            )
    except Exception as exc:
        summary["status"] = "FAIL"
        summary["error"] = repr(exc)
        raise
    finally:
        ensure_parent(args.summary_json)
        Path(args.summary_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({"status": summary["status"], "summary_json": args.summary_json}, indent=2))


if __name__ == "__main__":
    main()
