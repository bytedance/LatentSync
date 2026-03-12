
#!/usr/bin/env python3

import argparse
import os
import subprocess
import tempfile
import urllib.request
import urllib.error
from pathlib import Path
from omegaconf import OmegaConf
import torch
import torchaudio
import cv2
import ffmpeg
from diffusers import AutoencoderKL, DDIMScheduler
from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from accelerate.utils import set_seed
from latentsync.whisper.audio2feature import Audio2Feature
from DeepCache import DeepCacheSDHelper
from datetime import datetime
from enum import Enum
from typing import Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SilenceLocation(Enum):
    START = 'start'
    END = 'end'


def download_file(url, temp_dir):
    """Download file from URL to temporary directory with proper error handling"""
    try:
        print(f"Starting download from: {url}")

        # Create a request with headers to avoid blocking
        request = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

        # Get filename from URL
        filename = os.path.basename(url.split('?')[0])  # Remove query parameters
        if not filename or '.' not in filename:
            # Try to get filename from response headers
            try:
                response = urllib.request.urlopen(request)
                content_disposition = response.headers.get('content-disposition')
                if content_disposition:
                    import re
                    filename_match = re.findall('filename=(.+)', content_disposition)
                    if filename_match:
                        filename = filename_match[0].strip('"')
                response.close()
            except:
                pass

            # Fall back to generic name with extension guessing
            if not filename or '.' not in filename:
                if 'video' in url.lower() or url.endswith('.mp4'):
                    filename = "downloaded_video.mp4"
                elif 'audio' in url.lower() or url.endswith(('.mp3', '.wav', '.mpga')):
                    filename = "downloaded_audio" + (os.path.splitext(url)[1] or '.mp3')
                else:
                    filename = "downloaded_file"

        filepath = os.path.join(temp_dir, filename)
        filepath = os.path.abspath(filepath)

        print(f"Downloading to: {filepath}")
        print(f"File will be saved as: {filename}")

        # Download with progress
        def progress_hook(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, (block_num * block_size * 100) / total_size)
                print(f"\rDownload progress: {percent:.1f}%", end='', flush=True)

        urllib.request.urlretrieve(url, filepath, reporthook=progress_hook)
        print()  # New line after progress

        # Verify download
        if not os.path.exists(filepath):
            raise RuntimeError(f"Download failed - file not created: {filepath}")

        file_size = os.path.getsize(filepath)
        if file_size == 0:
            raise RuntimeError(f"Download failed - empty file: {filepath}")

        print(f"Download successful: {filepath} ({file_size} bytes)")
        return filepath

    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP Error {e.code}: {e.reason} for URL: {url}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"URL Error: {e.reason} for URL: {url}")
    except Exception as e:
        raise RuntimeError(f"Download failed for {url}: {str(e)}")


def convert_audio_to_wav(input_path, output_path):
    """Convert audio file to WAV format using ffmpeg"""
    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)

    print(f"Converting audio {input_path} to WAV format: {output_path}")

    try:
        result = subprocess.run([
            'ffmpeg', '-i', input_path,
            '-ar', '16000',  # Sample rate for whisper
            '-ac', '1',      # Mono channel
            '-f', 'wav',
            '-y',            # Overwrite output file
            output_path
        ], check=True, capture_output=True, text=True)
        print(f"Audio conversion completed: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg stderr: {e.stderr}")
        print(f"FFmpeg stdout: {e.stdout}")
        raise RuntimeError(f"FFmpeg conversion failed: {e.stderr}")
    except FileNotFoundError:
        raise RuntimeError("FFmpeg not found. Please install FFmpeg and ensure it's in your PATH")


def get_video_duration(video_path):
    """Get video duration using ffmpeg"""
    if not os.path.exists(video_path):
        return 0

    try:
        probe = ffmpeg.probe(video_path, v='error', select_streams='v:0',
                             show_entries='format=duration')
        return float(probe['format']['duration'])
    except Exception as e:
        logger.warning(f"Could not get duration for {video_path}: {e}")
        return 0


def has_audio(video_path):
    """Check if video has audio stream"""
    if not os.path.exists(video_path):
        return False

    try:
        probe = ffmpeg.probe(video_path, v='error', select_streams='a')
        return len(probe.get('streams', [])) > 0
    except Exception as e:
        logger.warning(f"Could not detect audio in {video_path}: {e}")
        return False


def reverse_video(video_path, temp_directory_path):
    """Reverse video with or without audio"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    reversed_path = f"{temp_directory_path}/reversed_{timestamp}.mp4"
    audio_exists = has_audio(video_path)

    try:
        if audio_exists:
            ffmpeg.input(video_path).output(
                reversed_path, vf='reverse', af='areverse',
                **{'c:v': 'libx264', 'preset': 'fast', 'crf': '23'}
            ).run(overwrite_output=True, quiet=True)
        else:
            ffmpeg.input(video_path).output(
                reversed_path, vf='reverse',
                **{'c:v': 'libx264', 'preset': 'fast', 'crf': '23'}
            ).run(overwrite_output=True, quiet=True)
    except ffmpeg.Error as e:
        logger.error(f"Video reversal failed: {e}")
        return video_path

    return reversed_path


def extend_video(video_path, target_duration, temp_directory_path, loop_from_end=True):
    """Extend video duration by looping"""
    original_duration = get_video_duration(video_path)
    if original_duration >= target_duration:
        return video_path

    audio_exists = has_audio(video_path)
    clips = [video_path]
    total_duration = original_duration

    try:
        while total_duration < target_duration:
            if loop_from_end:
                reversed_clip = reverse_video(clips[-1], temp_directory_path)
                if reversed_clip != clips[-1]:  # Only add if reversal was successful
                    clips.append(reversed_clip)
                else:
                    break
            else:
                clips.append(clips[0])  # Use original instead of last
            total_duration += original_duration

        if len(clips) <= 1:
            return video_path

        # Create extended video
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        extended_video_path = f"{temp_directory_path}/extended_{timestamp}.mp4"

        # Use ffmpeg-python for concatenation
        inputs = [ffmpeg.input(clip) for clip in clips]
        ffmpeg.concat(*inputs, v=1, a=1 if audio_exists else 0).output(
            extended_video_path,
            **{'c:v': 'libx264', 'preset': 'fast', 'crf': '23'}
        ).run(overwrite_output=True, quiet=True)

        return extended_video_path

    except Exception as e:
        logger.error(f"Video extension failed: {e}")
        return video_path


def trim_video(video_path, target_duration, temp_directory_path):
    """Trim video to target duration"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trimmed_path = f"{temp_directory_path}/trimmed_{timestamp}.mp4"

    try:
        ffmpeg.input(video_path, ss=0, to=target_duration).output(
            trimmed_path,
            **{'c:v': 'libx264', 'preset': 'fast', 'crf': '23', 'c:a': 'aac'}
        ).run(overwrite_output=True, quiet=True)
        return trimmed_path
    except ffmpeg.Error as e:
        logger.error(f"Video trimming failed: {e}")
        return video_path


def add_silence_to_audio(audio_path, temp_directory_path, silence_location=SilenceLocation.END, silence_duration=1.0):
    """Add silence to audio at start or end"""
    waveform, sample_rate = torchaudio.load(audio_path)

    # Ensure float32 for better quality
    if waveform.dtype != torch.float32:
        waveform = waveform.float()

    silence_samples = int(silence_duration * sample_rate)
    silence = torch.zeros((waveform.shape[0], silence_samples), dtype=waveform.dtype)

    if silence_location == SilenceLocation.START:
        modified_waveform = torch.cat((silence, waveform), dim=1)
        suffix = "silence_start"
    else:
        modified_waveform = torch.cat((waveform, silence), dim=1)
        suffix = "silence_end"

    input_name = os.path.splitext(os.path.basename(audio_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{temp_directory_path}/{input_name}_{suffix}_{timestamp}.wav"

    torchaudio.save(output_path, modified_waveform, sample_rate)
    return output_path


def pad_audio_to_multiple_of_16(audio_path, temp_directory_path, target_fps=25):
    """Pad audio duration to multiple of 16 frames"""
    waveform, sample_rate = torchaudio.load(audio_path)

    # Use float32 for audio processing to maintain quality
    if waveform.dtype != torch.float32:
        waveform = waveform.float()

    audio_duration = waveform.shape[1] / sample_rate
    num_frames = int(audio_duration * target_fps)
    remainder = num_frames % 16

    if remainder > 0:
        pad_frames = 16 - remainder
        pad_samples = int((pad_frames / target_fps) * sample_rate)

        # More memory efficient padding
        padded_waveform = torch.nn.functional.pad(waveform, (0, pad_samples))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        padded_audio_path = f"{temp_directory_path}/padded_audio_{timestamp}.wav"
        torchaudio.save(padded_audio_path, padded_waveform, sample_rate)

        final_num_frames = int((padded_waveform.shape[1] / sample_rate) * target_fps)
        return padded_audio_path, final_num_frames
    else:
        final_num_frames = num_frames
        return audio_path, final_num_frames


def convert_video_fps(input_path, target_fps, temp_directory_path, prefix="converted"):
    """Convert video FPS with optimized ffmpeg settings"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{temp_directory_path}/{prefix}_{target_fps}fps_{timestamp}.mp4"

    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-filter:v", f"fps={target_fps}",
        "-c:v", "libx264", "-preset", "medium", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        output_path
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg conversion failed: {e.stderr}")
        raise


def replace_audio_in_video(video_path, original_audio_path, output_path):
    """Replace audio in video with original high-quality audio using ffmpeg"""
    video_path = os.path.abspath(video_path)
    original_audio_path = os.path.abspath(original_audio_path)
    output_path = os.path.abspath(output_path)

    print(f"Replacing audio in {video_path} with original audio from {original_audio_path}")
    print(f"Final output: {output_path}")

    try:
        result = subprocess.run([
            'ffmpeg',
            '-i', video_path,           # Input video (with lip-sync)
            '-i', original_audio_path,  # Original high-quality audio
            '-c:v', 'copy',             # Copy video stream without re-encoding
            '-c:a', 'aac',              # Encode audio as AAC
            '-b:a', '192k',             # High quality audio bitrate
            '-map', '0:v:0',            # Use video from first input
            '-map', '1:a:0',            # Use audio from second input
            '-shortest',                # Match duration to shortest stream
            '-y',                       # Overwrite output file
            output_path
        ], check=True, capture_output=True, text=True)
        print(f"Audio replacement completed: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg stderr: {e.stderr}")
        print(f"FFmpeg stdout: {e.stdout}")
        raise RuntimeError(f"FFmpeg audio replacement failed: {e.stderr}")


def process_video_audio_sync(video_path, audio_path, temp_directory_path,
                             add_silence=True, silence_location=SilenceLocation.END,
                             loop_from_end=True):
    """Process video and audio to match durations"""

    # Convert video to 25 FPS first
    video_25fps = convert_video_fps(video_path, 25, temp_directory_path, "base")

    # Get durations
    video_duration = get_video_duration(video_25fps)
    waveform, sample_rate = torchaudio.load(audio_path)
    audio_duration = waveform.shape[1] / sample_rate

    logger.info(f"Durations - Video: {video_duration:.2f}s, Audio: {audio_duration:.2f}s")

    processed_video = video_25fps
    processed_audio = audio_path

    # Handle duration mismatch
    if abs(video_duration - audio_duration) > 0.1:
        if audio_duration > video_duration:
            logger.info(f"Extending video by {audio_duration - video_duration:.2f}s")
            processed_video = extend_video(video_25fps, audio_duration, temp_directory_path, loop_from_end)
        else:
            if add_silence:
                silence_duration = video_duration - audio_duration
                logger.info(f"Adding {silence_duration:.2f}s silence to audio ({silence_location.value})")
                processed_audio = add_silence_to_audio(audio_path, temp_directory_path, silence_location, silence_duration)
            else:
                logger.info(f"Trimming video by {video_duration - audio_duration:.2f}s")
                processed_video = trim_video(video_25fps, audio_duration, temp_directory_path)

    return processed_video, processed_audio


def main_inference(video_path, audio_wav_path, temp_output_path, download_temp_dir):
    """Main inference method with hardcoded config paths"""

    # Convert all paths to absolute paths
    video_path = os.path.abspath(video_path)
    audio_wav_path = os.path.abspath(audio_wav_path)
    temp_output_path = os.path.abspath(temp_output_path)

    # Hardcoded configuration paths
    unet_config_path = "configs/unet/stage2_512.yaml"
    inference_ckpt_path = "checkpoints/latentsync_unet.pt"

    # Load configuration
    config = OmegaConf.load(unet_config_path)

    # Check input files exist with detailed info
    print(f"Checking video file: {video_path}")
    print(f"Video file exists: {os.path.exists(video_path)}")
    if os.path.exists(video_path):
        print(f"Video file size: {os.path.getsize(video_path)} bytes")
    else:
        raise RuntimeError(f"Video path '{video_path}' not found")

    print(f"Checking audio file: {audio_wav_path}")
    print(f"Audio file exists: {os.path.exists(audio_wav_path)}")
    if os.path.exists(audio_wav_path):
        print(f"Audio file size: {os.path.getsize(audio_wav_path)} bytes")
    else:
        raise RuntimeError(f"Audio path '{audio_wav_path}' not found")

    # Check if the GPU supports float16
    is_fp16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
    dtype = torch.float16 if is_fp16_supported else torch.float32

    print(f"Input video path (absolute): {video_path}")
    print(f"Input audio path (absolute): {audio_wav_path}")
    print(f"Loaded checkpoint path: {inference_ckpt_path}")
    print(f"Temporary output path (absolute): {temp_output_path}")

    scheduler = DDIMScheduler.from_pretrained("configs")

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
        inference_ckpt_path,
        device="cpu",
    )

    unet = unet.to(dtype=dtype)

    pipeline = LipsyncPipeline(
        vae=vae,
        audio_encoder=audio_encoder,
        unet=unet,
        scheduler=scheduler,
    ).to("cuda")

    # Default parameters
    inference_steps = 20
    guidance_scale = 1.5
    seed = 1247

    # Create separate temp dir for pipeline processing (not where downloads are stored)
    pipeline_temp_dir = os.path.join(download_temp_dir, "pipeline_temp")
    os.makedirs(pipeline_temp_dir, exist_ok=True)

    if seed != -1:
        set_seed(seed)
    else:
        torch.seed()

    print(f"Initial seed: {torch.initial_seed()}")
    print(f"Pipeline temp dir: {pipeline_temp_dir}")

    # Double-check files exist right before pipeline call
    print(f"Final check - Video exists: {os.path.exists(video_path)}")
    print(f"Final check - Audio exists: {os.path.exists(audio_wav_path)}")

    # Run the pipeline with separate temp directory
    pipeline(
        video_path=video_path,
        audio_path=audio_wav_path,
        video_out_path=temp_output_path,
        num_frames=config.data.num_frames,
        num_inference_steps=inference_steps,
        guidance_scale=guidance_scale,
        weight_dtype=dtype,
        width=config.data.resolution,
        height=config.data.resolution,
        mask_image_path=config.data.mask_image_path,
        temp_dir=pipeline_temp_dir,  # Use separate temp dir for pipeline
    )

    print(f"Lip-sync inference completed. Temporary output saved to: {temp_output_path}")


def main():
    parser = argparse.ArgumentParser(description="LatentSync inference with URL inputs and duration matching")
    parser.add_argument("--video_url", type=str, required=True, help="URL of the input video")
    parser.add_argument("--audio_url", type=str, required=True, help="URL of the input audio")
    parser.add_argument("--output_path", type=str, required=True, help="Path for output video")
    parser.add_argument("--temp_dir", type=str, default="temp1", help="Temporary directory for downloads")
    parser.add_argument("--add_silence", action="store_true", default=True, help="Add silence to audio if shorter than video")
    parser.add_argument("--silence_location", type=str, choices=['start', 'end'], default='end',
                        help="Where to add silence (start or end)")
    parser.add_argument("--loop_from_end", action="store_true", default=True,
                        help="Loop video by reversing from end (creates smoother loop)")
    parser.add_argument("--output_fps", type=int, default=25, help="Output video FPS")

    args = parser.parse_args()

    # Create temporary directory with absolute path
    temp_dir = os.path.abspath(args.temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    print(f"Using temporary directory: {temp_dir}")

    # Test URLs accessibility
    print(f"Video URL: {args.video_url}")
    print(f"Audio URL: {args.audio_url}")

    video_path = None
    original_audio_path = None
    audio_wav_path = None
    temp_output_path = None

    try:
        # Download video and audio files
        print("=" * 50)
        print("DOWNLOADING FILES")
        print("=" * 50)

        print("Downloading video...")
        video_path = download_file(args.video_url, temp_dir)

        print("Downloading audio...")
        original_audio_path = download_file(args.audio_url, temp_dir)

        # Verify files were downloaded successfully
        if not os.path.exists(video_path):
            raise RuntimeError(f"Failed to download video from {args.video_url}")
        if not os.path.exists(original_audio_path):
            raise RuntimeError(f"Failed to download audio from {args.audio_url}")

        print(f"\nDownload Summary:")
        print(f"Video: {video_path} ({os.path.getsize(video_path)} bytes)")
        print(f"Audio: {original_audio_path} ({os.path.getsize(original_audio_path)} bytes)")

        # Convert audio to WAV for inference if needed
        audio_wav_path = os.path.join(temp_dir, "audio_for_inference.wav")
        audio_wav_path = os.path.abspath(audio_wav_path)

        # Check if audio is already in WAV format
        if original_audio_path.lower().endswith('.wav'):
            print("Audio is already in WAV format")
            audio_wav_path = original_audio_path
        else:
            convert_audio_to_wav(original_audio_path, audio_wav_path)

        # Process video and audio to match durations
        print("\n" + "=" * 50)
        print("PROCESSING VIDEO AND AUDIO SYNC")
        print("=" * 50)

        silence_location = SilenceLocation.START if args.silence_location == 'start' else SilenceLocation.END
        processed_video, processed_audio = process_video_audio_sync(
            video_path, audio_wav_path, temp_dir,
            add_silence=args.add_silence,
            silence_location=silence_location,
            loop_from_end=args.loop_from_end
        )

        # Pad audio to multiple of 16 frames
        print("Padding audio to multiple of 16 frames...")
        padded_audio, final_frames = pad_audio_to_multiple_of_16(processed_audio, temp_dir)
        print(f"Final audio frames: {final_frames}")

        # Create temporary output path for lip-sync video (without final audio)
        temp_output_path = os.path.join(temp_dir, "lipsync_output_temp.mp4")
        temp_output_path = os.path.abspath(temp_output_path)

        # Run main inference - pass the download temp dir so pipeline can create its own subdir
        print("\n" + "=" * 50)
        print("RUNNING LIP-SYNC INFERENCE")
        print("=" * 50)
        main_inference(processed_video, padded_audio, temp_output_path, temp_dir)

        # Verify inference output exists
        if not os.path.exists(temp_output_path):
            raise RuntimeError(f"Inference failed - output file not created: {temp_output_path}")

        print(f"Inference output size: {os.path.getsize(temp_output_path)} bytes")

        # Convert FPS if needed
        final_video = temp_output_path
        if args.output_fps != 25:
            print(f"Converting to {args.output_fps} FPS...")
            final_video = convert_video_fps(temp_output_path, args.output_fps, temp_dir, "final")

        # Replace audio with original high-quality audio
        print("\n" + "=" * 50)
        print("REPLACING WITH ORIGINAL AUDIO")
        print("=" * 50)
        output_path = os.path.abspath(args.output_path)
        replace_audio_in_video(final_video, original_audio_path, output_path)

        print(f"\nProcess completed successfully! Final output: {output_path}")

        # Only cleanup on success
        print("Cleaning up temporary files...")

        # Clean up the entire temp directory
        # import shutil
        # if os.path.exists(temp_dir):
        #     try:
        #         shutil.rmtree(temp_dir)
        #         print(f"Cleaned up temp directory: {temp_dir}")
        #     except Exception as e:
        #         print(f"Warning: Could not clean up temp directory {temp_dir}: {e}")

        return 0

    except Exception as e:
        print(f"\nError: {e}")
        print(f"Temporary files preserved for debugging in: {temp_dir}")

        # # Print debug info about existing files
        # if os.path.exists(temp_dir):
        #     print("\nFiles in temp directory:")
        #     try:
        #         for root, dirs, files in os.walk(temp_dir):
        #             level = root.replace(temp_dir, '').count(os.sep)
        #             indent = ' ' * 2 * level
        #             print(f"{indent}{os.path.basename(root)}/")
        #             subindent = ' ' * 2 * (level + 1)
        #             for file in files:
        #                 file_path = os.path.join(root, file)
        #                 if os.path.isfile(file_path):
        #                     print(f"{subindent}{file} ({os.path.getsize(file_path)} bytes)")
        #     except Exception as walk_error:
        #         print(f"Error walking directory: {walk_error}")

        return 1


if __name__ == "__main__":
    exit(main())
