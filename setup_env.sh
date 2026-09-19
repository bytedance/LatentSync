#!/bin/bash
set -e

command -v uv >/dev/null 2>&1 || { echo "uv is required: https://docs.astral.sh/uv/"; exit 1; }

# Create a virtual environment with uv (downloads Python if needed)
uv venv --python 3.13.9
source .venv/bin/activate

# ffmpeg must be installed on the system
if ! command -v ffmpeg >/dev/null 2>&1; then
    if command -v brew >/dev/null 2>&1; then
        brew install ffmpeg
    elif command -v apt >/dev/null 2>&1; then
        sudo apt -y install ffmpeg
    else
        echo "Please install ffmpeg manually, then re-run this script."
        exit 1
    fi
fi

# Python dependencies
uv pip install -r requirements.txt

# OpenCV dependencies (only needed with apt-based Linux)
if [ "$(uname)" = "Linux" ] && command -v apt >/dev/null 2>&1; then
    sudo apt -y install libgl1
fi

# Download the checkpoints required for inference from HuggingFace
hf download ByteDance/LatentSync-1.6 whisper/tiny.pt --local-dir checkpoints
hf download ByteDance/LatentSync-1.6 latentsync_unet.pt --local-dir checkpoints
hf download ByteDance/LatentSync-1.5 latentsync_unet.pt --local-dir checkpoints/v1.5

echo "Done. Activate the environment with: source .venv/bin/activate"
