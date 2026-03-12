# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import os
import time
import subprocess

MODEL_CACHE = "checkpoints"
MODEL_URL = "https://weights.replicate.delivery/default/chunyu-li/LatentSync/model.tar"


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # Download the model weights
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)

        # Soft links for the auxiliary models
        os.system("mkdir -p ~/.cache/torch/hub/checkpoints")
        os.system(
            "ln -s $(pwd)/checkpoints/auxiliary/vgg16-397923af.pth ~/.cache/torch/hub/checkpoints/vgg16-397923af.pth"
        )

    def predict(
        self,
        video: Path = Input(description="Input video", default=None),
        audio: Path = Input(description="Input audio to ", default=None),
        guidance_scale: float = Input(description="Guidance scale", ge=1, le=3, default=1.7),
        inference_steps: int = Input(description="Inference steps", ge=20, le=50, default=20),
        seed: int = Input(description="Set to 0 for Random seed", default=0),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed <= 0:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        video_path = str(video)
        audio_path = str(audio)
        config_path = "configs/unet/stage2.yaml"
        ckpt_path = "checkpoints/latentsync_unet.pt"
        output_path = "/tmp/video_out.mp4"

        # Run the following command:
        os.system(
            f"python -m scripts.inference --unet_config_path {config_path} --inference_ckpt_path {ckpt_path} --guidance_scale {str(guidance_scale)} --video_path {video_path} --audio_path {audio_path} --video_out_path {output_path} --seed {seed} --inference_steps {inference_steps}"
        )
        return Path(output_path)
