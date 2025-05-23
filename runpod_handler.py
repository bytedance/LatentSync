import subprocess
import os
import tempfile
from predict import Predictor # Assuming Predictor is in predict.py

# Global predictor instance to reuse (RunPod might keep workers warm)
# Setup will be called if the instance is not ready
predictor_instance = None

def download_file(url, destination_dir):
    """Downloads a file from a URL to a specified directory using pget."""
    print(f"Downloading {url} to {destination_dir}")
    # Ensure pget is available or handle potential errors
    try:
        # Using pget for potentially faster downloads, similar to cog.yaml
        # pget automatically extracts if it's a tar archive, we might need to adjust if it's just a raw file
        # For simple file download, ensure pget doesn't try to extract.
        # The -x flag in cog.yaml is for extraction. We might not need it if downloading raw video/audio.
        # However, pget might be smart enough. If not, might need to use curl or requests.
        # Let's assume pget handles direct file downloads correctly or we can adjust command.
        # Create a unique filename within the destination directory
        filename = os.path.join(destination_dir, os.path.basename(url))
        subprocess.check_call(["pget", url, filename], close_fds=False) # Simpler pget command for single file
        print(f"Successfully downloaded {url} to {filename}")
        return filename
    except subprocess.CalledProcessError as e:
        print(f"Error downloading {url}: {e}")
        raise
    except FileNotFoundError:
        print("Error: pget command not found. Ensure it is installed and in PATH.")
        # Fallback or error
        raise Exception("pget not found. Cannot download files.")


def handler(event):
    global predictor_instance

    job_input = event.get('input', {})
    if not job_input:
        return {"error": "No input provided in event"}

    video_url = job_input.get('video_url')
    audio_url = job_input.get('audio_url')

    if not video_url or not audio_url:
        return {"error": "video_url and audio_url are required."}

    # Parameters for the model, with defaults matching predict.py if not provided
    guidance_scale = float(job_input.get('guidance_scale', 2.0))
    inference_steps = int(job_input.get('inference_steps', 20))
    seed = int(job_input.get('seed', 0)) # 0 for random seed as per predict.py

    # Initialize predictor if not already done
    if predictor_instance is None:
        print("Initializing Predictor...")
        predictor_instance = Predictor()
        predictor_instance.setup() # This downloads model weights, sets up links
        print("Predictor initialized.")

    try:
        # Create temporary directories for downloaded files
        with tempfile.TemporaryDirectory() as tmpdir_video, tempfile.TemporaryDirectory() as tmpdir_audio:
            print(f"Created temporary directories: {tmpdir_video}, {tmpdir_audio}")
            
            local_video_path = download_file(video_url, tmpdir_video)
            local_audio_path = download_file(audio_url, tmpdir_audio)

            print(f"Calling predictor.predict with video: {local_video_path}, audio: {local_audio_path}")
            # Call the predict method
            output_path_object = predictor_instance.predict(
                video=local_video_path, # predict.py expects string paths
                audio=local_audio_path, # predict.py expects string paths
                guidance_scale=guidance_scale,
                inference_steps=inference_steps,
                seed=seed
            )
            
            output_path_str = str(output_path_object) # Convert Path object to string
            print(f"Prediction successful. Output at: {output_path_str}")

            # RunPod typically handles uploading the file if a path is returned.
            # Alternatively, one might upload to S3 and return the S3 URL.
            # For now, just returning the path as per Cog's behavior.
            return {"output_path": output_path_str}

    except Exception as e:
        print(f"Error during prediction: {e}")
        return {"error": str(e)}

if __name__ == '__main__':
    # This part is for local testing of the handler, if needed.
    # It simulates a RunPod event.
    # You would need to have pget installed and model weights accessible.
    # And ensure predict.py and its dependencies are in PYTHONPATH.
    
    print("Attempting local test run of the handler...")
    
    # Create dummy files for Predictor.setup() if it expects them
    # This setup part is complex due to symlinks in original predict.py
    # For a simple test, we might need to mock parts of Predictor.setup() or ensure files exist
    if not os.path.exists("checkpoints/auxiliary"):
        os.makedirs("checkpoints/auxiliary", exist_ok=True)
        # Create dummy files that setup() tries to link
        open("checkpoints/auxiliary/2DFAN4-cd938726ad.zip", 'a').close()
        open("checkpoints/auxiliary/s3fd-619a316812.pth", 'a').close()
        open("checkpoints/auxiliary/vgg16-397923af.pth", 'a').close()

    # Mock event
    mock_event = {
        "input": {
            "video_url": "https://raw.githubusercontent.com/runpod-workers/worker-template/main/README.md", # Replace with actual small video/text file URL for testing download
            "audio_url": "https://raw.githubusercontent.com/runpod-workers/worker-template/main/README.md", # Replace with actual small audio/text file URL for testing download
            "guidance_scale": 2.0,
            "inference_steps": 20,
            "seed": 42
        }
    }
    
    # Note: The actual model prediction won't work with dummy md files.
    # This local test primarily checks the handler logic, input parsing, and download.
    # To fully test, you'd need valid small video/audio URLs and the model itself.
    # The Predictor().setup() will try to download actual model weights if MODEL_CACHE doesn't exist.
    
    # Before running local test, ensure pget is installed:
    # Check if pget is installed, if not, download it (Linux x86_64 example)
    if subprocess.run(["which", "pget"], capture_output=True).returncode != 0:
        print("pget not found. Attempting to download pget...")
        pget_url = "https://github.com/replicate/pget/releases/download/v0.10.2/pget_linux_x86_64"
        pget_dest = "./pget" # Download to current dir for local test
        subprocess.check_call(["curl", "-L", pget_url, "-o", pget_dest])
        subprocess.check_call(["chmod", "+x", pget_dest])
        # Update PATH for this script's execution or use ./pget
        os.environ["PATH"] = os.getcwd() + os.pathsep + os.environ["PATH"]
        print("pget downloaded and added to PATH for this session.")

    result = handler(mock_event)
    print(f"Local test run result: {result}")
