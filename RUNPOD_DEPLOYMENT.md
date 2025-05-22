# Deploying the Model to RunPod Serverless

This guide provides instructions on how to deploy this model to RunPod Serverless using the provided `Dockerfile` and `runpod_handler.py`.

## Prerequisites

1.  **Docker Installed:** You need Docker installed and running on your local machine to build the image.
2.  **RunPod Account:** You'll need a RunPod account.
3.  **Container Registry:** A place to push your Docker image (e.g., Docker Hub, AWS ECR, Google GCR, or RunPod's registry if available).

## Build and Push the Docker Image

1.  **Navigate to the Project Root:**
    Open your terminal and change to the directory containing the `Dockerfile` and the rest of the project files.

2.  **Build the Docker Image:**
    Run the following command to build the image. Replace `<your-registry>/<your-image-name>:<tag>` with your desired image name and tag (e.g., `docker.io/yourusername/latentsync-runpod:latest`).

    ```bash
    docker build -t <your-registry>/<your-image-name>:<tag> .
    ```

3.  **Log in to Your Container Registry (if needed):**
    If you're using Docker Hub:
    ```bash
    docker login
    ```
    For other registries, follow their specific login instructions.

4.  **Push the Docker Image:**
    Push the built image to your container registry:
    ```bash
    docker push <your-registry>/<your-image-name>:<tag>
    ```

## Deploy on RunPod Serverless

1.  **Go to Serverless Endpoints:**
    In your RunPod console, navigate to the "Serverless" or "Endpoints" section.

2.  **Create a New Endpoint:**
    *   Choose to create a new serverless endpoint.
    *   Point to the Docker image you just pushed to your registry (e.g., `<your-registry>/<your-image-name>:<tag>`).
    *   Configure the necessary GPU resources. This model requires a GPU. Refer to the original model's requirements for GPU memory, but a T4 or A10G should generally be a good starting point.
    *   Set environment variables if necessary (though most are handled in the Dockerfile).
    *   Define the container disk size if required (the model weights are downloaded into the container).
    *   RunPod should pick up the `CMD` from the Dockerfile, which starts the Python worker with `runpod_handler.handler`.

3.  **Worker Initialization and Cold Starts:**
    *   The `Predictor.setup()` method, which downloads model weights and sets up auxiliary models, is called on the first request to a new worker or when a worker is initialized. This means the very first request to a new/cold worker might take longer.
    *   Subsequent requests to a warm worker should be faster.
    *   The `MODEL_CACHE` is set to "checkpoints" within the container. Ensure your RunPod worker configuration provides enough disk space for these checkpoints.

## Making Requests to the Endpoint

Once your endpoint is active, you can send requests to it using its unique URL.

*   **Method:** `POST`
*   **Headers:**
    *   `Content-Type: application/json`
    *   `Authorization: Bearer YOUR_RUNPOD_API_KEY` (if your endpoint is private)

*   **Body (JSON):**
    The input should be a JSON object with an `input` key.

    ```json
    {
      "input": {
        "video_url": "https://example.com/path/to/your_video.mp4",
        "audio_url": "https://example.com/path/to/your_audio.wav",
        "guidance_scale": 2.0,  // Optional, default: 2.0, range: 1-3
        "inference_steps": 20, // Optional, default: 20, range: 20-50
        "seed": 0                 // Optional, default: 0 (for random seed)
      }
    }
    ```

    *   `video_url`: Publicly accessible URL to the input video file.
    *   `audio_url`: Publicly accessible URL to the input audio file.
    *   `guidance_scale`, `inference_steps`, `seed` are optional parameters with defaults as specified in `predict.py`.

*   **Output:**
    The endpoint will return a JSON response. If successful, it will contain the path to the generated output video within the RunPod environment. RunPod typically provides a way to access this output, often by temporarily storing it and providing a URL, or by integrating with S3.

    Example successful response (structure might vary slightly based on RunPod's wrapper):
    ```json
    {
      "output_path": "/tmp/video_out.mp4" // Or similar path where the output is stored
    }
    ```
    Or, if RunPod automatically uploads and returns a URL:
    ```json
    {
      "output": { // Key might vary
         "video_url": "https://runpod-generated-url/to/output.mp4"
      }
    }
    ```
    If an error occurs:
    ```json
    {
      "error": "Description of the error."
    }
    ```

This documentation should provide a good starting point for the user.
