#!/bin/bash
docker build -t latentsync .
echo "✅ Build completata!"
echo "👉 Per eseguire il container usa:"
echo "docker run --rm -it --gpus all -p 8082:8080  latentsync"
