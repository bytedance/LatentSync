#!/bin/bash
docker build -t lsaggese/ai_latentsync .
docker push lsaggese/ai_latentsync
echo "✅ Build completata!"
echo "👉 Per eseguire il container usa:"
echo "docker run --rm -it --gpus all -p 8082:8080  lsaggese/ai_latentsync"
