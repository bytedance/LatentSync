# Usa un'immagine di base con Python 3.10
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Imposta il maintainer
LABEL maintainer="Andrea"

# Imposta la directory di lavoro
WORKDIR /app

# Installa i pacchetti di sistema necessari
RUN apt update && apt install -y \
    python3.10 python3.10-venv python3.10-dev python3-pip \
    git wget curl unzip ffmpeg libgl1-mesa-glx \
    ninja-build build-essential \
    && apt clean


RUN ln -s /usr/local/cuda/include /usr/include/cuda \
    && ln -s /usr/local/cuda/lib64 /usr/lib/cuda

ENV CUDA_HOME=/usr/local/cuda
ENV CPLUS_INCLUDE_PATH=$CUDA_HOME/include
ENV LIBRARY_PATH=$CUDA_HOME/lib64
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Crea un ambiente virtuale e attivalo
RUN python3 -m venv venv
ENV PATH="/app/venv/bin:$PATH"

# Installa pip e PyTorch prima per sfruttare la cache
RUN apt-get update && apt-get install -y ninja-build \
 && pip install --upgrade pip \
 && pip install wheel \
 && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 \
 && pip install --no-build-isolation diso

# Copia solo il file requirements.txt per ottimizzare il caching
COPY requirements.txt /tmp/

# Installa le dipendenze senza invalidare la cache quando il codice cambia
RUN pip install -r /tmp/requirements.txt

# Ora copia il resto del codice (non invalida il caching delle dipendenze)
COPY . /app

# Installa le dipendenze aggiuntive richieste
#RUN pip install gradio==4.0.2 sentencepiece

# Imposta le variabili di ambiente per CUDA e Torch
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="$CUDA_HOME/bin:$PATH"
ENV LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# Imposta TORCH_CUDA_ARCH_LIST per evitare errori di compilazione
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0"
ENV HF_HOME=/huggingface

RUN conda install -y -c conda-forge ffmpeg



# OpenCV dependencies
RUN apt -y install libgl1

# Download the checkpoints required for inference from HuggingFace
RUN huggingface-cli download ByteDance/LatentSync-1.5 whisper/tiny.pt --local-dir checkpoints
RUN huggingface-cli download ByteDance/LatentSync-1.5 latentsync_unet.pt --local-dir checkpoints

# Verifica che NVCC sia disponibile
#RUN nvcc --version

#RUN pip install flash-attn --no-build-isolation

#RUN source patchtransformers.sh

# Espone la porta per Gradio
EXPOSE 8080

# Comando di default per avviare il server Gradio
CMD ["python3", "gradio_app.py", "--share", "--port", "8080"]
