# Base Image (Giữ nguyên)
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# HF_HOME cố định trong image (Network Volume mount tại /workspace).
ENV HF_HOME="/workspace/cache"
ENV PYTHONUNBUFFERED=1

# HF_TOKEN KHÔNG bake vào image.
# - Build-time: truyền qua `--build-arg HF_TOKEN=...` để builder.py tải model vào cache.
# - Runtime: truyền qua RunPod Console → Endpoint → Environment Variables.
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# 1. Cài các thư viện cơ bản trước
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 2. [THỦ THUẬT] Cài Pyannote 4.0.2 nhưng KHÔNG cài dependencies (để tránh tải lại Torch)
# Sau đó cài bù các thư viện phụ mà Pyannote cần (trừ Torch)
RUN pip install --no-cache-dir pyannote-audio==4.0.2 --no-deps && \
    pip install --no-cache-dir \
    "lightning>=2.0.0" \
    "pyannote-core>=6.0.1" \
    "pyannote-pipeline>=4.0.0" \
    "pyannote-database>=6.1.0" \
    "pyannote-metrics>=4.0.0" \
    "torch-audiomentations>=0.12.0" \
    "torchmetrics>=1.8.2" \
    "asteroid-filterbanks>=0.4.0" \
    "einops>=0.8.0" \
    "speechbrain>=1.0.0" \
    "semver>=3.0.0" \
    "omegaconf>=2.1" \
    "docopt>=0.6.2" \
    "opentelemetry-api>=1.0.0" \
    "opentelemetry-sdk>=1.0.0" \
    "opentelemetry-exporter-otlp>=1.0.0" \
    "tensorboard>=2.0.0" \
    "soundfile>=0.12.1" \
    "rich>=12.0.0"
RUN pip install nvidia-cudnn-cu12==9.1.0.70
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
COPY . .

RUN mkdir -p /workspace/cache
RUN python builder.py

CMD [ "python", "-u", "handler.py" ]