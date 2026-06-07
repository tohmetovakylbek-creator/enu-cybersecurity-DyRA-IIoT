# ─────────────────────────────────────────────────────────────────────────────
# DyRA-IIoT  —  Dockerfile
# CUDA 12.1 + Python 3.11 base (adjust tag for your GPU driver)
# CPU-only build: replace base image with python:3.11-slim
# ─────────────────────────────────────────────────────────────────────────────
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

LABEL org.opencontainers.image.title="DyRA-IIoT"
LABEL org.opencontainers.image.description="Replication environment for DyRA-IIoT: A Hybrid Framework for Asset-Aware Dynamic Risk Assessment in IIoT Networks"
LABEL org.opencontainers.image.authors="Akylbek Tokhmetov <tokhmetov_ab@enu.kz>"

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        wget \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ────────────────────────────────────────────────────────
WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── Copy project source ────────────────────────────────────────────────────────
COPY . /workspace/DyRA-IIoT
WORKDIR /workspace/DyRA-IIoT
RUN pip install --no-cache-dir -e .

# ── Data volume mount point ────────────────────────────────────────────────────
# Mount your dataset directory here:
#   docker run -v /your/data:/data ...
VOLUME ["/data"]

# ── Default command: print help ────────────────────────────────────────────────
CMD ["python", "scripts/train_all.py", "--help"]
