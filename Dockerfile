# ============================================================================
# Parakeet RunPod Serverless - Dockerfile
# ============================================================================
# This Dockerfile creates a minimal container image for the Parakeet
# speech-to-text serverless endpoint. All heavy dependencies (NeMo, model files)
# are installed to the network-attached volume during bootstrap.
# =============================================================================

# Base image: NVIDIA CUDA with cuDNN
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Metadata
LABEL maintainer="Parakeet Serverless Team"
LABEL description="Parakeet TDT Speech-to-Text RunPod Serverless"
LABEL version="1.0.0"

# Set environment variables for non-interactive installation and Python
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set working directory
WORKDIR /app

# ---------------------------------------------------------------------------
# Install Python 3.12 and Essential System Packages
# ---------------------------------------------------------------------------

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    git \
    ca-certificates \
    curl \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    ffmpeg \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/bin/pip3 /usr/local/bin/pip

# Verify Python installation
RUN python --version && pip --version

# ---------------------------------------------------------------------------
# Install RunPod Serverless SDK
# ---------------------------------------------------------------------------

# This is the only Python package installed in the container itself.
# All other dependencies are installed to the volume during bootstrap.
RUN pip install --no-cache-dir --upgrade "runpod>=1.7.0"

# ---------------------------------------------------------------------------
# Copy Application Files
# ---------------------------------------------------------------------------

# Copy Python modules
COPY config.py /app/config.py
COPY bootstrap.py /app/bootstrap.py
COPY handler.py /app/handler.py

# Copy utility modules
COPY utils/__init__.py /app/utils/__init__.py
COPY utils/audio.py /app/utils/audio.py
COPY utils/s3.py /app/utils/s3.py
COPY utils/validation.py /app/utils/validation.py

# Copy requirements.txt for reference
COPY requirements.txt /app/requirements.txt

# ---------------------------------------------------------------------------
# Health Check (Optional)
# ---------------------------------------------------------------------------

# RunPod manages container health, but we can add a basic check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# ---------------------------------------------------------------------------
# Set the Entry Point
# ---------------------------------------------------------------------------

# The handler.py script will:
# 1. Run bootstrap on first launch (if needed)
# 2. Start the RunPod serverless worker
CMD ["python3", "-u", "handler.py"]
