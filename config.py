#!/usr/bin/env python3
"""
Configuration constants for Parakeet RunPod Serverless.

This module contains all configuration parameters used throughout the project,
including paths, versions, and model settings.
"""

# =============================================================================
# Volume and Installation Paths
# =============================================================================

VOLUME_MOUNT = "/runpod-volume"
PARAKEET_DIR = f"{VOLUME_MOUNT}/Parakeet"
VENV_DIR = f"{PARAKEET_DIR}/venv"
# HF cache paths removed - RunPod manages model caching via serverless config

# Installation marker file - created when bootstrap completes successfully
INSTALLATION_MARKER = f"{PARAKEET_DIR}/.installation_complete"

# =============================================================================
# Model Configuration
# =============================================================================

MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v3"
MODEL_HUGGINGFACE_URL = f"https://huggingface.co/{MODEL_NAME}"

# =============================================================================
# PyTorch Version Configuration
# =============================================================================

PYTORCH_VERSION = "torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0"
PYTORCH_INDEX_URL = "https://download.pytorch.org/whl/cu128"

# =============================================================================
# Flash Attention Configuration
# =============================================================================

FLASH_ATTN_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/"
    "v2.8.1/flash_attn-2.8.1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
)
FLASH_ATTN_VERSION = "2.8.1"

# =============================================================================
# Audio Configuration
# =============================================================================

# Supported input formats from users
SUPPORTED_INPUT_FORMATS = [".m4a", ".ogg", ".opus", ".wav", ".mp3", ".flac"]

# Parakeet model requirements
PARAKEET_SAMPLE_RATE = 16000  # 16kHz
PARAKEET_CHANNELS = 1  # mono
PARAKEET_FORMATS = [".wav", ".flac"]

# Audio conversion settings
DEFAULT_OUTPUT_FORMAT = "wav"

# =============================================================================
# S3 Configuration (Environment Variables)
# =============================================================================

S3_ACCESS_KEY_ENV = "S3_ACCESS_KEY"
S3_SECRET_KEY_ENV = "S3_SECRET_KEY"
S3_ENDPOINT_ENV = "S3_ENDPOINT"
S3_BUCKET_ENV = "S3_BUCKET"

# =============================================================================
# Timeout and Performance Settings
# =============================================================================

# Maximum audio duration for full attention (24 minutes)
MAX_FULL_ATTENTION_DURATION_SEC = 24 * 60

# Maximum audio duration for local attention (3 hours)
MAX_LOCAL_ATTENTION_DURATION_SEC = 3 * 60 * 60

# S3 download timeout
S3_DOWNLOAD_TIMEOUT = 300  # 5 minutes

# =============================================================================
# NeMo Configuration
# =============================================================================

# Attention context sizes for long audio
ATTENTION_LEFT_CONTEXT = 256
ATTENTION_RIGHT_CONTEXT = 256

# =============================================================================
# Logging Configuration
# =============================================================================

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# =============================================================================
# Response Configuration
# =============================================================================

# Default timestamp flag value
DEFAULT_INCLUDE_TIMESTAMPS = False

# Timestamp types that can be requested
TIMESTAMP_TYPES = ["word", "segment", "char"]
