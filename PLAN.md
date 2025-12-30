# Parakeet RunPod Serverless - Implementation Plan

## Project Overview

Build a RunPod serverless container that provides speech-to-text functionality using NVIDIA's Parakeet TDT 0.6B v3 model. The serverless will accept audio files in multiple formats, convert them as needed, and return transcribed text with optional timestamps.

**Model**: [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
**Base Image**: `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04`
**Python Version**: 3.12
**GPU Requirement**: A6000 (48GB) or any 24GB+ VRAM GPU

---

## Architecture Summary

```
User Request (Audio)
       |
       v
[RunPod Serverless Handler]
       |
       +---> Bootstrap Process
       |     |
       |     +-- Check if /runpod-volume/Parakeet/venv exists
       |     |
       |     +-- If not: First-time installation
       |     |     - Create Python venv at /runpod-volume/Parakeet/venv
       |     |     - Install PyTorch 2.8.0 (CUDA 12.8)
       |     |     - Install flash-attn 2.8.1
       |     |     - Install nemo_toolkit['asr']
       |     |     - Install huggingface_hub, ffmpeg-python, etc.
       |     |     - Set HF_HOME and HF_HUB_CACHE
       |     |     - Download Parakeet model
       |     |     - Create installation marker file
       |     |
       |     +-- If yes: Skip installation, proceed to inference
       |
       +---> Audio Input Processing
       |     |
       |     +-- Detect input type (base64 OR S3 pre-signed URL)
       |     |
       |     +-- If S3 URL: Download using boto3 (credentials from env vars)
       |     |
       |     +-- If base64: Decode to bytes
       |     |
       |     +-- Convert audio to 16kHz mono .wav (using ffmpeg)
       |
       +---> Load Parakeet Model (from /runpod-volume/Parakeet cache)
       |
       +---> Run Transcription
       |     |
       |     +-- If timestamp=false: Return transcribed text only
       |     +-- If timestamp=true: Return text with word/segment timestamps
       |
       v
Return JSON Response to User
```

---

## Directory Structure

```
parakeet/
├── Dockerfile                 # Container build instructions
├── handler.py                 # RunPod serverless handler (entry point)
├── requirements.txt           # Python dependencies (for reference only)
├── config.py                  # Configuration constants
├── bootstrap.py               # Installation and setup logic
├── utils/
│   ├── audio.py               # Audio conversion utilities
│   ├── s3.py                  # S3 download utilities
│   └── validation.py          # Input validation
├── tests/
│   ├── test_handler.py        # Handler tests
│   └── test_audio.py          # Audio conversion tests
├── .gitignore
├── README.md                  # Project documentation
└── PLAN.md                    # This file
```

---

## Implementation Steps

### Step 1: Create Project Structure and Configuration Files

#### 1.1 Create `config.py` - Configuration Constants

```python
# Volume and installation paths
VOLUME_MOUNT = "/runpod-volume"
PARAKEET_DIR = f"{VOLUME_MOUNT}/Parakeet"
VENV_DIR = f"{PARAKEET_DIR}/venv"
HF_HOME_DIR = f"{PARAKEET_DIR}/hf_home"
HF_HUB_CACHE = f"{PARAKEET_DIR}/hf_hub_cache"

# Installation marker file
INSTALLATION_MARKER = f"{PARAKEET_DIR}/.installation_complete"

# Model configuration
MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v3"

# PyTorch version
PYTORCH_VERSION = "torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0"
PYTORCH_INDEX_URL = "https://download.pytorch.org/whl/cu128"

# Flash attention version
FLASH_ATTN_WHEEL = "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"

# Supported audio input formats
SUPPORTED_INPUT_FORMATS = [".m4a", ".ogg", ".opus", ".wav", ".mp3", ".flac"]

# Parakeet audio requirements
PARAKEET_SAMPLE_RATE = 16000
PARAKEET_CHANNELS = 1  # mono
PARAKEET_FORMATS = [".wav", ".flac"]

# S3 Environment variables
S3_ACCESS_KEY_ENV = "S3_ACCESS_KEY"
S3_SECRET_KEY_ENV = "S3_SECRET_KEY"
S3_ENDPOINT_ENV = "S3_ENDPOINT"
S3_BUCKET_ENV = "S3_BUCKET"
```

#### 1.2 Create `requirements.txt` - Python Dependencies Reference

```
# Note: These are installed to the venv during bootstrap
# This file is for reference only

# Core ML
torch==2.8.0
torchvision==0.23.0
torchaudio==2.8.0

# Flash Attention (installed via wheel)
# flash_attn @ https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

# NVIDIA NeMo
nemo-toolkit[asr]

# Hugging Face
huggingface_hub

# Audio processing
ffmpeg-python

# S3 support (optional, for URL downloads)
boto3

# RunPod SDK
runpod
```

---

### Step 2: Create Dockerfile

The Dockerfile should be minimal with only runtime dependencies. All heavy packages are installed to the volume during bootstrap.

```dockerfile
# nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 as the base image
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PYTHONUNBUFFERED=1

# Working directory
WORKDIR /app

# Install Python 3.12 and essential system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev python3-pip \
    git ca-certificates curl build-essential cmake ninja-build pkg-config ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && ln -sf /usr/bin/pip3 /usr/local/bin/pip

# Install runpod SDK (minimal, for serverless handler)
RUN pip install --no-cache-dir runpod

# Copy application files
COPY handler.py config.py bootstrap.py /app/
COPY utils/ /app/utils/

# Set the entry point
CMD ["python3", "-u", "handler.py"]
```

---

### Step 3: Create Bootstrap Module (`bootstrap.py`)

This module handles first-time installation and verification of the software stack.

```python
#!/usr/bin/env python3
"""
Bootstrap module for Parakeet RunPod Serverless.
Handles first-time installation and verification of the software stack.
"""

import os
import subprocess
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def run_command(cmd, cwd=None, env=None):
    """Run a shell command and return success status."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=cwd,
        env=env or os.environ.copy(),
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        return False
    print(result.stdout)
    return True


def is_installed():
    """Check if installation has already been completed."""
    marker = Path(config.INSTALLATION_MARKER)
    venv_python = Path(config.VENV_DIR) / "bin" / "python"

    if not marker.exists():
        return False
    if not venv_python.exists():
        return False

    # Verify marker contents include version info
    try:
        content = marker.read_text()
        # Could add version verification here
        return True
    except Exception:
        return False


def create_directories():
    """Create necessary directories on the volume."""
    dirs = [
        config.PARAKEET_DIR,
        config.HF_HOME_DIR,
        config.HF_HUB_CACHE,
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print(f"Created directories: {dirs}")


def create_virtual_environment():
    """Create Python virtual environment."""
    cmd = [sys.executable, "-m", "venv", config.VENV_DIR]
    return run_command(cmd)


def get_venv_python():
    """Get path to Python executable in venv."""
    return str(Path(config.VENV_DIR) / "bin" / "python")


def get_venv_pip():
    """Get path to pip executable in venv."""
    return str(Path(config.VENV_DIR) / "bin" / "pip")


def install_pytorch():
    """Install PyTorch with CUDA 12.8 support."""
    cmd = [
        get_venv_pip(), "install",
        *config.PYTORCH_VERSION.split(),
        "--index-url", config.PYTORCH_INDEX_URL
    ]
    return run_command(cmd)


def install_flash_attn():
    """Install flash-attention from wheel."""
    cmd = [get_venv_pip(), "install", config.FLASH_ATTN_WHEEL]
    return run_command(cmd)


def install_nemo():
    """Install NVIDIA NeMo toolkit with ASR support."""
    cmd = [get_venv_pip(), "install", "nemo-toolkit[asr]"]
    return run_command(cmd)


def install_huggingface_hub():
    """Install Hugging Face Hub CLI."""
    cmd = [get_venv_pip(), "install", "huggingface_hub"]
    return run_command(cmd)


def install_other_dependencies():
    """Install additional dependencies."""
    dependencies = [
        "ffmpeg-python",
        "boto3",  # For S3 downloads
    ]
    cmd = [get_venv_pip(), "install", *dependencies]
    return run_command(cmd)


def download_model():
    """Download Parakeet model using Python API (uses HF env vars)."""
    # Set environment variables for download location
    env = os.environ.copy()
    env["HF_HOME"] = config.HF_HOME_DIR
    env["HF_HUB_CACHE"] = config.HF_HUB_CACHE

    # Download using hf CLI
    cmd = [
        get_venv_pip(), "-m", "huggingface_hub.cli",
        "download", config.MODEL_NAME,
        "--local-dir", f"{config.HF_HUB_CACHE}/{config.MODEL_NAME}",
        "--local-dir-use-symlinks", "False"
    ]
    return run_command(cmd, env=env)


def mark_installation_complete():
    """Create marker file to indicate successful installation."""
    marker = Path(config.INSTALLATION_MARKER)
    marker.write_text("installation_complete\n")
    print(f"Created installation marker: {marker}")


def install_all():
    """Run full installation process."""
    print("=== Starting Parakeet Bootstrap Installation ===")

    steps = [
        ("Creating directories", create_directories),
        ("Creating virtual environment", create_virtual_environment),
        ("Installing PyTorch", install_pytorch),
        ("Installing flash-attention", install_flash_attn),
        ("Installing NeMo toolkit", install_nemo),
        ("Installing Hugging Face Hub", install_huggingface_hub),
        ("Installing additional dependencies", install_other_dependencies),
        ("Downloading Parakeet model", download_model),
        ("Marking installation complete", mark_installation_complete),
    ]

    for step_name, step_func in steps:
        print(f"\n--- {step_name} ---")
        if not step_func():
            print(f"FAILED: {step_name}")
            return False

    print("\n=== Bootstrap Installation Complete ===")
    return True


def bootstrap_if_needed():
    """Check if installation is needed and run if necessary."""
    if is_installed():
        print("Software already installed. Skipping bootstrap.")
        return True

    print("First run detected. Starting bootstrap installation...")
    return install_all()


if __name__ == "__main__":
    success = bootstrap_if_needed()
    sys.exit(0 if success else 1)
```

---

### Step 4: Create Audio Utilities (`utils/audio.py`)

```python
"""
Audio conversion utilities.
Ensures audio is compatible with Parakeet (16kHz, mono, .wav or .flac).
"""

import os
import tempfile
from pathlib import Path
import ffmpeg

from config import PARAKEET_SAMPLE_RATE, PARAKEET_CHANNELS, PARAKEET_FORMATS


def convert_audio_for_parakeet(input_path: str, output_format: str = "wav") -> str:
    """
    Convert audio file to Parakeet-compatible format.

    Args:
        input_path: Path to input audio file
        output_format: Output format ("wav" or "flac")

    Returns:
        Path to converted audio file
    """
    # Verify input file exists
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input audio file not found: {input_path}")

    # Create temp file for output
    suffix = f".{output_format}"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        output_path = tmp.name

    try:
        # Use ffmpeg to convert
        stream = ffmpeg.input(input_path)
        stream = ffmpeg.output(
            stream,
            output_path,
            acodec="pcm_s16le" if output_format == "wav" else "flac",
            ac=str(PARAKEET_CHANNELS),
            ar=str(PARAKEET_SAMPLE_RATE)
        )
        ffmpeg.run(stream, overwrite_output=True, quiet=True)

        return output_path
    except ffmpeg.Error as e:
        os.unlink(output_path)
        raise RuntimeError(f"FFmpeg conversion failed: {e.stderr.decode('utf8')}")
    except Exception as e:
        os.unlink(output_path)
        raise


def get_audio_duration(file_path: str) -> float:
    """Get audio file duration in seconds using ffprobe."""
    try:
        probe = ffmpeg.probe(file_path)
        return float(probe['format']['duration'])
    except Exception as e:
        raise RuntimeError(f"Failed to get audio duration: {e}")


def is_valid_audio_file(file_path: str) -> bool:
    """Check if file is a valid audio file."""
    path = Path(file_path)
    if not path.exists():
        return False
    # Check extension
    valid_extensions = [".m4a", ".ogg", ".opus", ".wav", ".mp3", ".flac"]
    return path.suffix.lower() in valid_extensions
```

---

### Step 5: Create S3 Utilities (`utils/s3.py`)

```python
"""
S3 download utilities for fetching audio files from pre-signed URLs.
"""

import os
import tempfile
import boto3
from botocore.exceptions import ClientError
from config import (
    S3_ACCESS_KEY_ENV, S3_SECRET_KEY_ENV,
    S3_ENDPOINT_ENV, S3_BUCKET_ENV
)


def download_from_presigned_url(presigned_url: str) -> str:
    """
    Download audio file from S3 pre-signed URL.

    Args:
        presigned_url: S3 pre-signed URL

    Returns:
        Path to downloaded file
    """
    import requests

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        try:
            response = requests.get(presigned_url, stream=True, timeout=300)
            response.raise_for_status()

            for chunk in response.iter_content(chunk_size=8192):
                tmp.write(chunk)

            return tmp.name
        except Exception as e:
            tmp.close()
            os.unlink(tmp.name)
            raise RuntimeError(f"Failed to download from S3: {e}")


def download_from_s3_direct(bucket: str, key: str) -> str:
    """
    Download file directly from S3 using credentials.
    Requires S3 credentials in environment variables.

    Args:
        bucket: S3 bucket name
        key: S3 object key

    Returns:
        Path to downloaded file
    """
    # Get S3 credentials from environment
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv(S3_ACCESS_KEY_ENV),
        aws_secret_access_key=os.getenv(S3_SECRET_KEY_ENV),
        endpoint_url=os.getenv(S3_ENDPOINT_ENV),
    )

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        try:
            s3_client.download_fileobj(bucket, key, tmp)
            return tmp.name
        except ClientError as e:
            tmp.close()
            os.unlink(tmp.name)
            raise RuntimeError(f"S3 download failed: {e}")
```

---

### Step 6: Create Validation Utilities (`utils/validation.py`)

```python
"""
Input validation utilities.
"""

import base64
import mimetypes
from config import SUPPORTED_INPUT_FORMATS


def validate_audio_input(data: dict) -> tuple:
    """
    Validate and determine the type of audio input.

    Args:
        data: Input data dictionary

    Returns:
        Tuple of (input_type, audio_data) where:
        - input_type: "base64" or "url"
        - audio_data: The base64 string or URL

    Raises:
        ValueError: If input is invalid
    """
    if not isinstance(data, dict):
        raise ValueError("Input must be a dictionary")

    # Check for base64 audio
    if "audio_base64" in data:
        audio_base64 = data["audio_base64"]
        if not isinstance(audio_base64, str) or not audio_base64:
            raise ValueError("audio_base64 must be a non-empty string")

        # Optionally verify format from mime type if provided
        if "audio_format" in data:
            fmt = data["audio_format"]
            if not fmt.startswith("."):
                fmt = f".{fmt}"
            if fmt.lower() not in SUPPORTED_INPUT_FORMATS:
                raise ValueError(f"Unsupported audio format: {fmt}")

        return "base64", audio_base64

    # Check for URL input
    elif "audio_url" in data:
        url = data["audio_url"]
        if not isinstance(url, str) or not url:
            raise ValueError("audio_url must be a non-empty string")

        # Could add URL validation here
        return "url", url

    else:
        raise ValueError("Input must contain either 'audio_base64' or 'audio_url'")


def decode_base64_audio(audio_base64: str, output_format: str = "wav") -> tuple:
    """
    Decode base64 audio data to a temporary file.

    Args:
        audio_base64: Base64 encoded audio data
        output_format: Desired output format (default: "wav")

    Returns:
        Tuple of (file_path, original_format) if successful
    """
    import tempfile
    import os

    # Detect format from data URI or use default
    original_format = None
    if "," in audio_base64:
        # Data URI format: data:audio/mp3;base64,...
        header, data = audio_base64.split(",", 1)
        audio_base64 = data
        if "audio/" in header:
            mime = header.split(";")[0].split("/")[1]
            original_format = f".{mime}"

    # Decode base64
    audio_bytes = base64.b64decode(audio_base64)

    # Create temp file
    suffix = original_format or f".{output_format}"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        return tmp.name, original_format


def validate_timestamp_flag(data: dict) -> bool:
    """
    Extract and validate the timestamp flag.

    Args:
        data: Input data dictionary

    Returns:
        Boolean value for timestamp flag (default: False)
    """
    return bool(data.get("timestamp", False))
```

---

### Step 7: Create RunPod Serverless Handler (`handler.py`)

```python
#!/usr/bin/env python3
"""
RunPod Serverless Handler for Parakeet Speech-to-Text.

This handler accepts audio files (base64 or URL) and returns transcribed text.
"""

import os
import sys
import json
import tempfile
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import runpod
from bootstrap import bootstrap_if_needed, get_venv_python
from config import (
    MODEL_NAME, PARAKEET_DIR, VENV_DIR,
    HF_HOME_DIR, HF_HUB_CACHE
)
from utils.audio import convert_audio_for_parakeet
from utils.validation import validate_audio_input, decode_base64_audio, validate_timestamp_flag
from utils.s3 import download_from_presigned_url


# Global model variable
_asr_model = None


def get_venv_site_packages():
    """Get path to venv site-packages for imports."""
    return str(Path(VENV_DIR) / "lib" / "python3.12" / "site-packages")


def setup_python_path():
    """Add venv site-packages to Python path."""
    site_packages = get_venv_site_packages()
    if site_packages not in sys.path:
        sys.path.insert(0, site_packages)


def load_model():
    """
    Load the Parakeet ASR model.
    Uses the virtual environment's Python.
    """
    global _asr_model

    if _asr_model is not None:
        return _asr_model

    # Setup Python path for venv
    setup_python_path()

    # Set HF environment variables
    os.environ["HF_HOME"] = HF_HOME_DIR
    os.environ["HF_HUB_CACHE"] = HF_HUB_CACHE

    # Import NeMo after setting up path
    import nemo.collections.asr as nemo_asr

    print(f"Loading model: {MODEL_NAME}")
    _asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_NAME)
    print("Model loaded successfully")

    return _asr_model


def transcribe_audio(audio_path: str, include_timestamps: bool = False):
    """
    Transcribe audio file using Parakeet model.

    Args:
        audio_path: Path to audio file (16kHz, mono, .wav or .flac)
        include_timestamps: Whether to include timestamps in output

    Returns:
        Transcription result
    """
    model = load_model()

    # Prepare transcription options
    transcribe_kwargs = {
        "audio": [audio_path],
        "timestamps": include_timestamps,
    }

    # For long audio, adjust attention model
    # (This can be configured based on audio length)

    result = model.transcribe(**transcribe_kwargs)

    return result[0]


def handler(event):
    """
    RunPod serverless handler function.

    Input JSON format:
    {
        "audio_base64": "<base64 encoded audio>",  // OR
        "audio_url": "<s3 presigned url>",
        "timestamp": false  // optional, default false
    }

    Output JSON format:
    {
        "text": "transcribed text",
        "timestamps": [...]  // only if timestamp=true
    }
    """
    try:
        print("Handler invoked")

        # Parse input
        if isinstance(event, str):
            data = json.loads(event)
        else:
            data = event

        # Validate and get input type
        input_type, audio_data = validate_audio_input(data)
        include_timestamps = validate_timestamp_flag(data)

        audio_path = None
        converted_path = None

        try:
            # Get audio file
            if input_type == "base64":
                audio_path, _ = decode_base64_audio(audio_data)
            else:  # URL
                audio_path = download_from_presigned_url(audio_data)

            # Convert to Parakeet-compatible format
            converted_path = convert_audio_for_parakeet(audio_path)

            # Transcribe
            result = transcribe_audio(converted_path, include_timestamps)

            # Build response
            response = {
                "text": result.text
            }

            if include_timestamps and hasattr(result, 'timestamp'):
                response["timestamps"] = result.timestamp

            return response

        finally:
            # Clean up temporary files
            if audio_path and os.path.exists(audio_path):
                os.unlink(audio_path)
            if converted_path and os.path.exists(converted_path):
                os.unlink(converted_path)

    except Exception as e:
        print(f"Error in handler: {e}")
        import traceback
        traceback.print_exc()
        return {
            "error": str(e)
        }


def main():
    """Main entry point for the container."""
    print("=== Parakeet RunPod Serverless Starting ===")

    # Run bootstrap if needed (first run)
    if not bootstrap_if_needed():
        print("ERROR: Bootstrap failed")
        sys.exit(1)

    print("Starting RunPod serverless handler...")
    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()
```

---

### Step 8: Create Additional Files

#### 8.1 `.gitignore`

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Testing
.pytest_cache/
.coverage
htmlcov/

# Temporary files
*.tmp
*.tmp.*
.DS_Store
```

#### 8.2 `README.md`

```markdown
# Parakeet RunPod Serverless

Speech-to-text serverless endpoint using NVIDIA Parakeet TDT 0.6B v3 model.

## Features

- Supports 25 European languages with automatic language detection
- Multiple audio input formats: .m4a, .ogg, .opus, .wav, .mp3, .flac
- Base64 or S3 pre-signed URL input methods
- Optional timestamps output
- Long audio support (up to 24 minutes with full attention)
- Automatic punctuation and capitalization

## Deployment

1. Connect this GitHub repository to RunPod
2. Configure the following environment variables:
   - `S3_ACCESS_KEY` (optional, for S3 downloads)
   - `S3_SECRET_KEY` (optional, for S3 downloads)
   - `S3_ENDPOINT` (optional, for S3 downloads)
   - `S3_BUCKET` (optional, for S3 downloads)
3. Select GPU: A6000 (48GB) or any 24GB+ GPU
4. Deploy

## API Usage

### Request Format

```json
{
  "audio_base64": "<base64 encoded audio data>"
}
```

OR

```json
{
  "audio_url": "<s3 presigned url>"
}
```

With timestamps:

```json
{
  "audio_base64": "<base64 encoded audio data>",
  "timestamp": true
}
```

### Response Format

```json
{
  "text": "Transcribed text here."
}
```

With timestamps:

```json
{
  "text": "Transcribed text here.",
  "timestamps": {
    "word": [...],
    "segment": [...],
    "char": [...]
  }
}
```

## Model Information

- Model: [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- License: CC-BY-4.0
- Architecture: FastConformer-TDT
- Parameters: 600M
```

---

## Step 9: Testing Strategy

### 9.1 Unit Tests

Create test files to verify individual components:

- `tests/test_audio.py`: Test audio conversion
- `tests/test_validation.py`: Test input validation
- `tests/test_s3.py`: Test S3 download (mocked)

### 9.2 Integration Testing

1. Build container locally: `docker build -t parakeet-test .`
2. Run with volume mount: `docker run -v $(pwd)/test-volume:/runpod-volume parakeet-test`
3. Test with sample audio files

### 9.3 Test Cases

| Test Case | Description |
|-----------|-------------|
| Base64 WAV input | Send base64 encoded WAV file |
| Base64 MP3 input | Send base64 encoded MP3 file (should convert) |
| S3 URL input | Download and transcribe from S3 URL |
| Timestamp flag false | Return only text |
| Timestamp flag true | Return text with timestamps |
| Multi-language | Test with various supported languages |
| Long audio | Test with audio > 5 minutes |
| Invalid input | Test error handling for invalid inputs |

---

## Step 10: RunPod Deployment Configuration

### 10.1 Environment Variables (Optional, for S3)

| Variable | Description | Example |
|----------|-------------|---------|
| `S3_ACCESS_KEY` | S3 access key | `your-access-key` |
| `S3_SECRET_KEY` | S3 secret key | `your-secret-key` |
| `S3_ENDPOINT` | S3 endpoint URL | `https://s3.amazonaws.com` |
| `S3_BUCKET` | Default S3 bucket | `my-audio-bucket` |

### 10.2 GPU Selection

Recommended GPUs:
- **NVIDIA A6000** (48GB VRAM) - Primary target
- **NVIDIA A40** (48GB VRAM)
- **NVIDIA A100** (40GB or 80GB VRAM)
- Any GPU with 24GB+ VRAM

### 10.3 Volume Configuration

- Mount Point: `/runpod-volume`
- Size: Minimum 20GB recommended (for model + venv + cache)
- Storage Class: SSD/Network-attached storage

---

## Step 11: Installation Verification Checklist

After first run, verify the following:

- [ ] `/runpod-volume/Parakeet/venv` directory exists
- [ ] `/runpod-volume/Parakeet/.installation_complete` marker exists
- [ ] `nemo_toolkit` is installed in venv
- [ ] Model files exist in `/runpod-volume/Parakeet/hf_hub/cache/`
- [ ] Transcription works without re-installing

---

## Step 12: Error Handling

### Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| Out of memory during model load | Ensure GPU has 24GB+ VRAM |
| FFmpeg conversion fails | Verify ffmpeg is installed in container |
| Model download fails | Check network connectivity, HF credentials |
| Venv not found | Check volume mount and permissions |
| S3 download fails | Verify S3 credentials and endpoint |

---

## Step 13: Performance Considerations

1. **Cold Start**: First request will trigger bootstrap (can take several minutes)
2. **Warm Start**: Subsequent requests skip installation (faster)
3. **Model Loading**: Model stays loaded in memory for subsequent requests
4. **Long Audio**: Consider chunking for audio > 24 minutes

---

## Implementation Order

1. Create project structure and config files
2. Write Dockerfile
3. Implement bootstrap.py
4. Implement utils (audio.py, s3.py, validation.py)
5. Implement handler.py
6. Create README.md and .gitignore
7. Test locally with Docker
8. Deploy to RunPod and test
9. Iterate based on testing results

---

## Dependencies Summary

### System Dependencies (in Dockerfile)
- Python 3.12
- ffmpeg
- build-essential, cmake, ninja-build (for compiling flash-attn)

### Python Dependencies (installed to venv on volume)
- torch==2.8.0 (CUDA 12.8)
- torchvision==0.23.0
- torchaudio==2.8.0
- flash_attn==2.8.1 (from wheel)
- nemo-toolkit[asr]
- huggingface_hub
- ffmpeg-python
- boto3
- runpod (installed in container, not venv)

---

## Notes

1. **No server-side installation**: Everything is containerized or on the volume
2. **Installation persistence**: Bootstrap only runs on first cold start
3. **Model caching**: Model is downloaded once and cached on the volume
4. **Python isolation**: All Python packages installed to venv on volume
5. **Environment variables**: HF_HOME and HF_HUB_CACHE ensure model downloads to volume

---

*This implementation plan provides a complete roadmap for building the Parakeet RunPod serverless endpoint. Follow the steps in order for systematic development.*
