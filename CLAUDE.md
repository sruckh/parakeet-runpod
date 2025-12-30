# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Required Commands

### Before ANY Task
```bash
codemap .                     # Project structure
codemap --deps                # How files connect
codemap --diff                # What changed vs main
codemap --diff --ref <branch> # Changes vs specific branch
```

**ALWAYS run `codemap .` BEFORE starting any task.**

**ALWAYS run `codemap --deps` when:**
- User asks how something works
- Refactoring or moving code
- Tracing imports or dependencies

**ALWAYS run `codemap --diff` when:**
- Reviewing or summarizing changes
- Before committing code
- User asks what changed

## Deployment Workflow

**This is a RunPod Serverless deployment - nothing runs locally.**

1. Make code changes
2. Push to GitHub
3. RunPod automatically builds the container image
4. Deploy to RunPod serverless endpoint

The container is built on RunPod's infrastructure, not on your local machine. RunPod is directly connected to this GitHub repository and triggers builds on every push.

## Common Development Commands

### Testing (Optional - if running tests locally)
```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_audio.py
pytest tests/test_handler.py

# Run with coverage
pytest --cov=utils --cov=handler tests/
```

### Syntax Check (before pushing to GitHub)
```bash
python3 -m py_compile *.py utils/*.py tests/*.py
```

### Local Docker Testing (Optional - only if you want to test before pushing)
```bash
# Build container locally
docker build -t parakeet-test .

# Run with GPU and volume mount
docker run -v $(pwd)/test-volume:/runpod-volume \
  --gpus all \
  -e NVIDIA_VISIBLE_DEVICES=all \
  parakeet-test

# Check bootstrap installation status
docker run -v $(pwd)/test-volume:/runpod-volume \
  --entrypoint python3 \
  parakeet-test /app/bootstrap.py --check
```

## Architecture Overview

### Request Flow
1. **handler.py** receives audio input (base64 or S3 URL)
2. **bootstrap.py** runs on first request only - installs all dependencies to `/runpod-volume/Parakeet/venv/`
3. **utils/validation.py** validates input format
4. **utils/s3.py** downloads audio if URL provided
5. **utils/audio.py** converts audio to 16kHz mono WAV/FLAC using ffmpeg
6. **handler.py** loads Parakeet model (cached in memory after first load) and transcribes
7. Returns JSON with text and optional timestamps

### Bootstrap Process (First Run Only)
The bootstrap creates a **persistent virtual environment on the volume** to avoid reinstalling on every cold start:

```
/runpod-volume/Parakeet/
├── venv/                           # Python venv with all dependencies
├── hf_home/                        # HuggingFace home directory
├── hf_hub/cache/                   # HuggingFace model cache
└── .installation_complete          # Marker file
```

Installation steps (5-15 minutes first run):
1. Create venv at `/runpod-volume/Parakeet/venv/`
2. Install PyTorch 2.8.0 with CUDA 12.8
3. Install Flash Attention 2.8.1 from wheel
4. Install NVIDIA NeMo toolkit with ASR
5. Install ffmpeg-python, boto3, huggingface_hub
6. Download Parakeet model to volume cache
7. Create marker file

**Key Pattern**: All heavy dependencies live on the network-attached volume, NOT in the container. This enables fast warm starts.

### Model Loading Pattern
`handler.py` uses a **global variable** to cache the loaded model:

```python
_asr_model = None  # Module-level variable

def load_model():
    global _asr_model
    if _asr_model is not None:
        return _asr_model
    # Setup venv Python path
    setup_python_path()
    # Import NeMo from venv
    import nemo.collections.asr as nemo_asr
    _asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_NAME)
    return _asr_model
```

**Critical**: NeMo must be imported AFTER adding venv site-packages to `sys.path`.

### Audio Conversion Pattern
All audio is converted to Parakeet-compatible format using ffmpeg-python:
- **Sample Rate**: 16kHz (PARAKEET_SAMPLE_RATE)
- **Channels**: Mono/1 (PARAKEET_CHANNELS)
- **Format**: WAV or FLAC (PARAKEET_FORMATS)

**Key Pattern**: Always use temporary files with context managers for cleanup.

### Configuration Management
All constants live in `config.py` and are imported as:
```python
import config
config.VOLUME_MOUNT  # Not: from config import VOLUME_MOUNT
```

## Code Style Requirements

### Type Hints
- **Required** on all functions
- Use `from typing import Any, Literal, TypedDict`
- Explicitly declare return types

### Docstrings
- Google-style format
- Required on all public functions
- Include Args, Returns, Raises sections

### Import Order
1. Standard library imports
2. Third-party imports
3. Local imports
Separate each group with a blank line.

### Naming Conventions
- Constants: `UPPER_SNAKE_CASE`
- Functions/variables: `lower_snake_case`
- Classes: `PascalCase`
- Module-level private: `_leading_underscore`

### String Formatting
- Prefer f-strings
- For errors: `f"audio_base64 must be a string, got {type(value).__name__}"`

### Exception Handling
- Raise specific exceptions with descriptive messages
- Handler responses always include `"success"` field:
  - Success: `{"text": "...", "success": true}` (with optional `"timestamps"`)
  - Error: `{"error": "message", "success": false}`
- Always clean up temp files in try/finally blocks

## Important Technical Details

### Volume Mount Point
All persistent data lives at `/runpod-volume`:
- Venv: `/runpod-volume/Parakeet/venv/`
- HuggingFace home: `/runpod-volume/Parakeet/hf_home/`
- Model cache: `/runpod-volume/Parakeet/hf_hub/cache/`
- Installation marker: `/runpod-volume/Parakeet/.installation_complete`

### Environment Variables (Optional, S3 Only)
- `S3_ACCESS_KEY`: S3 credentials
- `S3_SECRET_KEY`: S3 credentials
- `S3_ENDPOINT`: S3-compatible endpoint URL
- `S3_BUCKET`: Default bucket name

### GPU Requirements
- **Minimum**: 24GB VRAM
- **Recommended**: A6000 (48GB), A40 (48GB), A100 (40GB/80GB)

### Attention Model Switching
- **Full attention**: Audio ≤ 24 minutes (MAX_FULL_ATTENTION_DURATION_SEC = 1440 seconds)
- **Local attention**: Audio > 24 minutes, up to 3 hours (MAX_LOCAL_ATTENTION_DURATION_SEC = 10800 seconds)
- Context: ATTENTION_LEFT_CONTEXT=256, ATTENTION_RIGHT_CONTEXT=256

### Supported Languages
25 European languages with automatic detection: bg, hr, cs, da, nl, en, et, fi, fr, de, el, hu, it, lv, lt, mt, pl, pt, ro, sk, sl, es, sv, ru, uk

## Testing Notes

### Test Structure
- `tests/test_audio.py`: Audio conversion utilities
- `tests/test_handler.py`: Handler validation and flow

### Running Single Tests
```bash
pytest tests/test_audio.py::TestAudioConversion::test_convert_wav -v
```

## Key Files and Their Roles

| File | Purpose |
|------|---------|
| `handler.py` | RunPod serverless entry point, model loading, transcription |
| `bootstrap.py` | First-run installation orchestration |
| `config.py` | All constants and configuration |
| `utils/audio.py` | FFmpeg audio conversion to 16kHz mono |
| `utils/s3.py` | S3 pre-signed URL download |
| `utils/validation.py` | Input validation and base64 decoding |
| `Dockerfile` | Container with system deps (Python, ffmpeg, build tools) |

## Dependency Hubs
From `codemap --deps`:
- **validation** (3← imports): Core validation used by handler and tests
- **audio** (3← imports): Audio conversion used by handler and tests
- **s3** (2← imports): S3 download functionality
