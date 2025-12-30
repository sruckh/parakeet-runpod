# Parakeet RunPod Serverless

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![CUDA 12.8](https://img.shields.io/badge/CUDA-12.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)

Speech-to-text serverless endpoint using NVIDIA's [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) model deployed on RunPod.

## Features

- **25 European Languages** - Automatic language detection, no prompting required
- **Multiple Audio Formats** - Supports .m4a, .ogg, .opus, .wav, .mp3, .flac input
- **Flexible Input Methods** - Base64 encoded audio or S3 pre-signed URLs
- **Optional Timestamps** - Word, segment, and character-level timestamps available
- **Long Audio Support** - Handles audio up to 24 minutes (full attention) or 3 hours (local attention)
- **Automatic Formatting** - Built-in punctuation and capitalization
- **Network-Attached Storage** - All software and models cached on shared volume for fast warm starts

## Supported Languages

Bulgarian (bg), Croatian (hr), Czech (cs), Danish (da), Dutch (nl), English (en), Estonian (et), Finnish (fi), French (fr), German (de), Greek (el), Hungarian (hu), Italian (it), Latvian (lv), Lithuanian (lt), Maltese (mt), Polish (pl), Portuguese (pt), Romanian (ro), Slovak (sk), Slovenian (sl), Spanish (es), Swedish (sv), Russian (ru), Ukrainian (uk)

## Quick Start

### 1. Deploy to RunPod

1. Connect this GitHub repository to RunPod
2. Configure GPU: **NVIDIA A6000** (48GB) or any GPU with 24GB+ VRAM
3. Configure Volume:
   - Mount point: `/runpod-volume`
   - Size: Minimum 20GB (for model + dependencies)
4. (Optional) Configure S3 environment variables for URL-based input
5. Deploy

### 2. First Request

The first request will trigger the bootstrap process (takes 5-15 minutes):
- Creates Python virtual environment on the volume
- Installs PyTorch 2.8.0 with CUDA 12.8
- Installs Flash Attention, NeMo toolkit, and dependencies
- Downloads Parakeet model (~2GB)

Subsequent requests skip installation and start immediately.

## API Usage

### Request Format

#### Base64 Audio Input

```json
{
  "audio_base64": "<base64 encoded audio data>",
  "timestamp": false
}
```

#### S3 Pre-signed URL Input

```json
{
  "audio_url": "<s3 presigned url>",
  "timestamp": false
}
```

#### With Timestamps

```json
{
  "audio_base64": "<base64 encoded audio data>",
  "timestamp": true
}
```

### Response Format

#### Success (Without Timestamps)

```json
{
  "text": "Transcribed text here with automatic punctuation and capitalization.",
  "success": true
}
```

#### Success (With Timestamps)

```json
{
  "text": "Transcribed text here.",
  "timestamps": {
    "word": [
      {"start": 0.0, "end": 0.25, "text": "Transcribed"},
      {"start": 0.25, "end": 0.5, "text": "text"}
    ],
    "segment": [
      {"start": 0.0, "end": 0.5, "segment": "Transcribed text"}
    ],
    "char": [...]
  },
  "success": true
}
```

#### Error

```json
{
  "error": "Error message describing what went wrong",
  "success": false
}
```

## Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `S3_ACCESS_KEY` | S3 access key for direct downloads | No | - |
| `S3_SECRET_KEY` | S3 secret key for direct downloads | No | - |
| `S3_ENDPOINT` | S3 endpoint URL (for S3-compatible services) | No | - |
| `S3_BUCKET` | Default S3 bucket name | No | - |

### GPU Requirements

| GPU | VRAM | Recommended |
|-----|------|-------------|
| NVIDIA A6000 | 48GB | ✅ Primary |
| NVIDIA A40 | 48GB | ✅ Yes |
| NVIDIA A100 | 40GB/80GB | ✅ Yes |
| NVIDIA A10G | 24GB | ✅ Minimum |
| NVIDIA RTX 6000 Ada | 48GB | ✅ Yes |

## Project Structure

```
parakeet/
├── Dockerfile           # Container build instructions
├── handler.py           # RunPod serverless handler (entry point)
├── bootstrap.py         # Installation and setup logic
├── config.py            # Configuration constants
├── requirements.txt     # Python dependencies (reference)
├── utils/
│   ├── __init__.py
│   ├── audio.py         # Audio conversion utilities
│   ├── s3.py            # S3 download utilities
│   └── validation.py    # Input validation
├── tests/
│   ├── __init__.py
│   ├── test_handler.py  # Handler tests
│   └── test_audio.py    # Audio conversion tests
├── .gitignore
├── README.md
└── PLAN.md              # Implementation plan
```

## How It Works

1. **Bootstrap (First Run Only)**
   - Checks for installation marker at `/runpod-volume/Parakeet/.installation_complete`
   - If missing, installs all dependencies to `/runpod-volume/Parakeet/venv/`
   - Downloads model to `/runpod-volume/Parakeet/hf_hub/cache/`
   - Creates marker file on success

2. **Request Processing**
   - Validates input (base64 or URL)
   - Downloads/decodes audio to temporary file
   - Converts to Parakeet-compatible format (16kHz, mono, .wav/.flac)
   - Loads model (cached after first request)
   - Runs transcription
   - Returns results

3. **Cleanup**
   - Temporary files deleted after each request
   - Model stays loaded in memory for subsequent requests

## Model Information

- **Model**: [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- **Architecture**: FastConformer-TDT
- **Parameters**: 600M
- **License**: [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/legalcode.en)
- **Sample Rate**: 16kHz
- **Channels**: Mono (1)

## Development

### Local Testing

```bash
# Build container
docker build -t parakeet-test .

# Run with volume mount
docker run -v $(pwd)/test-volume:/runpod-volume \
  --gpus all \
  -e NVIDIA_VISIBLE_DEVICES=all \
  parakeet-test

# Run bootstrap manually
docker run -v $(pwd)/test-volume:/runpod-volume \
  --entrypoint python3 \
  parakeet-test /app/bootstrap.py --check
```

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/

# Run with coverage
pytest --cov=utils --cov=handler tests/
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of memory during model load | Ensure GPU has 24GB+ VRAM |
| FFmpeg conversion fails | Verify ffmpeg is installed in container |
| Model download fails | Check network connectivity, HF credentials |
| Venv not found | Check volume mount and permissions |
| S3 download fails | Verify S3 credentials and endpoint |

## Performance Considerations

- **Cold Start**: First request triggers bootstrap (5-15 minutes)
- **Warm Start**: Subsequent requests start in seconds
- **Model Loading**: Model loads once and stays in memory
- **Long Audio**: For audio > 24 minutes, local attention is used automatically

## License

This project's code is provided under the MIT License.

The [Parakeet TDT 0.6B v3 model](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) is licensed under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/legalcode.en).

## References

- [NVIDIA Parakeet Model](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- [NVIDIA NeMo Documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/models.html)
- [RunPod Serverless Documentation](https://github.com/runpod/docs)
- [Parakeet Technical Report](https://arxiv.org/abs/2509.14128)
