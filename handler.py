#!/usr/bin/env python3
"""
RunPod Serverless Handler for Parakeet Speech-to-Text.

This handler accepts audio files (base64 or URL) and returns transcribed text
using NVIDIA's Parakeet TDT 0.6B v3 model.

Supported features:
- 25 European languages with automatic language detection
- Multiple audio formats: .m4a, .ogg, .opus, .wav, .mp3, .flac
- Base64 or S3 pre-signed URL input methods
- Optional timestamps output (word, segment, character level)
- Long audio support (up to 24 minutes)

Author: Parakeet Serverless Team
Version: 1.0.0
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import traceback
from logging import basicConfig, getLogger
from pathlib import Path
from typing import Any

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config

# Import bootstrap and utilities
from bootstrap import bootstrap_if_needed, get_venv_python, get_venv_site_packages
from utils.audio import convert_audio_for_parakeet, get_audio_duration
from utils.s3 import download_from_presigned_url
from utils.validation import (
    decode_base64_audio,
    validate_audio_input,
    validate_timestamp_flag,
)


# =============================================================================
# Logging Setup
# =============================================================================

basicConfig(
    level=config.LOG_LEVEL,
    format=config.LOG_FORMAT,
    stream=sys.stdout,
)
logger = getLogger("parakeet_handler")


# =============================================================================
# Global Model Cache
# =============================================================================

_asr_model = None
_nemo_loaded = False


# =============================================================================
# Virtual Environment Management
# =============================================================================


def setup_python_path():
    """
    Add virtual environment site-packages to Python path.

    This allows importing NeMo and other packages installed on the volume.
    """
    site_packages = get_venv_site_packages()

    if site_packages not in sys.path:
        sys.path.insert(0, site_packages)
        logger.info(f"Added venv site-packages to path: {site_packages}")


def setup_huggingface_cache():
    """
    Set Hugging Face environment variables for model cache location.

    Ensures models are downloaded to and read from the volume.
    """
    os.environ["HF_HOME"] = config.HF_HOME_DIR
    os.environ["HF_HUB_CACHE"] = config.HF_HUB_CACHE
    os.environ["HF_DATASETS_CACHE"] = f"{config.HF_HOME_DIR}/datasets"
    logger.info(f"HF_HOME set to: {config.HF_HOME_DIR}")
    logger.info(f"HF_HUB_CACHE set to: {config.HF_HUB_CACHE}")


# =============================================================================
# Model Loading
# =============================================================================


def load_model():
    """
    Load the Parakeet ASR model.

    The model is loaded once and cached in memory for subsequent requests.
    Uses the virtual environment's Python packages.

    Returns:
        Loaded NeMo ASR model

    Raises:
        RuntimeError: If model loading fails
    """
    global _asr_model, _nemo_loaded

    if _asr_model is not None:
        logger.info("Model already loaded, reusing cached instance")
        return _asr_model

    logger.info(f"Loading model: {config.MODEL_NAME}")

    # Setup Python path for venv
    setup_python_path()

    # Setup HF cache environment
    setup_huggingface_cache()

    try:
        # Import NeMo after setting up path
        import nemo.collections.asr as nemo_asr

        _nemo_loaded = True
        logger.info("NeMo imported successfully")

    except ImportError as e:
        logger.error(f"Failed to import NeMo: {e}")
        raise RuntimeError(
            f"NeMo toolkit not found in virtual environment. "
            f"Ensure bootstrap has completed successfully: {e}"
        )

    try:
        # Load the model
        _asr_model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=config.MODEL_NAME
        )
        logger.info("Model loaded successfully")

        # Disable CUDA graphs to avoid CUDA error 35
        try:
            _asr_model.decoding.decoding.decoding_computer.disable_cuda_graphs()
            logger.info("CUDA graphs disabled successfully")
        except Exception as e:
            logger.warning(f"Could not disable CUDA graphs: {e}")

        return _asr_model

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Model loading failed: {e}")


# =============================================================================
# Transcription
# =============================================================================


def transcribe_audio(
    audio_path: str,
    include_timestamps: bool = False
) -> dict[str, Any]:
    """
    Transcribe audio file using Parakeet model.

    Args:
        audio_path: Path to audio file (16kHz, mono, .wav or .flac)
        include_timestamps: Whether to include timestamps in output

    Returns:
        Transcription result dict with:
        - text: Transcribed text
        - timestamps: Timestamp data (if requested)

    Raises:
        RuntimeError: If transcription fails
    """
    model = load_model()

    # Get audio duration for logging
    try:
        duration = get_audio_duration(audio_path)
        logger.info(f"Transcribing audio: {audio_path} ({duration:.2f}s)")
    except Exception:
        logger.info(f"Transcribing audio: {audio_path}")

    # Check if we need to adjust attention model for long audio
    use_long_audio_mode = False
    try:
        duration = get_audio_duration(audio_path)
        if duration > config.MAX_FULL_ATTENTION_DURATION_SEC:
            logger.info(f"Long audio detected ({duration:.2f}s), using local attention")
            use_long_audio_mode = True
            # Adjust attention model for long audio
            model.change_attention_model(
                self_attention_model="rel_pos_local_attn",
                att_context_size=[
                    config.ATTENTION_LEFT_CONTEXT,
                    config.ATTENTION_RIGHT_CONTEXT
                ]
            )
    except Exception as e:
        logger.warning(f"Could not check audio duration: {e}")

    try:
        # Prepare transcription kwargs
        transcribe_kwargs = {
            "audio": [audio_path],
            "timestamps": include_timestamps,
        }

        # Run transcription
        result = model.transcribe(**transcribe_kwargs)
        transcription = result[0]

        # Build response
        response = {
            "text": transcription.text
        }

        # Add timestamps if requested
        if include_timestamps and hasattr(transcription, 'timestamp'):
            response["timestamps"] = transcription.timestamp

        logger.info(f"Transcription complete: {len(transcription.text)} characters")

        return response

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Transcription failed: {e}")


# =============================================================================
# Request Handler
# =============================================================================


def handler(event: dict[str, Any] | str) -> dict[str, Any]:
    """
    RunPod serverless handler function.

    This function is called by RunPod for each request.

    Input JSON format (as sent by RunPod):
    {
        "input": {
            "audio_base64": "<base64 encoded audio>",  // OR
            "audio_url": "<s3 presigned url>",
            "timestamp": false  // optional, default false
        }
    }

    Output JSON format:
    {
        "text": "transcribed text",
        "timestamps": {...}  // only if timestamp=true
    }

    Args:
        event: Request event (dict or JSON string)

    Returns:
        Response dict with transcription or error
    """
    try:
        logger.info("=" * 60)
        logger.info("Handler invoked")

        # Setup Python path to access venv packages FIRST
        # This must happen before any imports from the venv (ffmpeg, boto3, etc.)
        setup_python_path()
        setup_huggingface_cache()

        # Parse input
        if isinstance(event, str):
            try:
                event = json.loads(event)
            except json.JSONDecodeError as e:
                return {
                    "error": f"Invalid JSON input: {e}",
                    "success": False
                }

        # RunPod wraps user input in an "input" field
        # Extract the actual input data
        if isinstance(event, dict) and "input" in event:
            data = event["input"]
            logger.info("Extracted input from RunPod event wrapper")
        else:
            data = event
            logger.info("Using event directly as input")

        logger.debug(f"Input data keys: {list(data.keys()) if isinstance(data, dict) else 'not a dict'}")

        # Validate and get input type
        try:
            input_type, audio_data = validate_audio_input(data)
            include_timestamps = validate_timestamp_flag(data)
            logger.info(f"Input type: {input_type}, timestamps: {include_timestamps}")
        except (ValueError, TypeError) as e:
            logger.error(f"Input validation failed: {e}")
            return {
                "error": f"Invalid input: {e}",
                "success": False
            }

        audio_path = None
        converted_path = None

        try:
            # Get audio file
            if input_type == "base64":
                logger.info("Decoding base64 audio")
                audio_path, _ = decode_base64_audio(audio_data)
            else:  # URL
                logger.info(f"Downloading audio from URL")
                audio_path = download_from_presigned_url(audio_data)

            logger.info(f"Audio file ready: {audio_path}")

            # Convert to Parakeet-compatible format
            logger.info("Converting audio to Parakeet format")
            converted_path = convert_audio_for_parakeet(audio_path)

            # Transcribe
            result = transcribe_audio(converted_path, include_timestamps)
            result["success"] = True

            return result

        except Exception as e:
            logger.error(f"Processing failed: {e}")
            logger.error(traceback.format_exc())
            return {
                "error": str(e),
                "success": False
            }

        finally:
            # Clean up temporary files
            for path in [audio_path, converted_path]:
                if path and os.path.exists(path):
                    try:
                        os.unlink(path)
                        logger.debug(f"Cleaned up temp file: {path}")
                    except Exception as e:
                        logger.warning(f"Failed to clean up {path}: {e}")

    except Exception as e:
        logger.error(f"Unexpected error in handler: {e}")
        logger.error(traceback.format_exc())
        return {
            "error": f"Unexpected error: {e}",
            "success": False
        }


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """
    Main entry point for the container.

    This function:
    1. Runs bootstrap if needed (first run only)
    2. Applies dependency patches if needed
    3. Starts the RunPod serverless worker
    """
    logger.info("=" * 60)
    logger.info("Parakeet RunPod Serverless Starting")
    logger.info("=" * 60)

    # Log configuration
    logger.info(f"Model: {config.MODEL_NAME}")
    logger.info(f"Volume mount: {config.VOLUME_MOUNT}")
    logger.info(f"Parakeet dir: {config.PARAKEET_DIR}")
    logger.info(f"Venv dir: {config.VENV_DIR}")

    # Run bootstrap if needed (first run)
    logger.info("Checking installation status...")
    if not bootstrap_if_needed():
        logger.error("Bootstrap failed - cannot start server")
        sys.exit(1)

    # ONE-TIME PATCH: Install missing dependencies (requests, cuda-python)
    # This can be removed once the patch completes successfully
    patch_script = Path(__file__).parent / "patch_dependencies.py"
    if patch_script.exists():
        logger.info("Running dependency patch...")
        try:
            result = subprocess.run(
                [get_venv_python(), str(patch_script)],
                capture_output=True,
                text=True,
                timeout=600
            )
            if result.returncode == 0:
                logger.info("Patch completed successfully")
                logger.info(result.stdout)
            else:
                logger.warning(f"Patch failed (continuing anyway): {result.stderr}")
        except Exception as e:
            logger.warning(f"Could not run patch (continuing anyway): {e}")

    logger.info("Starting RunPod serverless handler...")

    try:
        import runpod
        runpod.serverless.start({"handler": handler})
    except ImportError:
        logger.error("RunPod SDK not found. This container must be run in RunPod environment.")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
