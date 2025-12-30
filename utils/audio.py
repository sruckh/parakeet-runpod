#!/usr/bin/env python3
"""
Audio conversion utilities for Parakeet RunPod Serverless.

This module provides functions to convert audio files to the format required
by the Parakeet model:
- Sample rate: 16kHz
- Channels: 1 (mono)
- Format: .wav or .flac
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Literal

import config

# Type aliases
AudioFormat = Literal["wav", "flac"]


# =============================================================================
# Audio Conversion
# =============================================================================


def convert_audio_for_parakeet(
    input_path: str,
    output_format: AudioFormat = "wav"
) -> str:
    """
    Convert audio file to Parakeet-compatible format.

    This function uses ffmpeg to convert the input audio to the format
    required by the Parakeet model (16kHz, mono, .wav or .flac).

    Args:
        input_path: Path to input audio file
        output_format: Output format ("wav" or "flac")

    Returns:
        Path to converted audio file

    Raises:
        FileNotFoundError: If input file doesn't exist
        RuntimeError: If ffmpeg conversion fails
    """
    # Verify input file exists
    input_path_obj = Path(input_path)
    if not input_path_obj.exists():
        raise FileNotFoundError(f"Input audio file not found: {input_path}")

    # Import ffmpeg here so it can be mocked for testing
    try:
        import ffmpeg
    except ImportError:
        raise RuntimeError(
            "ffmpeg-python is not installed. "
            "Please ensure it's installed in the virtual environment."
        )

    # Determine output codec based on format
    if output_format == "wav":
        codec = "pcm_s16le"
    elif output_format == "flac":
        codec = "flac"
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

    # Create temp file for output
    suffix = f".{output_format}"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        output_path = tmp.name

    try:
        # Build ffmpeg conversion pipeline
        input_stream = ffmpeg.input(input_path)

        # Convert to required format
        output_stream = ffmpeg.output(
            input_stream,
            output_path,
            acodec=codec,
            ac=str(config.PARAKEET_CHANNELS),
            ar=str(config.PARAKEET_SAMPLE_RATE)
        )

        # Run conversion
        ffmpeg.run(output_stream, overwrite_output=True, quiet=True, capture_stdout=True, capture_stderr=True)

        return output_path

    except ffmpeg.Error as e:
        # Clean up failed output file
        if os.path.exists(output_path):
            os.unlink(output_path)

        error_msg = e.stderr.decode('utf8') if e.stderr else "Unknown error"
        raise RuntimeError(f"FFmpeg conversion failed: {error_msg}")

    except Exception as e:
        # Clean up failed output file
        if os.path.exists(output_path):
            os.unlink(output_path)
        raise RuntimeError(f"Audio conversion failed: {e}")


# =============================================================================
# Audio Metadata
# =============================================================================


def get_audio_duration(file_path: str) -> float:
    """
    Get audio file duration in seconds using ffprobe.

    Args:
        file_path: Path to audio file

    Returns:
        Duration in seconds

    Raises:
        RuntimeError: If ffprobe fails
    """
    try:
        import ffmpeg
    except ImportError:
        raise RuntimeError("ffmpeg-python is not installed")

    if not Path(file_path).exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    try:
        probe = ffmpeg.probe(file_path)
        duration_str = probe.get('format', {}).get('duration')

        if duration_str is None:
            raise RuntimeError(f"Could not get duration for {file_path}")

        return float(duration_str)

    except ffmpeg.Error as e:
        error_msg = e.stderr.decode('utf8') if e.stderr else "Unknown error"
        raise RuntimeError(f"Failed to get audio duration: {error_msg}")
    except (ValueError, KeyError) as e:
        raise RuntimeError(f"Failed to parse duration: {e}")


def get_audio_info(file_path: str) -> dict:
    """
    Get detailed audio file information using ffprobe.

    Args:
        file_path: Path to audio file

    Returns:
        Dictionary containing:
        - duration: Duration in seconds
        - sample_rate: Sample rate in Hz
        - channels: Number of audio channels
        - codec: Audio codec name
        - format: Container format

    Raises:
        RuntimeError: If ffprobe fails
    """
    try:
        import ffmpeg
    except ImportError:
        raise RuntimeError("ffmpeg-python is not installed")

    if not Path(file_path).exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    try:
        probe = ffmpeg.probe(file_path)

        format_info = probe.get('format', {})
        audio_stream = None

        for stream in probe.get('streams', []):
            if stream.get('codec_type') == 'audio':
                audio_stream = stream
                break

        return {
            'duration': float(format_info.get('duration', 0)),
            'sample_rate': int(audio_stream.get('sample_rate', 0)) if audio_stream else 0,
            'channels': int(audio_stream.get('channels', 0)) if audio_stream else 0,
            'codec': audio_stream.get('codec_name', 'unknown') if audio_stream else 'unknown',
            'format': format_info.get('format_name', 'unknown'),
        }

    except ffmpeg.Error as e:
        error_msg = e.stderr.decode('utf8') if e.stderr else "Unknown error"
        raise RuntimeError(f"Failed to get audio info: {error_msg}")


# =============================================================================
# Audio Validation
# =============================================================================


def is_valid_audio_file(file_path: str) -> bool:
    """
    Check if file is a valid audio file.

    Args:
        file_path: Path to file to check

    Returns:
        True if file exists and has a supported audio extension
    """
    path = Path(file_path)

    if not path.exists():
        return False

    if not path.is_file():
        return False

    # Check extension
    valid_extensions = {ext.lower() for ext in config.SUPPORTED_INPUT_FORMATS}
    return path.suffix.lower() in valid_extensions


def is_parakeet_compatible(file_path: str) -> tuple[bool, str]:
    """
    Check if audio file is already compatible with Parakeet requirements.

    Args:
        file_path: Path to audio file

    Returns:
        Tuple of (is_compatible, reason)
        - is_compatible: True if file meets all requirements
        - reason: String explaining why not compatible (or "OK" if compatible)
    """
    if not is_valid_audio_file(file_path):
        return False, "Invalid or unsupported audio file"

    try:
        info = get_audio_info(file_path)

        # Check sample rate
        if info['sample_rate'] != config.PARAKEET_SAMPLE_RATE:
            return False, f"Sample rate is {info['sample_rate']}Hz, requires {config.PARAKEET_SAMPLE_RATE}Hz"

        # Check channels
        if info['channels'] != config.PARAKEET_CHANNELS:
            return False, f"Audio has {info['channels']} channel(s), requires {config.PARAKEET_CHANNELS}"

        # Check format
        path = Path(file_path)
        if path.suffix.lower() not in config.PARAKEET_FORMATS:
            return False, f"Format is {path.suffix}, requires {config.PARAKEET_FORMATS}"

        return True, "OK"

    except Exception as e:
        return False, f"Error checking file: {e}"


# =============================================================================
# Temporary File Management
# =============================================================================


class TempAudioFile:
    """
    Context manager for temporary audio file handling.

    Automatically cleans up the temporary file when done.
    """

    def __init__(self, suffix: str = ".wav"):
        """
        Initialize temp audio file context manager.

        Args:
            suffix: File suffix/extension
        """
        self.suffix = suffix
        self.path = None

    def __enter__(self) -> str:
        """Create and return path to temp file."""
        tmp = tempfile.NamedTemporaryFile(suffix=self.suffix, delete=False)
        tmp.close()
        self.path = tmp.name
        return self.path

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up temp file."""
        if self.path and os.path.exists(self.path):
            try:
                os.unlink(self.path)
            except Exception:
                pass
        return False
