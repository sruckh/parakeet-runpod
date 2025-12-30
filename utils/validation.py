#!/usr/bin/env python3
"""
Input validation utilities for Parakeet RunPod Serverless.

This module provides functions to validate and parse input data from
RunPod serverless requests.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import tempfile
from typing import Any, Literal, TypedDict

import config


# =============================================================================
# Type Definitions
# =============================================================================

InputType = Literal["base64", "url"]


class ValidationResult(TypedDict):
    """Result of input validation."""
    input_type: InputType
    audio_data: str
    timestamp: bool


class AudioInputData(TypedDict, total=False):
    """Typed dict for audio input data."""
    audio_base64: str
    audio_url: str
    audio_format: str
    timestamp: bool


# =============================================================================
# Audio Input Validation
# =============================================================================


def validate_audio_input(data: dict[str, Any]) -> tuple[InputType, str]:
    """
    Validate and determine the type of audio input.

    The input should contain either:
    - "audio_base64": Base64 encoded audio data
    - "audio_url": URL to audio file (S3 pre-signed URL or similar)

    Args:
        data: Input data dictionary from the request

    Returns:
        Tuple of (input_type, audio_data) where:
        - input_type: "base64" or "url"
        - audio_data: The base64 string or URL

    Raises:
        ValueError: If input is invalid or missing required fields
        TypeError: If input is not a dictionary
    """
    if not isinstance(data, dict):
        raise TypeError(
            f"Input must be a dictionary, got {type(data).__name__}"
        )

    # Check for base64 audio
    if "audio_base64" in data:
        audio_base64 = data["audio_base64"]

        if not isinstance(audio_base64, str):
            raise TypeError(
                f"audio_base64 must be a string, got {type(audio_base64).__name__}"
            )

        if not audio_base64:
            raise ValueError("audio_base64 cannot be empty")

        # Optionally validate format from mime type if provided
        if "audio_format" in data:
            audio_format = data["audio_format"]
            if not isinstance(audio_format, str):
                raise TypeError("audio_format must be a string")

            # Ensure format starts with a dot
            fmt = audio_format if audio_format.startswith(".") else f".{audio_format}"

            if fmt.lower() not in config.SUPPORTED_INPUT_FORMATS:
                raise ValueError(
                    f"Unsupported audio format: {fmt}. "
                    f"Supported formats: {config.SUPPORTED_INPUT_FORMATS}"
                )

        return "base64", audio_base64

    # Check for URL input
    elif "audio_url" in data:
        audio_url = data["audio_url"]

        if not isinstance(audio_url, str):
            raise TypeError(
                f"audio_url must be a string, got {type(audio_url).__name__}"
            )

        if not audio_url:
            raise ValueError("audio_url cannot be empty")

        # Basic URL validation
        if not (audio_url.startswith("http://") or
                audio_url.startswith("https://") or
                audio_url.startswith("s3://")):
            raise ValueError(
                "audio_url must be a valid URL starting with http://, https://, or s3://"
            )

        return "url", audio_url

    else:
        raise ValueError(
            "Input must contain either 'audio_base64' or 'audio_url' field"
        )


# =============================================================================
# Base64 Audio Decoding
# =============================================================================


def decode_base64_audio(
    audio_base64: str,
    output_format: str = "wav"
) -> tuple[str, str | None]:
    """
    Decode base64 audio data to a temporary file.

    Handles both raw base64 data and data URI format:
    - Raw: "base64encodeddata..."
    - Data URI: "data:audio/mp3;base64,base64encodeddata..."

    Args:
        audio_base64: Base64 encoded audio data
        output_format: Desired output format (default: "wav")

    Returns:
        Tuple of (file_path, original_format) where:
        - file_path: Path to decoded audio file
        - original_format: Detected original format (with dot), or None

    Raises:
        ValueError: If base64 data is invalid
        RuntimeError: If file writing fails
    """
    original_format = None
    data_to_decode = audio_base64

    # Check for data URI format
    if "," in audio_base64:
        header, data = audio_base64.split(",", 1)
        data_to_decode = data

        # Parse MIME type from header
        if "audio/" in header:
            # Extract MIME type
            mime_part = header.split(";")[0]
            mime_type = mime_part.split(":")[1] if ":" in mime_part else ""

            # Convert MIME type to file extension
            if mime_type:
                extensions = mimetypes.guess_all_extensions(mime_type)
                if extensions:
                    original_format = extensions[0]  # Use first (most common) extension

    # Decode base64
    try:
        audio_bytes = base64.b64decode(data_to_decode)
    except Exception as e:
        raise ValueError(f"Invalid base64 data: {e}")

    if len(audio_bytes) == 0:
        raise ValueError("Decoded audio data is empty")

    # Determine file suffix
    suffix = original_format or f".{output_format}"

    # Create temp file and write decoded data
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
    except Exception as e:
        raise RuntimeError(f"Failed to write decoded audio to file: {e}")

    return tmp_path, original_format


def encode_audio_to_base64(file_path: str) -> str:
    """
    Encode an audio file to base64 string.

    Args:
        file_path: Path to audio file

    Returns:
        Base64 encoded string

    Raises:
        FileNotFoundError: If file doesn't exist
        RuntimeError: If encoding fails
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    try:
        with open(file_path, "rb") as f:
            audio_bytes = f.read()
        return base64.b64encode(audio_bytes).decode('utf-8')
    except Exception as e:
        raise RuntimeError(f"Failed to encode audio to base64: {e}")


# =============================================================================
# Timestamp Flag Validation
# =============================================================================


def validate_timestamp_flag(data: dict[str, Any]) -> bool:
    """
    Extract and validate the timestamp flag.

    Args:
        data: Input data dictionary

    Returns:
        Boolean value for timestamp flag (default: False)

    Raises:
        TypeError: If timestamp value is not a boolean or boolean-like string
    """
    if "timestamp" not in data:
        return config.DEFAULT_INCLUDE_TIMESTAMPS

    timestamp_value = data["timestamp"]

    # Handle boolean
    if isinstance(timestamp_value, bool):
        return timestamp_value

    # Handle integer (0 or 1)
    if isinstance(timestamp_value, int):
        if timestamp_value in (0, 1):
            return bool(timestamp_value)
        raise TypeError(
            f"timestamp must be 0 or 1 when using integer, got {timestamp_value}"
        )

    # Handle string
    if isinstance(timestamp_value, str):
        lower_val = timestamp_value.lower()
        if lower_val in ("true", "1", "yes", "on"):
            return True
        if lower_val in ("false", "0", "no", "off", ""):
            return False
        raise TypeError(
            f"timestamp string must be 'true', 'false', '1', or '0', got '{timestamp_value}'"
        )

    raise TypeError(
        f"timestamp must be a boolean, got {type(timestamp_value).__name__}"
    )


# =============================================================================
# Request Validation
# =============================================================================


def validate_request(data: dict[str, Any] | str) -> ValidationResult:
    """
    Validate a complete request and extract all parameters.

    Args:
        data: Request data (dict or JSON string)

    Returns:
        ValidationResult with all extracted parameters

    Raises:
        TypeError: If data structure is invalid
        ValueError: If data content is invalid
    """
    # Parse JSON string if needed
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in request: {e}")

    # Validate audio input
    input_type, audio_data = validate_audio_input(data)

    # Validate timestamp flag
    timestamp = validate_timestamp_flag(data)

    return {
        "input_type": input_type,
        "audio_data": audio_data,
        "timestamp": timestamp,
    }


# =============================================================================
# Sanitization
# =============================================================================


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename by removing dangerous characters.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename safe for filesystem use
    """
    # Keep only alphanumeric, dash, underscore, and dot
    import re
    sanitized = re.sub(r'[^\w\-.]', '_', filename)

    # Remove leading dots/dashes (hidden files)
    sanitized = sanitized.lstrip('.').lstrip('-')

    # Limit length
    if len(sanitized) > 255:
        name, ext = os.path.splitext(sanitized)
        sanitized = name[:255 - len(ext)] + ext

    return sanitized or "audio"


def get_file_extension_from_format(audio_format: str) -> str:
    """
    Convert a format specifier to a file extension.

    Args:
        audio_format: Format string (with or without leading dot)

    Returns:
        File extension with leading dot
    """
    if not audio_format:
        return ".wav"

    # Already has dot
    if audio_format.startswith("."):
        return audio_format.lower()

    # Add dot
    return f".{audio_format.lower()}"
