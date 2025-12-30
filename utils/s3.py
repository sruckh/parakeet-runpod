#!/usr/bin/env python3
"""
S3 download utilities for Parakeet RunPod Serverless.

This module provides functions to download audio files from S3 using either:
1. A pre-signed URL (no credentials required)
2. Direct S3 access using credentials from environment variables
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import config


# =============================================================================
# Pre-signed URL Download
# =============================================================================


def download_from_presigned_url(presigned_url: str) -> str:
    """
    Download audio file from S3 pre-signed URL.

    This function uses the requests library to download a file from a
    pre-signed URL. No S3 credentials are required as the URL contains
    all necessary authentication.

    Args:
        presigned_url: S3 pre-signed URL

    Returns:
        Path to downloaded file

    Raises:
        RuntimeError: If download fails
    """
    try:
        import requests
    except ImportError:
        raise RuntimeError(
            "requests library is not installed. "
            "Please ensure it's available in the virtual environment."
        )

    # Determine file extension from URL or default to .bin
    url_path = presigned_url.split('?')[0]  # Remove query parameters
    suffix = Path(url_path).suffix or ".bin"

    # Create temp file for download
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name

    try:
        print(f"Downloading from pre-signed URL: {presigned_url[:100]}...")

        response = requests.get(
            presigned_url,
            stream=True,
            timeout=config.S3_DOWNLOAD_TIMEOUT
        )
        response.raise_for_status()

        # Get file size for progress tracking
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        # Download with chunking for memory efficiency
        with open(tmp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

                    # Log progress for large files
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        if downloaded % (1024 * 1024) == 0:  # Every MB
                            print(f"Downloaded {downloaded / (1024 * 1024):.1f}MB / {total_size / (1024 * 1024):.1f}MB ({percent:.1f}%)")

        print(f"Download complete: {tmp_path}")
        return tmp_path

    except requests.exceptions.RequestException as e:
        # Clean up failed download
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise RuntimeError(f"Failed to download from pre-signed URL: {e}")

    except Exception as e:
        # Clean up failed download
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise RuntimeError(f"Unexpected error during download: {e}")


# =============================================================================
# Direct S3 Download
# =============================================================================


def download_from_s3_direct(
    bucket: str,
    key: str,
    region: str | None = None,
    endpoint_url: str | None = None
) -> str:
    """
    Download file directly from S3 using credentials.

    S3 credentials are read from environment variables:
    - S3_ACCESS_KEY (or AWS_ACCESS_KEY_ID)
    - S3_SECRET_KEY (or AWS_SECRET_ACCESS_KEY)
    - S3_ENDPOINT (optional, for S3-compatible services)
    - S3_REGION (optional, defaults to us-east-1)

    Args:
        bucket: S3 bucket name
        key: S3 object key (path within bucket)
        region: AWS region (optional, uses S3_REGION env var or default)
        endpoint_url: Custom S3 endpoint URL (optional)

    Returns:
        Path to downloaded file

    Raises:
        RuntimeError: If download fails or credentials are missing
    """
    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError:
        raise RuntimeError(
            "boto3 library is not installed. "
            "Please ensure it's available in the virtual environment."
        )

    # Get credentials from environment
    access_key = os.getenv(config.S3_ACCESS_KEY_ENV) or os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv(config.S3_SECRET_KEY_ENV) or os.getenv("AWS_SECRET_ACCESS_KEY")
    endpoint = endpoint_url or os.getenv(config.S3_ENDPOINT_ENV)
    aws_region = region or os.getenv("S3_REGION", "us-east-1")

    if not access_key:
        raise RuntimeError(
            f"S3 credentials not found. Please set {config.S3_ACCESS_KEY_ENV} "
            f"or AWS_ACCESS_KEY_ID environment variable."
        )

    if not secret_key:
        raise RuntimeError(
            f"S3 credentials not found. Please set {config.S3_SECRET_KEY_ENV} "
            f"or AWS_SECRET_ACCESS_KEY environment variable."
        )

    # Create S3 client
    s3_client = boto3.client(
        's3',
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=aws_region,
        endpoint_url=endpoint,
    )

    # Determine file extension from key
    suffix = Path(key).suffix or ".bin"

    # Create temp file for download
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name

    try:
        print(f"Downloading from S3: s3://{bucket}/{key}")

        s3_client.download_file(bucket, key, tmp_path)

        print(f"Download complete: {tmp_path}")
        return tmp_path

    except ClientError as e:
        # Clean up failed download
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        error_msg = e.response.get('Error', {}).get('Message', str(e))
        raise RuntimeError(f"S3 download failed ({error_code}): {error_msg}")

    except Exception as e:
        # Clean up failed download
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise RuntimeError(f"Unexpected error during S3 download: {e}")


# =============================================================================
# URL Parsing
# =============================================================================


def parse_s3_url(s3_url: str) -> tuple[str, str] | None:
    """
    Parse an S3 URL into bucket and key.

    Supports formats:
    - s3://bucket-name/path/to/file
    - https://s3.amazonaws.com/bucket-name/path/to/file
    - https://bucket-name.s3.amazonaws.com/path/to/file

    Args:
        s3_url: S3 URL to parse

    Returns:
        Tuple of (bucket, key) if valid S3 URL, None otherwise
    """
    # S3 URL format: s3://bucket/key
    if s3_url.startswith("s3://"):
        path = s3_url[5:]  # Remove "s3://"
        parts = path.split("/", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        return parts[0], ""

    # S3 HTTPS format: https://bucket.s3.amazonaws.com/key
    if ".s3.amazonaws.com" in s3_url:
        # Extract bucket from hostname
        from urllib.parse import urlparse
        parsed = urlparse(s3_url)
        bucket = parsed.hostname.replace(".s3.amazonaws.com", "")
        key = parsed.path.lstrip("/")
        return bucket, key

    # Generic S3 endpoint: https://s3.amazonaws.com/bucket/key
    if "s3.amazonaws.com" in s3_url:
        from urllib.parse import urlparse
        parsed = urlparse(s3_url)
        parts = parsed.path.lstrip("/").split("/", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        return parts[0], ""

    return None


def is_s3_url(url: str) -> bool:
    """
    Check if a URL is an S3 URL.

    Args:
        url: URL to check

    Returns:
        True if URL appears to be an S3 URL
    """
    return (
        url.startswith("s3://") or
        "s3.amazonaws.com" in url or
        ".s3." in url  # e.g., s3.us-east-1.amazonaws.com
    )


# =============================================================================
# Download Factory
# =============================================================================


def download_audio_file(url: str) -> str:
    """
    Download an audio file from a URL.

    Automatically detects if the URL is:
    - An S3 URL (parses and downloads directly)
    - A pre-signed URL (downloads with requests)

    Args:
        url: URL to download from

    Returns:
        Path to downloaded file

    Raises:
        RuntimeError: If download fails
    """
    # Check if it's an S3 URL
    s3_parts = parse_s3_url(url)
    if s3_parts:
        bucket, key = s3_parts
        return download_from_s3_direct(bucket, key)

    # Otherwise treat as pre-signed URL
    return download_from_presigned_url(url)
