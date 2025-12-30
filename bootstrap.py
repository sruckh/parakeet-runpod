#!/usr/bin/env python3
"""
Bootstrap module for Parakeet RunPod Serverless.

This module handles the first-time installation and verification of the software
stack on the network-attached volume. Subsequent runs will skip installation
if the marker file exists.

Installation includes:
- Python virtual environment creation
- PyTorch with CUDA 12.8 support
- Flash Attention
- NVIDIA NeMo toolkit with ASR support
- Hugging Face Hub CLI
- Additional dependencies (ffmpeg-python, boto3)
- Parakeet model download

The bootstrap process only runs once when the volume is first attached.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config

# =============================================================================
# Logging
# =============================================================================


def log_info(message: str) -> None:
    """Log an info message."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [BOOTSTRAP] {message}", flush=True)


def log_error(message: str) -> None:
    """Log an error message."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [BOOTSTRAP] ERROR: {message}", flush=True, file=sys.stderr)


def log_success(message: str) -> None:
    """Log a success message."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [BOOTSTRAP] SUCCESS: {message}", flush=True)


# =============================================================================
# Command Execution
# =============================================================================


def run_command(
    cmd: list[str],
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    description: str = ""
) -> bool:
    """
    Run a shell command and return success status.

    Args:
        cmd: Command to run as a list of strings
        cwd: Working directory for the command
        env: Environment variables for the command
        description: Description of what the command does

    Returns:
        True if command succeeded, False otherwise
    """
    if description:
        log_info(f"Running: {description}")

    cmd_str = " ".join(cmd)
    log_info(f"Executing: {cmd_str}")

    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            env=merged_env,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minute timeout per command
        )

        if result.returncode != 0:
            log_error(f"Command failed with exit code {result.returncode}")
            if result.stderr:
                log_error(f"STDERR: {result.stderr}")
            if result.stdout:
                log_info(f"STDOUT: {result.stdout}")
            return False

        if result.stdout:
            # Log stdout but limit verbosity
            lines = result.stdout.strip().split("\n")
            if len(lines) > 20:
                log_info(f"STDOUT (showing first/last 10 lines of {len(lines)}):")
                for line in lines[:5]:
                    print(f"  {line}")
                print("  ...")
                for line in lines[-5:]:
                    print(f"  {line}")
            else:
                log_info(f"STDOUT: {result.stdout}")

        return True

    except subprocess.TimeoutExpired:
        log_error(f"Command timed out after 30 minutes")
        return False
    except Exception as e:
        log_error(f"Command execution failed: {e}")
        return False


# =============================================================================
# Installation State Management
# =============================================================================


def is_installed() -> bool:
    """
    Check if installation has already been completed.

    Verifies:
    - Installation marker file exists
    - Virtual environment directory exists
    - Python executable in venv exists

    Returns:
        True if installation is complete, False otherwise
    """
    marker = Path(config.INSTALLATION_MARKER)
    venv_python = Path(config.VENV_DIR) / "bin" / "python"
    venv_pip = Path(config.VENV_DIR) / "bin" / "pip"

    if not marker.exists():
        log_info("Installation marker not found - first run detected")
        return False

    if not venv_python.exists():
        log_info("Virtual environment Python not found")
        return False

    if not venv_pip.exists():
        log_info("Virtual environment pip not found")
        return False

    # Verify marker contents
    try:
        content = marker.read_text()
        data = json.loads(content)
        log_info(f"Installation marker found: {data.get('status', 'unknown')}")
        log_info(f"Installation completed at: {data.get('timestamp', 'unknown')}")
        return True
    except json.JSONDecodeError:
        # Old format marker, treat as valid
        log_info("Installation marker found (legacy format)")
        return True
    except Exception as e:
        log_info(f"Could not read marker file: {e}")
        return False


def mark_installation_complete() -> bool:
    """
    Create marker file to indicate successful installation.

    Returns:
        True if marker was created successfully
    """
    marker = Path(config.INSTALLATION_MARKER)
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "status": "complete",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": config.MODEL_NAME,
            "pytorch_version": "2.8.0",
            "flash_attn_version": config.FLASH_ATTN_VERSION,
        }
        marker.write_text(json.dumps(data, indent=2))
        log_success(f"Created installation marker: {marker}")
        return True
    except Exception as e:
        log_error(f"Failed to create installation marker: {e}")
        return False


# =============================================================================
# Installation Steps
# =============================================================================


def create_directories() -> bool:
    """Create necessary directories on the volume."""
    log_info("Creating directory structure...")

    dirs = [
        config.PARAKEET_DIR,
        config.HF_HOME_DIR,
        config.HF_HUB_CACHE,
    ]

    try:
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)
        log_success(f"Created directories: {dirs}")
        return True
    except Exception as e:
        log_error(f"Failed to create directories: {e}")
        return False


def create_virtual_environment() -> bool:
    """Create Python virtual environment."""
    log_info("Creating Python 3.12 virtual environment...")

    cmd = [sys.executable, "-m", "venv", config.VENV_DIR]
    return run_command(cmd, description="Create virtual environment")


def get_venv_python() -> str:
    """Get path to Python executable in venv."""
    return str(Path(config.VENV_DIR) / "bin" / "python")


def get_venv_pip() -> str:
    """Get path to pip executable in venv."""
    return str(Path(config.VENV_DIR) / "bin" / "pip")


def get_venv_site_packages() -> str:
    """Get path to site-packages directory in venv."""
    return str(Path(config.VENV_DIR) / "lib" / "python3.12" / "site-packages")


def get_venvHF() -> str:
    """Get path to HF CLI executable in venv."""
    return str(Path(config.VENV_DIR) / "bin" / "hf")


def install_pytorch() -> bool:
    """Install PyTorch with CUDA 12.8 support."""
    log_info("Installing PyTorch 2.8.0 with CUDA 12.8 support...")

    packages = config.PYTORCH_VERSION.split()
    cmd = [
        get_venv_pip(), "install",
        *packages,
        "--index-url", config.PYTORCH_INDEX_URL
    ]

    return run_command(
        cmd,
        description=f"Install PyTorch with CUDA 12.8",
        env={"PIP_NO_CACHE_DIR": "1"}
    )


def install_flash_attn() -> bool:
    """Install flash-attention from wheel."""
    log_info("Installing flash-attention 2.8.1...")

    cmd = [get_venv_pip(), "install", config.FLASH_ATTN_WHEEL]

    return run_command(
        cmd,
        description=f"Install flash-attn from wheel",
        env={"PIP_NO_CACHE_DIR": "1"}
    )


def install_nemo() -> bool:
    """Install NVIDIA NeMo toolkit with ASR support."""
    log_info("Installing NVIDIA NeMo toolkit with ASR support...")

    cmd = [get_venv_pip(), "install", "nemo-toolkit[asr]"]

    return run_command(
        cmd,
        description="Install NeMo toolkit",
        env={"PIP_NO_CACHE_DIR": "1"}
    )


def install_huggingface_hub() -> bool:
    """Install Hugging Face Hub CLI."""
    log_info("Installing Hugging Face Hub CLI...")

    cmd = [get_venv_pip(), "install", "huggingface_hub>=0.25.0"]

    return run_command(
        cmd,
        description="Install Hugging Face Hub",
        env={"PIP_NO_CACHE_DIR": "1"}
    )


def install_other_dependencies() -> bool:
    """Install additional dependencies."""
    log_info("Installing additional dependencies...")

    dependencies = [
        "ffmpeg-python>=0.2.0",
        "boto3>=1.34.0",
        "requests>=2.31.0",  # Required for S3 pre-signed URL downloads
    ]

    cmd = [get_venv_pip(), "install", *dependencies]

    return run_command(
        cmd,
        description="Install ffmpeg-python, boto3, and requests",
        env={"PIP_NO_CACHE_DIR": "1"}
    )


def download_model() -> bool:
    """
    Download Parakeet model using the Python API.
    The model will be cached to the configured HF_HUB_CACHE location.
    """
    log_info(f"Downloading Parakeet model: {config.MODEL_NAME}")

    # Set environment variables for download location
    env = os.environ.copy()
    env["HF_HOME"] = config.HF_HOME_DIR
    env["HF_HUB_CACHE"] = config.HF_HUB_CACHE

    # Use Python to download via huggingface_hub
    # This is more reliable than the CLI for large models
    download_script = f'''
import os
os.environ["HF_HOME"] = "{config.HF_HOME_DIR}"
os.environ["HF_HUB_CACHE"] = "{config.HF_HUB_CACHE}"

from huggingface_hub import snapshot_download

print("Downloading model: {config.MODEL_NAME}")
snapshot_download(
    repo_id="{config.MODEL_NAME}",
    local_dir="{config.HF_HUB_CACHE}/{config.MODEL_NAME}",
    local_dir_use_symlinks=False,
)
print("Model download complete")
'''

    # Write script to temp file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(download_script)
        script_path = f.name

    try:
        cmd = [get_venv_python(), script_path]
        success = run_command(
            cmd,
            description=f"Download {config.MODEL_NAME}",
            env=env
        )
        return success
    finally:
        os.unlink(script_path)


# =============================================================================
# Main Installation Flow
# =============================================================================


def install_all() -> bool:
    """
    Run the complete installation process.

    Returns:
        True if all steps succeeded, False otherwise
    """
    log_info("=" * 60)
    log_info("Starting Parakeet Bootstrap Installation")
    log_info("=" * 60)

    steps: list[tuple[str, Callable[[], bool]]] = [
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

    failed_step = None

    for step_name, step_func in steps:
        log_info("-" * 40)
        log_info(f"STEP: {step_name}")
        if not step_func():
            log_error(f"FAILED: {step_name}")
            failed_step = step_name
            break
        log_success(f"Completed: {step_name}")

    if failed_step:
        log_error("=" * 60)
        log_error(f"Bootstrap installation FAILED at: {failed_step}")
        log_error("=" * 60)
        return False

    log_info("=" * 60)
    log_success("Bootstrap Installation Complete!")
    log_info("=" * 60)

    # Print summary
    log_info(f"Virtual environment: {config.VENV_DIR}")
    log_info(f"HuggingFace cache: {config.HF_HUB_CACHE}")
    log_info(f"Model: {config.MODEL_NAME}")

    return True


def bootstrap_if_needed() -> bool:
    """
    Check if installation is needed and run if necessary.

    This is the main entry point called from handler.py.

    Returns:
        True if ready for inference, False if bootstrap failed
    """
    log_info("Checking installation status...")

    if is_installed():
        log_info("Software already installed. Skipping bootstrap.")
        return True

    log_info("First run detected. Starting bootstrap installation...")
    return install_all()


# =============================================================================
# CLI Entry Point
# =============================================================================


def main() -> int:
    """
    CLI entry point for running bootstrap manually.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    import argparse

    parser = argparse.ArgumentParser(description="Parakeet Bootstrap Installer")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-installation even if marker exists"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check installation status, don't install"
    )

    args = parser.parse_args()

    if args.check:
        if is_installed():
            print("Installation is complete")
            return 0
        else:
            print("Installation is incomplete or not found")
            return 1

    if args.force:
        # Remove marker and force reinstall
        marker = Path(config.INSTALLATION_MARKER)
        if marker.exists():
            marker.unlink()
            log_info("Installation marker removed - forcing reinstall")
        return 0 if install_all() else 1

    return 0 if bootstrap_if_needed() else 1


if __name__ == "__main__":
    sys.exit(main())
