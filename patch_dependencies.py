#!/usr/bin/env python3
"""
One-time patch to install missing dependencies.

This script installs requests and cuda-python packages that were added
after the initial bootstrap. It can be deleted once the installation completes.
"""

import subprocess
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, Path(__file__).parent.absolute().as_posix())
import config


def get_venv_pip():
    """Get path to pip in the venv."""
    return str(Path(config.VENV_DIR) / "bin" / "pip")


def is_package_installed(package_name: str) -> bool:
    """Check if a package is installed in the venv."""
    try:
        result = subprocess.run(
            [str(Path(config.VENV_DIR) / "bin" / "python"), "-c", f"import {package_name}"],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


def install_missing_packages():
    """Install packages that are missing."""
    print("=" * 60)
    print("PATCH: Checking for missing dependencies...")
    print("=" * 60)

    packages_to_install = []

    # Check requests
    if not is_package_installed("requests"):
        print("❌ requests not found - will install")
        packages_to_install.append("requests>=2.31.0")
    else:
        print("✅ requests already installed")

    # Check cuda-python (import name is 'cuda')
    if not is_package_installed("cuda"):
        print("❌ cuda-python not found - will install")
        packages_to_install.append("cuda-python>=12.3")
    else:
        print("✅ cuda-python already installed")

    if not packages_to_install:
        print("\n✅ All dependencies already installed!")
        return True

    # Install missing packages
    print(f"\n📦 Installing: {', '.join(packages_to_install)}")
    cmd = [get_venv_pip(), "install"] + packages_to_install

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode == 0:
            print("✅ Installation successful!")
            print("\n" + "=" * 60)
            print("PATCH COMPLETE - You can delete patch_dependencies.py")
            print("=" * 60)
            return True
        else:
            print(f"❌ Installation failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ Installation error: {e}")
        return False


if __name__ == "__main__":
    success = install_missing_packages()
    sys.exit(0 if success else 1)
