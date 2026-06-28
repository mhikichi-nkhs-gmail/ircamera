#!/usr/bin/env bash
set -euo pipefail

# Build environment setup script for the IR camera + TouchDesigner OSC sender.
# Run on a fresh Raspberry Pi OS (Bookworm) install with internet access.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------
echo "Installing system packages ..."
sudo apt-get update
sudo apt-get install -y \
    python3-venv \
    python3-pip \
    python3-picamera2 \
    python3-opencv \
    python3-numpy \
    python3-libcamera \
    libcap-dev

# ---------------------------------------------------------------------------
# 2. Python virtual environment
# ---------------------------------------------------------------------------
echo "Creating virtual environment at ${VENV_DIR} ..."
python3 -m venv "${VENV_DIR}" --system-site-packages

# ---------------------------------------------------------------------------
# 3. Python packages not available via apt
# ---------------------------------------------------------------------------
echo "Installing python-osc ..."
"${VENV_DIR}/bin/pip" install --upgrade pip
"${VENV_DIR}/bin/pip" install python-osc

# ---------------------------------------------------------------------------
# 4. Done
# ---------------------------------------------------------------------------
echo ""
echo "Setup complete."
echo "Activate with:  source ${VENV_DIR}/bin/activate"
echo "Run with:       python camera_server.py"
echo ""
echo "Optional environment variables:"
echo "  OSC_IP              TouchDesigner IP  (default: 192.168.1.100)"
echo "  OSC_PORT            TouchDesigner port (default: 9000)"
echo "  PROJECTION_ASPECT   Projection width/height (default: 1.7778)"
