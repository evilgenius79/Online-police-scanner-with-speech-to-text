#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  Police Scanner – setup script
#  Tested on Ubuntu 22.04/24.04 with Python 3.10/3.11/3.12
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PYTHON=${PYTHON:-python3}

echo ""
echo "  Police Scanner – setup"
echo ""

# ── 1.  Python version check ──────────────────────────────────────────────────
PY_VER=$($PYTHON -c "import sys; print(sys.version_info[:2])" 2>/dev/null || echo "")
if [ -z "$PY_VER" ]; then
  echo "ERROR: python3 not found.  Install Python 3.10 or later."
  exit 1
fi
echo "  Python: $($PYTHON --version)"

# ── 2.  Virtual environment ───────────────────────────────────────────────────
if [ ! -d "venv" ]; then
  echo "  Creating virtual environment in ./venv ..."
  $PYTHON -m venv venv
fi
source venv/bin/activate
echo "  Activated: $(which python)"

pip install --upgrade pip --quiet

# ── 3.  System audio library (PortAudio, required by sounddevice) ─────────────
#  On Ubuntu/Debian:
if command -v apt-get &>/dev/null; then
  echo "  Installing PortAudio (system dependency for sounddevice)..."
  sudo apt-get install -y --no-install-recommends \
    libportaudio2 portaudio19-dev \
    2>/dev/null || echo "  WARNING: could not install PortAudio via apt; install it manually."
fi
#  On Fedora/RHEL:
#    sudo dnf install portaudio portaudio-devel

# ── 4.  Python packages ────────────────────────────────────────────────────────
echo "  Installing Python packages..."

# First try webrtcvad-wheels (pre-built wheel, works on Python 3.10-3.12).
# Fall back to the original webrtcvad if wheels are not available for this platform.
pip install webrtcvad-wheels 2>/dev/null \
  || pip install webrtcvad \
  || echo "  WARNING: webrtcvad could not be installed; energy-only VAD will be used."

pip install -r requirements.txt

# ── 5.  CUDA / cuDNN check ────────────────────────────────────────────────────
echo ""
echo "  Checking GPU availability..."
if command -v nvidia-smi &>/dev/null; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  echo ""
  echo "  GPU found.  faster-whisper will use CUDA (float16)."
  echo "  Ensure CUDA 11.x/12.x and cuDNN 8.x/9.x are installed."
  echo "  Install with:"
  echo "    pip install nvidia-cublas-cu12 nvidia-cudnn-cu12"
  echo "  OR follow: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/"
else
  echo "  No NVIDIA GPU detected.  Transcription will run on CPU."
  echo "  Edit config.py and set:"
  echo "    WHISPER_DEVICE = 'cpu'"
  echo "    WHISPER_COMPUTE_TYPE = 'int8'"
fi

# ── 6.  Create required directories ──────────────────────────────────────────
mkdir -p clips models
echo ""
echo "  Created: clips/  models/"

# ── 7.  Done ──────────────────────────────────────────────────────────────────
echo ""
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo "  1.  Plug your Uniden scanner into the mic/line-in jack."
echo "  2.  Run:  python list_devices.py"
echo "        → Find your scanner's input device index."
echo "  3.  Edit config.py  → set AUDIO_DEVICE_INDEX = <that index>"
echo "  4.  (Optional) adjust VAD_AGGRESSIVENESS if clips are not being captured."
echo "  5.  Start the server:"
echo "        source venv/bin/activate"
echo "        python run.py"
echo "  6.  Open:  http://localhost:8000"
echo ""
