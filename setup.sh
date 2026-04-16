#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  Police Scanner – setup script
#  Tested on Ubuntu 22.04/24.04 and Windows (Git Bash) with Python 3.10/3.11/3.12
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# On Windows 'python3' is often not in PATH — try it first, fall back to 'python'.
if [ -z "${PYTHON:-}" ]; then
  if command -v python3 &>/dev/null; then
    PYTHON=python3
  elif command -v python &>/dev/null; then
    PYTHON=python
  fi
fi

echo ""
echo "  Police Scanner – setup"
echo ""

# ── 1.  Python version check ──────────────────────────────────────────────────
PY_VER=$($PYTHON -c "import sys; print(sys.version_info[:2])" 2>/dev/null || echo "")
if [ -z "$PY_VER" ]; then
  echo "ERROR: Python not found.  Install Python 3.10 or later and add it to PATH."
  exit 1
fi
echo "  Python: $($PYTHON --version)"

# ── 2.  Virtual environment ───────────────────────────────────────────────────
if [ ! -d "venv" ]; then
  echo "  Creating virtual environment in ./venv ..."
  $PYTHON -m venv venv
fi

# Windows (Git Bash / MSYS2) puts the activate script under Scripts/.
# Linux / macOS / WSL use bin/.
if [ -f "venv/Scripts/activate" ]; then
  ACTIVATE="venv/Scripts/activate"
  VENV_PYTHON="venv/Scripts/python"
else
  ACTIVATE="venv/bin/activate"
  VENV_PYTHON="venv/bin/python"
fi

source "$ACTIVATE"
echo "  Activated: $(which python)"

# Use 'python -m pip' throughout — works correctly on both Windows and Linux,
# and avoids the Windows restriction on upgrading pip via the pip executable itself.
PIP="python -m pip"

$PIP install --upgrade pip --quiet

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
#  On Windows:
#    PortAudio is bundled inside the sounddevice pip wheel — no extra install needed.

# ── 4.  Python packages ────────────────────────────────────────────────────────
echo "  Installing Python packages..."

# First try webrtcvad-wheels (pre-built wheel, works on Python 3.10-3.12).
# Fall back to the original webrtcvad if wheels are not available for this platform.
$PIP install webrtcvad-wheels 2>/dev/null \
  || $PIP install webrtcvad \
  || echo "  WARNING: webrtcvad could not be installed; energy-only VAD will be used."

$PIP install -r requirements.txt

# ── 5.  CUDA / cuDNN check + install ─────────────────────────────────────────
echo ""
echo "  Checking GPU availability..."
if command -v nvidia-smi &>/dev/null; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  echo ""
  echo "  GPU found.  Installing CUDA 12 runtime libraries via pip..."
  echo "  (This installs the correct .so files without requiring a system CUDA install.)"
  echo "  Your NVIDIA driver must be >= 525.  Run 'nvidia-smi' to verify."
  echo ""
  $PIP install "nvidia-cublas-cu12>=12.3.0" "nvidia-cudnn-cu12>=9.0.0,<10"
  echo ""
  echo "  CUDA libraries installed.  faster-whisper will try large-v3 (float16) first."
  echo "  If VRAM is tight it will automatically fall back to a smaller model."
  echo "  To change the starting model, edit WHISPER_MODEL in config.py"
else
  echo "  No NVIDIA GPU detected.  Transcription will run on CPU."
  echo "  Edit config.py and set:"
  echo "    WHISPER_DEVICE = 'cpu'"
  echo "    WHISPER_COMPUTE_TYPE = 'int8'"
  echo "    WHISPER_MODEL = 'small.en'  (faster on CPU)"
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
echo "  3.  Open config.py in any text editor and set:"
echo "        AUDIO_DEVICE_INDEX = <that index>"
echo "  4.  (Optional) adjust VAD_AGGRESSIVENESS if clips are not being captured."
echo "  5.  Start the server:"
echo "        Re-activate the venv in your terminal, then run python run.py"
echo ""
if [ -f "venv/Scripts/activate" ]; then
  echo "        Git Bash / MSYS2:  source venv/Scripts/activate"
  echo "        PowerShell:        venv\\Scripts\\Activate.ps1"
  echo "        Command Prompt:    venv\\Scripts\\activate.bat"
else
  echo "        Linux / macOS:     source venv/bin/activate"
fi
echo ""
echo "        python run.py"
echo "  6.  Open:  http://localhost:8000"
echo ""
