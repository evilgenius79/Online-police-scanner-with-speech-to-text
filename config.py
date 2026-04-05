"""
Police Scanner Configuration
Edit these values to match your setup.
Run `python list_devices.py` to find your microphone device index.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
CLIPS_DIR = BASE_DIR / "clips"
MODELS_DIR = BASE_DIR / "models"
DB_PATH = BASE_DIR / "scanner.db"

# ─── Audio Capture ────────────────────────────────────────────────────────────
# Sample rate: 16000 Hz is optimal for both webrtcvad and Whisper
SAMPLE_RATE = 16000
CHANNELS = 1

# Frame size for VAD: webrtcvad supports exactly 10, 20, or 30 ms frames.
# 30ms at 16000 Hz = 480 samples = 960 bytes
CHUNK_DURATION_MS = 30
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)  # 480 samples

# Set to the device index of your microphone/scanner input.
# Run `python list_devices.py` to find the correct index.
# None = system default input device.
AUDIO_DEVICE_INDEX = None

# ─── Voice Activity Detection (VAD) ──────────────────────────────────────────
# Aggressiveness: 0 (least aggressive) to 3 (most aggressive at filtering noise).
# For scanner audio with squelch: 2 works well.
VAD_AGGRESSIVENESS = 2

# How many consecutive speech frames required to start recording a clip.
# 5 frames * 30ms = 150ms of speech to trigger. Prevents false positives.
SPEECH_FRAMES_TRIGGER = 5

# How many consecutive silent frames before ending a clip.
# 67 frames * 30ms = ~2.0 seconds of silence = end of transmission.
SILENCE_FRAMES_END = 67

# Minimum clip length in frames (clips shorter than this are discarded).
# 17 frames * 30ms = ~0.5 seconds
MIN_CLIP_FRAMES = 17

# Maximum clip length in frames before force-saving.
# 4000 frames * 30ms = 120 seconds
MAX_CLIP_FRAMES = 4000

# Pre-roll: frames of audio to include BEFORE speech is detected.
# 17 frames * 30ms = ~0.5 seconds before transmission
PRE_ROLL_FRAMES = 17

# Post-roll: frames of silence to include AFTER last speech detected.
# 10 frames * 30ms = ~0.3 seconds after transmission
POST_ROLL_FRAMES = 10

# ─── Whisper Speech-to-Text ───────────────────────────────────────────────────
# Model size: "large-v3" gives best accuracy. Other options:
#   "medium.en"  - faster, good accuracy, English only
#   "small.en"   - even faster, decent accuracy
#   "large-v3"   - best accuracy, requires ~6GB VRAM (fits on RTX 4060)
WHISPER_MODEL = "large-v3"

# Device: "cuda" for GPU (RTX 4060), "cpu" for CPU fallback
WHISPER_DEVICE = "cuda"

# Compute type for CUDA:
#   "float16"  - fast, accurate, recommended for RTX 4060
#   "int8_float16" - slightly faster, minimal quality loss
#   "int8"     - CPU only
WHISPER_COMPUTE_TYPE = "float16"

# Force English transcription (scanner audio is almost always English)
WHISPER_LANGUAGE = "en"

# Beam size: higher = more accurate but slower. 5 is a good balance.
WHISPER_BEAM_SIZE = 5

# Initial prompt: helps Whisper understand the audio context and improves
# accuracy of police radio terminology and phonetic alphabet.
WHISPER_INITIAL_PROMPT = (
    "Police radio scanner transmission. Ten-four. Copy that. "
    "Unit responding. En route. Code three. Dispatch. Over."
)

# ─── Server ───────────────────────────────────────────────────────────────────
HOST = "0.0.0.0"
PORT = 8000
