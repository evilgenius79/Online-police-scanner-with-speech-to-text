"""
GPU-accelerated speech-to-text using faster-whisper.

faster-whisper wraps OpenAI Whisper with CTranslate2, giving 2-4x speed-up
over the original implementation and much lower memory usage via float16 or
int8 quantisation on an RTX 4060.

Model download:  ~3 GB for large-v3, stored in the models/ directory.
                 Downloaded automatically on first run.

The worker runs in its own daemon thread, serialising all inference so the
GPU is never called concurrently.  The transcription_queue is a
threading.Queue filled by the audio capture worker.
"""
import logging
import queue
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Module state ──────────────────────────────────────────────────────────────
_running = False
_thread: Optional[threading.Thread] = None
_transcription_queue: Optional[queue.Queue] = None
_broadcaster = None
_model = None          # faster_whisper.WhisperModel instance


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def configure(broadcaster) -> None:
    """Wire up the broadcaster before calling start()."""
    global _broadcaster
    _broadcaster = broadcaster


def start(transcription_queue: queue.Queue) -> None:
    """Spawn the transcription worker thread."""
    global _running, _transcription_queue, _thread

    _transcription_queue = transcription_queue
    _running = True

    _thread = threading.Thread(
        target=_worker,
        name="transcription-worker",
        daemon=True,
    )
    _thread.start()
    logger.info("Transcription worker started")


def stop() -> None:
    """Signal the worker to stop and wait for it to finish."""
    global _running
    _running = False
    if _thread is not None:
        _thread.join(timeout=15)
    logger.info("Transcription worker stopped")


def is_model_loaded() -> bool:
    return _model is not None


# ─────────────────────────────────────────────────────────────────────────────
# Worker thread
# ─────────────────────────────────────────────────────────────────────────────

def _worker() -> None:
    """Main loop: load model once, then transcribe clips as they arrive."""
    global _model

    try:
        _model = _load_model()
    except Exception:
        logger.exception("Cannot load Whisper model – transcription disabled")
        return

    logger.info("Transcription worker ready")

    while _running:
        try:
            item = _transcription_queue.get(timeout=2.0)
        except queue.Empty:
            continue

        clip_id: int = item["id"]
        audio_path: str = item["path"]

        try:
            transcript = _transcribe(audio_path)

            # Update DB (imports database here to avoid circular-import at
            # module load time since database.py imports nothing from this file).
            from app.database import update_transcript
            update_transcript(clip_id, transcript)

            if transcript:
                logger.info("Clip %d: %s", clip_id, transcript[:120])
            else:
                logger.debug("Clip %d: no speech detected by Whisper", clip_id)

            # Push the transcript to connected WebSocket clients.
            if _broadcaster is not None:
                _broadcaster.broadcast_event({
                    "type": "transcript_ready",
                    "clip_id": clip_id,
                    "transcript": transcript,
                })

        except Exception:
            logger.exception("Transcription failed for clip %d", clip_id)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_model():
    """
    Load the faster-whisper model onto the GPU.

    Configuration is read from config.py so the user can easily switch models
    or fall back to CPU by changing WHISPER_DEVICE = "cpu".
    """
    import config
    from faster_whisper import WhisperModel

    # Log expected VRAM requirements so the user knows what to expect.
    vram_table = {
        "tiny":     "~1 GB",
        "tiny.en":  "~1 GB",
        "base":     "~1 GB",
        "base.en":  "~1 GB",
        "small":    "~2 GB",
        "small.en": "~2 GB",
        "medium":   "~5 GB",
        "medium.en":"~5 GB",
        "large-v2": "~10 GB",
        "large-v3": "~10 GB",
    }
    vram = vram_table.get(config.WHISPER_MODEL, "unknown")
    logger.info(
        "Loading Whisper model '%s' on %s (%s) – VRAM: %s",
        config.WHISPER_MODEL,
        config.WHISPER_DEVICE,
        config.WHISPER_COMPUTE_TYPE,
        vram,
    )

    model = WhisperModel(
        config.WHISPER_MODEL,
        device=config.WHISPER_DEVICE,
        compute_type=config.WHISPER_COMPUTE_TYPE,
        download_root=str(config.MODELS_DIR),
        # Use all available CPU cores for pre/post-processing.
        cpu_threads=0,
        num_workers=1,
    )
    logger.info("Whisper model loaded")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def _transcribe(audio_path: str) -> str:
    """
    Transcribe a WAV file and return the transcript as a single string.

    Key parameters:
      beam_size=5       – wider beam gives better accuracy at modest cost.
      language="en"     – skip language detection (scanner is always English).
      vad_filter=True   – Whisper's built-in Silero VAD removes silences,
                          giving cleaner transcripts for scanner audio.
      initial_prompt    – primes the model with police-radio vocabulary so it
                          handles phonetic alphabet, codes, and clipped speech
                          more accurately.
      temperature=0     – deterministic output; disable temperature fallback
                          to prevent Whisper hallucinating on silence.
      no_speech_threshold=0.6
                        – segments below this probability are dropped.
      condition_on_previous_text=False
                        – each segment is decoded independently; prevents
                          errors in one segment contaminating the next.
    """
    import config

    segments, _info = _model.transcribe(
        audio_path,
        beam_size=config.WHISPER_BEAM_SIZE,
        language=config.WHISPER_LANGUAGE,
        vad_filter=True,
        vad_parameters={
            "min_silence_duration_ms": 300,
            "speech_pad_ms": 200,
            "threshold": 0.4,
        },
        initial_prompt=config.WHISPER_INITIAL_PROMPT,
        temperature=0,
        no_speech_threshold=0.6,
        condition_on_previous_text=False,
    )

    # segments is a lazy generator; consume it fully inside this function.
    parts = [seg.text.strip() for seg in segments if seg.text.strip()]
    return " ".join(parts)
