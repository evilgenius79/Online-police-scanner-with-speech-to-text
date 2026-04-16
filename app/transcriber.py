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
import os
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
_model_info: dict = {} # {"model": "large-v3", "device": "cuda", "compute_type": "float16"}


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


def get_model_info() -> dict:
    """Return which model/device/compute_type actually loaded (empty dict until ready)."""
    return _model_info.copy()


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

    # Re-queue any clips that were saved but never transcribed (e.g. the server
    # was killed mid-queue or crashed before this worker could process them).
    _recover_orphaned_clips()

    while _running:
        try:
            item = _transcription_queue.get(timeout=2.0)
        except queue.Empty:
            continue

        clip_id: int = item["id"]
        audio_path: str = item["path"]

        try:
            transcript = _transcribe(audio_path)

            # Lazy imports avoid circular-import at module load time.
            from app.database import delete_clip, update_transcript

            if not transcript:
                # Whisper found no speech — delete the file and DB record.
                # This satisfies the requirement that only speech-containing
                # clips are kept on disk.
                try:
                    os.remove(audio_path)
                except OSError:
                    pass
                delete_clip(clip_id)
                logger.debug("Clip %d: no speech – file and record deleted", clip_id)
                continue

            # Speech confirmed — persist the transcript.
            update_transcript(clip_id, transcript)
            logger.info("Clip %d: %s", clip_id, transcript[:120])

            # Now broadcast new_clip with the transcript already attached.
            # Clients never see a clip before its transcript is ready.
            if _broadcaster is not None:
                _broadcaster.broadcast_event({
                    "type": "new_clip",
                    "clip": {
                        "id": clip_id,
                        "filename": item["filename"],
                        "start_time": item["start_time"],
                        "duration": round(item["duration"], 2),
                        "transcript": transcript,
                        "audio_url": f"/audio/{item['filename']}",
                    },
                })

        except Exception:
            logger.exception("Transcription failed for clip %d", clip_id)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading  –  automatic CUDA OOM cascade
# ─────────────────────────────────────────────────────────────────────────────

# VRAM estimates for float16 inference on GPU (weights + activation overhead).
_VRAM_TABLE: dict[str, str] = {
    "tiny":            "~1 GB",
    "tiny.en":         "~1 GB",
    "base":            "~1 GB",
    "base.en":         "~1 GB",
    "small":           "~2 GB",
    "small.en":        "~2 GB",
    "medium":          "~3 GB",
    "medium.en":       "~3 GB",
    "large-v2":        "~5 GB",
    "large-v3":        "~5 GB",   # float16; fits comfortably on RTX 4060 (8 GB)
    "large-v3-turbo":  "~2.5 GB", # distilled large-v3; 3× faster, near same accuracy
    "turbo":           "~2.5 GB",
    "distil-large-v3": "~4 GB",
}

# CUDA model cascade — ordered from best quality to smallest.
# The loader starts at the entry that matches config.WHISPER_MODEL and walks
# forward only if a CUDA out-of-memory error occurs.
_CUDA_CHAIN: list[tuple[str, str]] = [
    ("large-v3",        "float16"),        # ~5 GB  ← ideal for RTX 4060
    ("large-v3",        "int8_float16"),   # ~3.5 GB — same weights, int8 matmuls
    ("large-v3-turbo",  "float16"),        # ~2.5 GB — 3× faster distilled model
    ("large-v3-turbo",  "int8_float16"),   # ~1.5 GB
    ("medium.en",       "float16"),        # ~3 GB
    ("medium.en",       "int8_float16"),   # ~1.5 GB
    ("small.en",        "float16"),        # ~1 GB
    ("small.en",        "int8_float16"),   # ~0.5 GB
]

# Last-resort CPU chain (no CUDA required).
_CPU_CHAIN: list[tuple[str, str]] = [
    ("small.en", "int8"),
    ("base.en",  "int8"),
    ("tiny.en",  "int8"),
]


def _is_oom_error(exc: Exception) -> bool:
    """Return True if the exception looks like a CUDA out-of-memory error."""
    msg = str(exc).lower()
    return any(kw in msg for kw in (
        "out of memory",
        "cudaoutofmemory",
        "cuda error",
        "insufficient memory",
        "cannot allocate",
        "allocation failed",
        "memory pool",
    ))


def _load_model():
    """
    Load the fastest/largest Whisper model that fits in GPU memory.

    Starting from the model specified in config.WHISPER_MODEL, the loader
    walks down _CUDA_CHAIN (largest → smallest) and retries on every CUDA
    out-of-memory error.  If all CUDA options are exhausted it falls back to
    CPU inference via _CPU_CHAIN.

    Non-memory errors (bad install, wrong path, etc.) are re-raised immediately
    so the operator sees a clear error rather than a silent downgrade.
    """
    import config
    from faster_whisper import WhisperModel

    def _try(model_name: str, device: str, compute_type: str) -> "WhisperModel":
        vram = _VRAM_TABLE.get(model_name, "?")
        logger.info(
            "Trying Whisper '%s' on %s/%s  (VRAM ≈ %s)",
            model_name, device, compute_type, vram,
        )
        m = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            download_root=str(config.MODELS_DIR),
            cpu_threads=0,
            num_workers=1,
        )
        # Record what actually loaded so /api/status can report it.
        global _model_info
        _model_info = {
            "model":        model_name,
            "device":       device,
            "compute_type": compute_type,
        }
        return m

    # ── CUDA path ─────────────────────────────────────────────────────────────
    if config.WHISPER_DEVICE == "cuda":
        # Find the configured model's position in the chain so we start there
        # (skip larger models that the user deliberately didn't configure).
        start_idx = next(
            (i for i, (m, _) in enumerate(_CUDA_CHAIN)
             if m == config.WHISPER_MODEL),
            0,   # unknown model → start from the top
        )

        for model_name, compute_type in _CUDA_CHAIN[start_idx:]:
            try:
                model = _try(model_name, "cuda", compute_type)
                logger.info(
                    "Whisper loaded: '%s' on cuda/%s", model_name, compute_type,
                )
                return model
            except Exception as exc:
                if _is_oom_error(exc):
                    logger.warning(
                        "CUDA OOM for '%s' %s – trying next option",
                        model_name, compute_type,
                    )
                    continue
                # Non-memory error: propagate so the operator sees it.
                raise

        logger.warning(
            "All CUDA options exhausted (RTX 4060 VRAM full or CUDA unavailable) "
            "– falling back to CPU inference"
        )

    # ── CPU path ──────────────────────────────────────────────────────────────
    # Use the configured model if device=cpu was set explicitly, otherwise use
    # the CPU chain which starts at small.en (good quality / low RAM).
    if config.WHISPER_DEVICE == "cpu":
        try:
            model = _try(config.WHISPER_MODEL, "cpu", "int8")
            logger.info("Whisper loaded: '%s' on cpu/int8", config.WHISPER_MODEL)
            return model
        except Exception as exc:
            if not _is_oom_error(exc):
                raise
            logger.warning("OOM loading configured CPU model – using CPU chain")

    for model_name, compute_type in _CPU_CHAIN:
        try:
            model = _try(model_name, "cpu", compute_type)
            logger.info("Whisper loaded: '%s' on cpu/%s", model_name, compute_type)
            return model
        except Exception as exc:
            if _is_oom_error(exc):
                logger.warning("OOM for cpu '%s' – trying smaller", model_name)
                continue
            raise

    raise RuntimeError("Failed to load any Whisper model (all options exhausted)")


def _recover_orphaned_clips() -> None:
    """
    Re-queue clips that have a DB record but no transcript (NULL).
    This handles cases where the server was stopped while clips were queued,
    or crashed before the transcription worker could process them.
    """
    import config
    from app.database import delete_clip, get_pending_clips

    pending = get_pending_clips()
    if not pending:
        return

    logger.info("Found %d orphaned clip(s) – re-queuing for transcription", len(pending))
    requeued = 0
    for clip in pending:
        filepath = config.CLIPS_DIR / clip["filename"]
        if not filepath.is_file():
            # File missing — clean up the dangling DB record.
            delete_clip(clip["id"])
            logger.warning(
                "Orphaned clip %d has no WAV file – removed DB record", clip["id"]
            )
            continue
        try:
            _transcription_queue.put_nowait({
                "id":         clip["id"],
                "path":       str(filepath),
                "filename":   clip["filename"],
                "start_time": clip["start_time"],
                "duration":   clip["duration"],
            })
            requeued += 1
        except queue.Full:
            logger.warning(
                "Transcription queue full – could not re-queue orphaned clip %d",
                clip["id"],
            )
    if requeued:
        logger.info("Re-queued %d orphaned clip(s)", requeued)


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
