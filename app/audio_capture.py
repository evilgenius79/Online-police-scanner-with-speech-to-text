"""
Audio capture, Voice Activity Detection, and clip saving.

Flow:
  sounddevice InputStream callback
      → raw_queue (threading.Queue of 30ms PCM frames)
          → _vad_worker thread
              → broadcaster (live audio to WebSocket clients)
              → _save_clip() → WAV file + DB row + transcription_queue

VAD strategy:
  Primary:   Energy-based pre-filter (RMS threshold).
             Scanner squelch gives clean silence between transmissions;
             a low RMS means the squelch is closed and no VAD needed.
  Secondary: webrtcvad (Google's WebRTC VAD) confirms voice in the frame.
             Falls back to energy-only if webrtcvad is unavailable.

Frame format expected by webrtcvad:
  - Sample rate: 16 000 Hz
  - Bit depth:   16-bit signed integer
  - Channels:    1 (mono)
  - Duration:    exactly 30 ms → 480 samples → 960 bytes
"""
import logging
import queue
import threading
import wave
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Try to import webrtcvad; fall back to energy-only VAD if unavailable ──────
try:
    import webrtcvad as _webrtcvad  # noqa: F401
    _HAS_WEBRTCVAD = True
except ImportError:
    _HAS_WEBRTCVAD = False
    logger.warning(
        "webrtcvad not found – using energy-only VAD. "
        "Install with: pip install webrtcvad-wheels"
    )

# ── Module-level state (protected by _state_lock where needed) ────────────────
_state_lock = threading.Lock()
_running = False
_stream = None           # sounddevice InputStream
_vad_thread = None       # background VAD worker thread

_raw_queue: queue.Queue = queue.Queue(maxsize=1000)
_transcription_queue: Optional[queue.Queue] = None
_broadcaster = None
_is_transmitting = False


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def configure(transcription_queue: queue.Queue, broadcaster) -> None:
    """Wire up dependencies before calling start()."""
    global _transcription_queue, _broadcaster
    _transcription_queue = transcription_queue
    _broadcaster = broadcaster


def start(device_index: Optional[int] = None) -> None:
    """
    Start the audio capture stream and VAD worker thread.
    Raises RuntimeError if sounddevice cannot open the requested device.
    """
    global _running, _stream, _vad_thread

    import config
    import sounddevice as sd

    with _state_lock:
        if _running:
            return

        config.CLIPS_DIR.mkdir(parents=True, exist_ok=True)

        _running = True

        # Start VAD worker before the stream so frames are never lost.
        _vad_thread = threading.Thread(
            target=_vad_worker,
            name="vad-worker",
            daemon=True,
        )
        _vad_thread.start()

        # Open the sounddevice InputStream.
        # blocksize == CHUNK_SIZE ensures each callback delivers exactly one
        # 30ms frame, which is the size webrtcvad requires.
        device = device_index if device_index is not None else config.AUDIO_DEVICE_INDEX
        try:
            _stream = sd.InputStream(
                samplerate=config.SAMPLE_RATE,
                channels=config.CHANNELS,
                dtype="float32",
                blocksize=config.CHUNK_SIZE,
                device=device,
                callback=_audio_callback,
                latency="low",
            )
            _stream.start()
        except Exception:
            _running = False
            _vad_thread = None
            raise

        logger.info(
            "Audio capture started – device=%s  %d Hz  %d ms/frame",
            device if device is not None else "default",
            config.SAMPLE_RATE,
            config.CHUNK_DURATION_MS,
        )


def stop() -> None:
    """Gracefully stop capture and wait for the VAD worker to finish."""
    global _running, _stream

    with _state_lock:
        if not _running:
            return
        _running = False
        if _stream is not None:
            try:
                _stream.stop()
                _stream.close()
            except Exception:
                pass
            _stream = None

    if _vad_thread is not None:
        _vad_thread.join(timeout=5)

    logger.info("Audio capture stopped")


def is_running() -> bool:
    return _running


def is_transmitting() -> bool:
    return _is_transmitting


def list_devices() -> list:
    """Return a list of available audio input devices."""
    import sounddevice as sd
    devices = []
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            devices.append({
                "index": idx,
                "name": dev["name"],
                "channels": dev["max_input_channels"],
                "default_samplerate": int(dev["default_samplerate"]),
            })
    return devices


# ─────────────────────────────────────────────────────────────────────────────
# sounddevice callback  (runs in the sounddevice audio thread)
# ─────────────────────────────────────────────────────────────────────────────

def _audio_callback(indata, frames, time_info, status):
    """
    Called by sounddevice for every audio block.
    Converts float32 → int16 and queues the raw bytes.
    Must not raise exceptions (sounddevice would abort the stream).
    """
    if status:
        logger.debug("sounddevice status: %s", status)

    # indata: shape (blocksize, channels), dtype float32, range [-1, 1]
    mono = indata[:, 0]
    # Clip before scaling to avoid int16 overflow from rare out-of-range samples.
    np.clip(mono, -1.0, 1.0, out=mono)
    pcm = (mono * 32767.0).astype(np.int16)

    try:
        _raw_queue.put_nowait(pcm.tobytes())
    except queue.Full:
        pass  # Drop the frame rather than block the audio thread.


# ─────────────────────────────────────────────────────────────────────────────
# VAD worker  (runs in _vad_thread)
# ─────────────────────────────────────────────────────────────────────────────

def _vad_worker() -> None:
    """
    Reads 30ms PCM frames from _raw_queue, applies VAD, and accumulates
    clips.  Runs for the lifetime of the application (daemon thread).
    """
    global _is_transmitting
    import config

    # Initialise webrtcvad if available.
    vad = None
    if _HAS_WEBRTCVAD:
        import webrtcvad
        vad = webrtcvad.Vad(config.VAD_AGGRESSIVENESS)

    # Pre-roll ring buffer: most-recent PRE_ROLL_FRAMES frames kept in memory
    # so they can be prepended to the next clip.
    pre_roll: deque = deque(maxlen=config.PRE_ROLL_FRAMES)

    # Clip state machine
    recording = False
    clip_frames: list = []
    clip_start: Optional[datetime] = None
    consecutive_speech = 0
    consecutive_silence = 0

    # Energy threshold (in int16 RMS units, 0–32767).
    # Below this value a frame is treated as silence without running webrtcvad.
    # The scanner squelch produces near-zero energy when closed.
    ENERGY_SILENCE = 200  # roughly -44 dBFS; tune if needed

    logger.info("VAD worker started (webrtcvad=%s)", _HAS_WEBRTCVAD)

    while _running:
        try:
            frame_bytes: bytes = _raw_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        # ── Broadcast every frame to live WebSocket listeners ─────────────────
        if _broadcaster is not None:
            _broadcaster.broadcast_audio(frame_bytes)

        # ── Sanity check: webrtcvad requires exactly 960 bytes ────────────────
        expected = config.CHUNK_SIZE * 2  # samples × 2 bytes/sample
        if len(frame_bytes) != expected:
            continue

        # ── Energy pre-filter ──────────────────────────────────────────────────
        samples = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt(np.mean(samples ** 2)))
        energy_is_speech = rms > ENERGY_SILENCE

        # ── webrtcvad (only if energy suggests something is there) ─────────────
        if vad is not None and energy_is_speech:
            try:
                is_speech = vad.is_speech(frame_bytes, config.SAMPLE_RATE)
            except Exception:
                is_speech = energy_is_speech
        else:
            is_speech = energy_is_speech

        # ── State machine ──────────────────────────────────────────────────────
        if not recording:
            pre_roll.append(frame_bytes)

            if is_speech:
                consecutive_speech += 1
            else:
                consecutive_speech = max(0, consecutive_speech - 1)

            if consecutive_speech >= config.SPEECH_FRAMES_TRIGGER:
                # Transition: SILENT → RECORDING
                recording = True
                consecutive_silence = 0
                clip_start = datetime.now()
                clip_frames = list(pre_roll)  # include pre-roll audio
                _is_transmitting = True

                if _broadcaster is not None:
                    _broadcaster.broadcast_event({
                        "type": "transmission_start",
                        "timestamp": clip_start.isoformat(),
                    })
                logger.debug("Transmission started")

        else:
            # Currently recording
            clip_frames.append(frame_bytes)

            if is_speech:
                consecutive_silence = 0
            else:
                consecutive_silence += 1

            # Force-save if the clip exceeds the maximum allowed length.
            if len(clip_frames) >= config.MAX_CLIP_FRAMES:
                logger.debug("Max clip length reached – saving partial clip")
                _save_clip(clip_frames, clip_start)
                # Reset and continue recording if the scanner is still active.
                recording = False
                clip_frames = []
                clip_start = None
                consecutive_speech = 0
                consecutive_silence = 0
                _is_transmitting = False

            # Normal end-of-transmission: sustained silence detected.
            elif consecutive_silence >= config.SILENCE_FRAMES_END:
                # Trim trailing silence, keeping POST_ROLL_FRAMES of it.
                keep = len(clip_frames) - consecutive_silence + config.POST_ROLL_FRAMES
                trimmed = clip_frames[:max(keep, 0)]

                if len(trimmed) >= config.MIN_CLIP_FRAMES:
                    _save_clip(trimmed, clip_start)
                else:
                    logger.debug(
                        "Discarding short clip (%d frames < %d min)",
                        len(trimmed), config.MIN_CLIP_FRAMES,
                    )

                recording = False
                clip_frames = []
                clip_start = None
                consecutive_speech = 0
                consecutive_silence = 0
                _is_transmitting = False

                if _broadcaster is not None:
                    _broadcaster.broadcast_event({
                        "type": "transmission_end",
                        "timestamp": datetime.now().isoformat(),
                    })
                logger.debug("Transmission ended")

    logger.info("VAD worker stopped")


# ─────────────────────────────────────────────────────────────────────────────
# Clip persistence
# ─────────────────────────────────────────────────────────────────────────────

def _save_clip(frames: list, start_time: datetime) -> None:
    """
    Write PCM frames to a WAV file, insert a DB record, and enqueue for STT.
    All I/O is synchronous and runs inside the VAD worker thread.
    """
    import config
    from app.database import insert_clip

    try:
        date_str = start_time.strftime("%Y-%m-%d")
        clip_dir = config.CLIPS_DIR / date_str
        clip_dir.mkdir(parents=True, exist_ok=True)

        # Filename: HHMMSS_ffffff.wav  (hours, minutes, seconds, microseconds)
        ts = start_time.strftime("%H%M%S_%f")
        filename = f"{ts}.wav"
        rel_path = f"{date_str}/{filename}"          # stored in DB
        filepath = clip_dir / filename

        audio_bytes = b"".join(frames)
        num_samples = len(audio_bytes) // 2          # 2 bytes per int16 sample
        duration = num_samples / config.SAMPLE_RATE

        # Write WAV container.
        with wave.open(str(filepath), "wb") as wf:
            wf.setnchannels(config.CHANNELS)
            wf.setsampwidth(2)                       # 16-bit = 2 bytes
            wf.setframerate(config.SAMPLE_RATE)
            wf.writeframes(audio_bytes)

        # Insert into database.
        clip_id = insert_clip(rel_path, start_time.isoformat(), duration)

        logger.info("Saved clip %d: %s  (%.1fs)", clip_id, rel_path, duration)

        # Enqueue for transcription.
        if _transcription_queue is not None:
            try:
                _transcription_queue.put_nowait({
                    "id": clip_id,
                    "path": str(filepath),
                    "filename": rel_path,
                    "start_time": start_time.isoformat(),
                    "duration": duration,
                })
            except queue.Full:
                logger.warning("Transcription queue full – clip %d will not be transcribed", clip_id)

        # Notify WebSocket clients about the new (not-yet-transcribed) clip.
        if _broadcaster is not None:
            _broadcaster.broadcast_event({
                "type": "new_clip",
                "clip": {
                    "id": clip_id,
                    "filename": rel_path,
                    "start_time": start_time.isoformat(),
                    "duration": round(duration, 2),
                    "transcript": None,
                    "audio_url": f"/audio/{rel_path}",
                },
            })

    except Exception:
        logger.exception("Error saving clip")
