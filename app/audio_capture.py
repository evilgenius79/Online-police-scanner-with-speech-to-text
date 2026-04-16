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
import json
import logging
import queue
import threading
import time
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
_stream = None            # sounddevice InputStream
_vad_thread = None        # background VAD worker thread
_watchdog_thread = None   # stream health monitor / auto-reconnect thread
_stream_stopped = threading.Event()   # set when the stream dies unexpectedly

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
    Start the audio capture stream, VAD worker, and reconnect watchdog.
    Raises RuntimeError if sounddevice cannot open the requested device.
    """
    global _running, _stream, _vad_thread, _watchdog_thread

    import config
    import sounddevice as sd

    with _state_lock:
        if _running:
            return

        config.CLIPS_DIR.mkdir(parents=True, exist_ok=True)

        _running = True
        _stream_stopped.clear()

        # Start VAD worker before the stream so frames are never lost.
        _vad_thread = threading.Thread(
            target=_vad_worker,
            name="vad-worker",
            daemon=True,
        )
        _vad_thread.start()

        # Start watchdog that auto-reconnects if the device is unplugged.
        _watchdog_thread = threading.Thread(
            target=_reconnect_watchdog,
            name="audio-watchdog",
            daemon=True,
        )
        _watchdog_thread.start()

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
                finished_callback=_on_stream_finished,
                latency="low",
            )
            _stream.start()
        except Exception:
            _running = False
            _vad_thread = None
            _watchdog_thread = None
            raise

        logger.info(
            "Audio capture started – device=%s  %d Hz  %d ms/frame",
            device if device is not None else "default",
            config.SAMPLE_RATE,
            config.CHUNK_DURATION_MS,
        )


def stop() -> None:
    """Gracefully stop capture and wait for the worker threads to finish."""
    global _running, _stream

    with _state_lock:
        if not _running:
            return
        _running = False
        _stream_stopped.set()   # wake the watchdog so it exits promptly
        if _stream is not None:
            try:
                _stream.stop()
                _stream.close()
            except Exception:
                pass
            _stream = None

    if _vad_thread is not None:
        _vad_thread.join(timeout=5)
    if _watchdog_thread is not None:
        _watchdog_thread.join(timeout=5)

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
# Stream finished callback + auto-reconnect watchdog
# ─────────────────────────────────────────────────────────────────────────────

def _on_stream_finished() -> None:
    """Called by sounddevice when the stream stops for any reason."""
    if _running:
        logger.warning("Audio stream stopped unexpectedly – reconnect watchdog activated")
        _stream_stopped.set()


def _reconnect_watchdog() -> None:
    """
    Monitors the audio stream and restarts it after device disconnects.
    Uses exponential backoff (5 s → 10 s → 20 s → 60 s max) between retries.
    """
    delay = 5
    while _running:
        # Block until the stream-stopped event fires (or we're shutting down).
        _stream_stopped.wait()
        if not _running:
            break

        logger.info("Reconnect watchdog: retrying in %d s…", delay)
        time.sleep(delay)
        if not _running:
            break

        try:
            _restart_stream()
            _stream_stopped.clear()
            delay = 5   # reset backoff after a successful reconnect
            logger.info("Audio device reconnected successfully")
        except Exception as exc:
            delay = min(delay * 2, 60)
            logger.warning("Reconnect failed (%s) – will retry in %d s", exc, delay)


def _restart_stream() -> None:
    """Close the current stream (if any) and open a fresh one."""
    global _stream
    import config
    import sounddevice as sd

    with _state_lock:
        if _stream is not None:
            try:
                _stream.stop()
                _stream.close()
            except Exception:
                pass
            _stream = None

        device = config.AUDIO_DEVICE_INDEX
        new_stream = sd.InputStream(
            samplerate=config.SAMPLE_RATE,
            channels=config.CHANNELS,
            dtype="float32",
            blocksize=config.CHUNK_SIZE,
            device=device,
            callback=_audio_callback,
            finished_callback=_on_stream_finished,
            latency="low",
        )
        new_stream.start()
        _stream = new_stream
        logger.info(
            "Audio stream restarted – device=%s",
            device if device is not None else "default",
        )


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

    # indata: shape (blocksize, channels), dtype float32, range [-1, 1].
    # PortAudio provides indata as a read-only buffer on some platforms, so we
    # must NOT use out=indata[:, 0] or any other in-place write on it.
    # np.clip without `out` always returns a new array — safe here.
    mono = np.clip(indata[:, 0], -1.0, 1.0)
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

def _compute_waveform(audio_bytes: bytes, num_bars: int = 60) -> str:
    """
    Downsample raw int16 PCM to `num_bars` RMS amplitude values.
    Returns a compact JSON string suitable for storage in SQLite and
    transmission to the browser for canvas rendering.
    """
    samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
    samples /= 32768.0
    chunk = max(1, len(samples) // num_bars)
    bars = []
    for i in range(num_bars):
        seg = samples[i * chunk : (i + 1) * chunk]
        bars.append(round(float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0, 4))
    return json.dumps(bars)


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

        # Compute waveform before writing (frames still in memory).
        waveform_json = _compute_waveform(audio_bytes)

        # Write WAV container.
        with wave.open(str(filepath), "wb") as wf:
            wf.setnchannels(config.CHANNELS)
            wf.setsampwidth(2)                       # 16-bit = 2 bytes
            wf.setframerate(config.SAMPLE_RATE)
            wf.writeframes(audio_bytes)

        # Insert into database (waveform stored alongside metadata).
        clip_id = insert_clip(rel_path, start_time.isoformat(), duration, waveform_json)

        logger.info("Saved clip %d: %s  (%.1fs)", clip_id, rel_path, duration)

        # Enqueue for transcription.
        # NOTE: the new_clip WebSocket event is sent by the transcriber AFTER
        # Whisper confirms speech is present.  Clips with no speech are deleted
        # automatically, so we never broadcast a clip that will disappear.
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

    except Exception:
        logger.exception("Error saving clip")
