"""
FastAPI application: routes, WebSocket endpoint, and lifecycle management.

Startup order:
  1. init_db()          – create/verify SQLite schema
  2. broadcaster.set_loop() – store asyncio loop reference for thread→async bridge
  3. transcriber.start() – spawn GPU transcription worker
  4. audio_capture.start() – open sounddevice stream + VAD thread

Shutdown order (reverse):
  audio_capture.stop() → transcriber.stop()
"""
import asyncio
import json
import logging
import os
import queue
import re
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

# Ensure project root is importable (needed when running as `python app/main.py`)
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from app import audio_capture, database, transcriber
from app.broadcaster import AudioBroadcaster

logger = logging.getLogger(__name__)

# ── Shared state ──────────────────────────────────────────────────────────────
broadcaster = AudioBroadcaster()
transcription_queue: queue.Queue = queue.Queue(maxsize=200)

BASE_DIR = Path(__file__).parent.parent


# ─────────────────────────────────────────────────────────────────────────────
# Lifecycle  (lifespan replaces the deprecated @on_event decorator)
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Run startup tasks, yield control to the application, then shut down."""
    # ── Startup ───────────────────────────────────────────────────────────────
    database.init_db(str(config.DB_PATH))

    # Store the running event loop so background threads can schedule callbacks.
    loop = asyncio.get_running_loop()
    broadcaster.set_loop(loop)

    audio_capture.configure(transcription_queue, broadcaster)
    transcriber.configure(broadcaster)

    transcriber.start(transcription_queue)

    try:
        audio_capture.start()
    except Exception as exc:
        # Keep the web interface alive for browsing past clips even if the
        # audio device is unavailable.
        logger.error("Audio capture could not start: %s", exc)

    logger.info("Scanner server ready at http://%s:%d", config.HOST, config.PORT)

    yield   # ── Application runs here ─────────────────────────────────────────

    # ── Shutdown ──────────────────────────────────────────────────────────────
    audio_capture.stop()
    transcriber.stop()


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Police Scanner",
    lifespan=_lifespan,
    # Disable auto-generated API docs to reduce attack surface on a local server.
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

# Static files (CSS, JS, etc.)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Allow connections from localhost and the LAN IP so other devices on the same
# network can view the scanner in a browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# HTML frontend
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index() -> FileResponse:
    return FileResponse(str(BASE_DIR / "templates" / "index.html"))


# ─────────────────────────────────────────────────────────────────────────────
# Audio clip serving
# ─────────────────────────────────────────────────────────────────────────────

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_FILENAME_RE = re.compile(r"^[\w\-]+\.wav$")


@app.get("/audio/{date}/{filename}")
async def serve_audio(date: str, filename: str) -> FileResponse:
    """
    Serve a saved audio clip.

    Path-traversal prevention:
      - `date` must match YYYY-MM-DD.
      - `filename` must match the pattern written by audio_capture (word chars
        + hyphens only, .wav extension).
      - The resolved filepath must be inside CLIPS_DIR.
    """
    if not _DATE_RE.match(date):
        raise HTTPException(status_code=400, detail="Invalid date format")
    # os.path.basename strips any sneaky directory components from filename.
    safe_name = os.path.basename(filename)
    if not _FILENAME_RE.match(safe_name):
        raise HTTPException(status_code=400, detail="Invalid filename")

    filepath = config.CLIPS_DIR / date / safe_name

    # Hard check: the resolved path must be inside CLIPS_DIR.
    try:
        filepath.resolve().relative_to(config.CLIPS_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")

    if not filepath.is_file():
        raise HTTPException(status_code=404, detail="Clip not found")

    return FileResponse(
        str(filepath),
        media_type="audio/wav",
        headers={"Accept-Ranges": "bytes", "Cache-Control": "public, max-age=31536000"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# REST API
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/status")
async def api_status() -> dict:
    """Current scanner status and aggregate statistics."""
    status = database.get_status()
    status["recording"] = audio_capture.is_running()
    status["transmitting"] = audio_capture.is_transmitting()
    status["model_loaded"] = transcriber.is_model_loaded()
    status["ws_clients"] = broadcaster.client_count()
    return status


@app.get("/api/devices")
async def api_devices() -> list:
    """List audio input devices (helpful for configuring AUDIO_DEVICE_INDEX)."""
    try:
        return audio_capture.list_devices()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/clips")
async def api_clips(
    page: int = Query(1, ge=1, description="Page number (1-based)"),
    per_page: int = Query(20, ge=1, le=100, description="Results per page"),
    date: Optional[str] = Query(None, description="Filter by date YYYY-MM-DD"),
) -> dict:
    """Return paginated clips, newest first."""
    # Validate optional date parameter.
    if date is not None and not _DATE_RE.match(date):
        raise HTTPException(status_code=400, detail="date must be YYYY-MM-DD")

    result = database.get_clips(page=page, per_page=per_page, date=date)
    for clip in result["clips"]:
        clip["audio_url"] = f"/audio/{clip['filename']}"
    return result


@app.get("/api/clips/{clip_id}")
async def api_clip(clip_id: int) -> dict:
    """Return a single clip by ID."""
    clip = database.get_clip(clip_id)
    if clip is None:
        raise HTTPException(status_code=404, detail="Clip not found")
    clip["audio_url"] = f"/audio/{clip['filename']}"
    return clip


@app.get("/api/search")
async def api_search(
    q: str = Query(..., min_length=1, max_length=500, description="Search query"),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
) -> dict:
    """Full-text search of clip transcripts."""
    result = database.search_clips(query=q.strip(), page=page, per_page=per_page)
    for clip in result["clips"]:
        clip["audio_url"] = f"/audio/{clip['filename']}"
    return result


@app.get("/api/dates")
async def api_dates() -> list:
    """Return all dates that have clips, with counts, newest first."""
    return database.get_dates()


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket – live audio + events
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws/audio")
async def ws_audio(websocket: WebSocket) -> None:
    """
    WebSocket endpoint that delivers:
      - Binary frames: raw 16-bit signed PCM @ 16 000 Hz, mono, little-endian.
                       Each frame is ~960 bytes (30 ms of audio).
      - Text frames:   UTF-8 JSON events (new_clip, transcript_ready,
                       transmission_start, transmission_end, ping).

    The client differentiates by checking `event.data instanceof ArrayBuffer`.
    """
    client_id, q = broadcaster.subscribe()
    await websocket.accept()
    logger.info("WebSocket client %d connected from %s", client_id, websocket.client)

    try:
        while True:
            # Wait up to 30 s for data; send a ping keepalive if idle.
            try:
                data = await asyncio.wait_for(q.get(), timeout=30.0)
            except asyncio.TimeoutError:
                await websocket.send_text(json.dumps({"type": "ping"}))
                continue

            if isinstance(data, bytes):
                await websocket.send_bytes(data)
            else:
                await websocket.send_text(data)

    except WebSocketDisconnect:
        logger.info("WebSocket client %d disconnected", client_id)
    except Exception as exc:
        logger.warning("WebSocket client %d error: %s", client_id, exc)
    finally:
        broadcaster.unsubscribe(client_id)
