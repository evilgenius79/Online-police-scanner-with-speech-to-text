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
import base64
import csv
import io
import json
import logging
import os
import queue
import re
import secrets
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# Ensure project root is importable (needed when running as `python app/main.py`)
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from app import audio_capture, cleanup, database, transcriber
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

    cleanup.start()

    logger.info("Scanner server ready at http://%s:%d", config.HOST, config.PORT)

    yield   # ── Application runs here ─────────────────────────────────────────

    # ── Shutdown ──────────────────────────────────────────────────────────────
    audio_capture.stop()
    transcriber.stop()
    cleanup.stop()


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
# Only GET is listed because this API has no mutating POST/PUT/DELETE endpoints.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _basic_auth(request: Request, call_next):
    """
    Optional HTTP Basic Authentication gate.
    Enabled only when AUTH_USERNAME and AUTH_PASSWORD are both non-empty in config.py.
    WebSocket upgrades pass through the same middleware automatically.
    """
    if not config.AUTH_USERNAME or not config.AUTH_PASSWORD:
        return await call_next(request)

    auth = request.headers.get("Authorization", "")
    if auth.startswith("Basic "):
        try:
            decoded = base64.b64decode(auth[6:]).decode("utf-8", errors="replace")
            username, _, password = decoded.partition(":")
            if secrets.compare_digest(username, config.AUTH_USERNAME) and \
               secrets.compare_digest(password, config.AUTH_PASSWORD):
                return await call_next(request)
        except Exception:
            pass

    return Response(
        content="Unauthorized",
        status_code=401,
        headers={"WWW-Authenticate": 'Basic realm="Police Scanner"'},
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
    status["model_info"] = transcriber.get_model_info()
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


@app.get("/api/export")
async def api_export(
    format: str = Query("txt", description="Export format: txt or csv"),
    date: Optional[str] = Query(None, description="Filter by date YYYY-MM-DD"),
) -> StreamingResponse:
    """
    Export clip transcripts as plain text or CSV.

    Query parameters:
      format  – "txt" (default) or "csv"
      date    – optional YYYY-MM-DD filter; omit to export everything
    """
    if format not in ("txt", "csv"):
        raise HTTPException(status_code=400, detail="format must be 'txt' or 'csv'")
    if date is not None and not _DATE_RE.match(date):
        raise HTTPException(status_code=400, detail="date must be YYYY-MM-DD")

    clips = database.export_clips(date=date)
    label = date or "all"

    if format == "csv":
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["id", "start_time", "duration_s", "transcript", "filename"])
        for clip in clips:
            writer.writerow([
                clip["id"],
                clip["start_time"],
                round(clip["duration"], 3),
                clip["transcript"] or "",
                clip["filename"],
            ])
        content = buf.getvalue()
        media_type = "text/csv; charset=utf-8"
        filename = f"scanner_{label}.csv"
    else:
        lines: list[str] = []
        for clip in clips:
            lines.append(f"[{clip['start_time']}]  ({clip['duration']:.1f}s)  {clip['filename']}")
            lines.append(clip["transcript"] or "(no transcript)")
            lines.append("")
        content = "\n".join(lines)
        media_type = "text/plain; charset=utf-8"
        filename = f"scanner_{label}.txt"

    return StreamingResponse(
        iter([content]),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket – live audio + events
# ─────────────────────────────────────────────────────────────────────────────

# Maximum simultaneous WebSocket clients.  Each client gets its own asyncio
# queue and receives every audio frame; too many connections would exhaust
# memory and CPU on the broadcaster's call_soon_threadsafe loop.
_WS_MAX_CLIENTS = 20


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
    await websocket.accept()
    if broadcaster.client_count() >= _WS_MAX_CLIENTS:
        await websocket.close(code=1008, reason="Too many connections")
        logger.warning("WebSocket rejected: client limit (%d) reached", _WS_MAX_CLIENTS)
        return

    client_id, q = broadcaster.subscribe()
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


# ─────────────────────────────────────────────────────────────────────────────
# Admin page + API
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/admin", response_class=HTMLResponse)
async def admin_page() -> FileResponse:
    return FileResponse(str(BASE_DIR / "templates" / "admin.html"))


@app.get("/api/admin/info")
async def api_admin_info() -> dict:
    """Server status + git metadata for the admin dashboard."""
    status = database.get_status()
    status["recording"]    = audio_capture.is_running()
    status["transmitting"] = audio_capture.is_transmitting()
    status["model_loaded"] = transcriber.is_model_loaded()
    status["model_info"]   = transcriber.get_model_info()
    status["ws_clients"]   = broadcaster.client_count()
    status["auth_enabled"] = bool(config.AUTH_USERNAME and config.AUTH_PASSWORD)

    def _git(args: list) -> str:
        try:
            r = subprocess.run(
                args, cwd=str(BASE_DIR),
                capture_output=True, text=True, timeout=5,
            )
            return r.stdout.strip()
        except Exception:
            return ""

    status["git_branch"]      = _git(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    status["git_commit"]      = _git(["git", "log", "-1", "--format=%h"])
    status["git_commit_msg"]  = _git(["git", "log", "-1", "--format=%s"])
    status["git_commit_date"] = _git(["git", "log", "-1", "--format=%ci"])
    return status


@app.post("/api/admin/check-update")
async def api_admin_check_update() -> dict:
    """
    Run `git fetch` then return how many commits HEAD is behind the remote.
    Safe — does not modify any files.
    """
    loop = asyncio.get_running_loop()

    def _check() -> dict:
        def _run(args, timeout=30):
            r = subprocess.run(
                args, cwd=str(BASE_DIR),
                capture_output=True, text=True, timeout=timeout,
            )
            return r.stdout.strip(), r.stderr.strip(), r.returncode

        _, err, rc = _run(["git", "fetch", "origin"])
        if rc != 0:
            return {"error": err or "git fetch failed"}

        branch, _, _  = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        behind_s, _,_ = _run(["git", "rev-list", "--count", f"HEAD..origin/{branch}"])
        ahead_s, _, _ = _run(["git", "rev-list", "--count", f"origin/{branch}..HEAD"])
        latest, _, _  = _run(["git", "log", "-1", f"origin/{branch}", "--format=%h %s"])

        return {
            "branch":        branch,
            "behind":        int(behind_s  or "0"),
            "ahead":         int(ahead_s   or "0"),
            "latest_commit": latest,
        }

    return await loop.run_in_executor(None, _check)


@app.post("/api/admin/update")
async def api_admin_update() -> dict:
    """
    Pull the latest code from the remote, reinstall Python dependencies if
    requirements.txt changed, then schedule a hot-restart via os.execv.
    Returns the combined git / pip output before restarting.
    """
    loop = asyncio.get_running_loop()

    def _update() -> dict:
        lines: list[str] = []

        def _run(args, timeout=300):
            lines.append("$ " + " ".join(str(a) for a in args))
            r = subprocess.run(
                args, cwd=str(BASE_DIR),
                capture_output=True, text=True, timeout=timeout,
            )
            out = (r.stdout + r.stderr).strip()
            if out:
                lines.append(out)
            return r.returncode, r.stdout

        rc, stdout = _run(["git", "pull"], timeout=60)
        if rc != 0:
            return {"success": False, "output": "\n".join(lines)}

        # Reinstall dependencies only if requirements.txt was updated.
        if "requirements.txt" in stdout:
            _run([sys.executable, "-m", "pip", "install", "-r",
                  str(BASE_DIR / "requirements.txt")])

        return {"success": True, "output": "\n".join(lines), "restarting": True}

    result = await loop.run_in_executor(None, _update)

    if result.get("restarting"):
        _schedule_restart()

    return result


@app.post("/api/admin/restart")
async def api_admin_restart() -> dict:
    """Hot-restart the server process via os.execv (no pull)."""
    _schedule_restart()
    return {"restarting": True}


def _schedule_restart() -> None:
    """
    Replace the current process with a fresh copy of itself after a short
    delay so the HTTP response has time to be sent to the client first.
    os.execv is safe: the OS cleans up all threads and file handles;
    SQLite WAL mode recovers cleanly on the next open.
    """
    def _restart():
        time.sleep(0.5)
        logger.info("Restarting server process via os.execv…")
        os.execv(sys.executable, [sys.executable] + sys.argv)

    threading.Thread(target=_restart, name="restart-trigger", daemon=True).start()
