"""
Automatic clip retention / disk cleanup.

A background daemon thread runs once at startup (after a short delay so the
server is fully up), then again every 24 hours.  It deletes WAV files and
database records older than config.CLIP_RETENTION_DAYS days.

Set CLIP_RETENTION_DAYS = 0 in config.py to disable cleanup entirely.
"""
import logging
import shutil
import threading
import time
from datetime import date, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

_thread: threading.Thread | None = None
_stop_event = threading.Event()


def start() -> None:
    """Spawn the cleanup daemon thread."""
    global _thread
    _stop_event.clear()
    _thread = threading.Thread(target=_loop, name="cleanup-worker", daemon=True)
    _thread.start()
    logger.info("Clip cleanup worker started")


def stop() -> None:
    """Signal the cleanup thread to exit and wait for it."""
    _stop_event.set()
    if _thread is not None:
        _thread.join(timeout=10)
    logger.info("Clip cleanup worker stopped")


# ─────────────────────────────────────────────────────────────────────────────

def _loop() -> None:
    # Short initial delay so the server finishes starting up first.
    _stop_event.wait(timeout=30)
    while not _stop_event.is_set():
        _run_cleanup()
        # Sleep 24 hours, but wake up immediately if stop() is called.
        _stop_event.wait(timeout=86_400)


def _run_cleanup() -> None:
    import config
    from app.database import delete_clips_before

    days = config.CLIP_RETENTION_DAYS
    if days <= 0:
        return

    cutoff = date.today() - timedelta(days=days)
    cutoff_str = cutoff.isoformat()   # "YYYY-MM-DD"

    # 1. Delete database records.
    deleted_rows = delete_clips_before(cutoff_str)

    # 2. Delete WAV directories on disk (YYYY-MM-DD folders older than cutoff).
    deleted_dirs = 0
    if config.CLIPS_DIR.is_dir():
        for day_dir in config.CLIPS_DIR.iterdir():
            if not day_dir.is_dir():
                continue
            try:
                dir_date = date.fromisoformat(day_dir.name)
            except ValueError:
                continue   # skip non-date directories
            if dir_date < cutoff:
                try:
                    shutil.rmtree(day_dir)
                    deleted_dirs += 1
                except OSError as exc:
                    logger.warning("Could not delete %s: %s", day_dir, exc)

    if deleted_rows or deleted_dirs:
        logger.info(
            "Retention cleanup: removed %d DB record(s) and %d day-folder(s) "
            "older than %s (%d-day retention)",
            deleted_rows, deleted_dirs, cutoff_str, days,
        )
