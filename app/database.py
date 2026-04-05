"""
SQLite database operations with FTS5 full-text search.

Each background thread gets its own thread-local connection.
The asyncio event loop (FastAPI routes) shares one connection via the main thread.
WAL mode allows concurrent reads while a write is in progress.
"""
import math
import sqlite3
import threading
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Thread-local storage for SQLite connections.
# SQLite connections must be used in the thread that created them.
_local = threading.local()

# Will be set to the DB path by init_db()
_db_path: Optional[str] = None


def _get_conn() -> sqlite3.Connection:
    """Return a thread-local SQLite connection, creating it if needed."""
    if not hasattr(_local, "conn") or _local.conn is None:
        conn = sqlite3.connect(_db_path, check_same_thread=True)
        conn.row_factory = sqlite3.Row
        # WAL mode: readers don't block writers; writers don't block readers.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA cache_size=-8000")  # 8 MB cache per connection
        _local.conn = conn
    return _local.conn


def init_db(db_path: str) -> None:
    """Initialize the database schema. Must be called before any other function."""
    global _db_path
    _db_path = db_path

    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS clips (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            filename    TEXT    NOT NULL UNIQUE,
            start_time  TEXT    NOT NULL,
            duration    REAL    NOT NULL,
            transcript  TEXT,
            created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_clips_start_time
            ON clips (start_time);

        CREATE INDEX IF NOT EXISTS idx_clips_date
            ON clips (date(start_time));

        -- FTS5 virtual table for full-text search on transcripts.
        -- content='clips' keeps it in sync via triggers below.
        CREATE VIRTUAL TABLE IF NOT EXISTS clips_fts USING fts5(
            transcript,
            content='clips',
            content_rowid='id'
        );

        -- Keep FTS index in sync with the clips table.
        CREATE TRIGGER IF NOT EXISTS clips_ai
            AFTER INSERT ON clips BEGIN
                INSERT INTO clips_fts (rowid, transcript)
                VALUES (new.id, COALESCE(new.transcript, ''));
            END;

        CREATE TRIGGER IF NOT EXISTS clips_au
            AFTER UPDATE OF transcript ON clips BEGIN
                INSERT INTO clips_fts (clips_fts, rowid, transcript)
                VALUES ('delete', old.id, COALESCE(old.transcript, ''));
                INSERT INTO clips_fts (rowid, transcript)
                VALUES (new.id, COALESCE(new.transcript, ''));
            END;

        CREATE TRIGGER IF NOT EXISTS clips_ad
            AFTER DELETE ON clips BEGIN
                INSERT INTO clips_fts (clips_fts, rowid, transcript)
                VALUES ('delete', old.id, COALESCE(old.transcript, ''));
            END;
    """)
    conn.commit()
    logger.info("Database initialized at %s", db_path)


def insert_clip(filename: str, start_time: str, duration: float) -> int:
    """
    Insert a new clip record.
    Returns the new clip's ID.
    Called from the audio capture thread.
    """
    conn = _get_conn()
    cur = conn.execute(
        "INSERT INTO clips (filename, start_time, duration) VALUES (?, ?, ?)",
        (filename, start_time, round(duration, 3)),
    )
    conn.commit()
    return cur.lastrowid


def update_transcript(clip_id: int, transcript: str) -> None:
    """
    Set the transcript for a clip after STT completes.
    Called from the transcription thread.
    """
    conn = _get_conn()
    conn.execute(
        "UPDATE clips SET transcript = ? WHERE id = ?",
        (transcript, clip_id),
    )
    conn.commit()


def get_clips(
    page: int = 1,
    per_page: int = 20,
    date: Optional[str] = None,
) -> dict:
    """
    Return paginated clips, newest first.
    Optionally filter to a single date (YYYY-MM-DD).
    """
    conn = _get_conn()
    offset = (page - 1) * per_page

    if date:
        total = conn.execute(
            "SELECT COUNT(*) FROM clips WHERE date(start_time) = ?",
            (date,),
        ).fetchone()[0]
        rows = conn.execute(
            """SELECT id, filename, start_time, duration, transcript, created_at
               FROM clips
               WHERE date(start_time) = ?
               ORDER BY start_time DESC
               LIMIT ? OFFSET ?""",
            (date, per_page, offset),
        ).fetchall()
    else:
        total = conn.execute("SELECT COUNT(*) FROM clips").fetchone()[0]
        rows = conn.execute(
            """SELECT id, filename, start_time, duration, transcript, created_at
               FROM clips
               ORDER BY start_time DESC
               LIMIT ? OFFSET ?""",
            (per_page, offset),
        ).fetchall()

    return {
        "clips": [dict(r) for r in rows],
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": math.ceil(total / per_page) if total > 0 else 0,
    }


def get_clip(clip_id: int) -> Optional[dict]:
    """Return a single clip by ID, or None if not found."""
    conn = _get_conn()
    row = conn.execute(
        """SELECT id, filename, start_time, duration, transcript, created_at
           FROM clips WHERE id = ?""",
        (clip_id,),
    ).fetchone()
    return dict(row) if row else None


def search_clips(
    query: str,
    page: int = 1,
    per_page: int = 20,
) -> dict:
    """
    Full-text search clips by transcript using FTS5.
    Falls back to a LIKE search if the FTS5 query syntax is invalid.
    """
    conn = _get_conn()
    offset = (page - 1) * per_page
    fts_query = _build_fts_query(query)

    try:
        total = conn.execute(
            "SELECT COUNT(*) FROM clips_fts WHERE clips_fts MATCH ?",
            (fts_query,),
        ).fetchone()[0]

        rows = conn.execute(
            """SELECT c.id, c.filename, c.start_time, c.duration,
                      c.transcript, c.created_at,
                      snippet(clips_fts, 0, '<mark>', '</mark>', ' ... ', 20) AS snippet
               FROM clips c
               JOIN clips_fts ON clips_fts.rowid = c.id
               WHERE clips_fts MATCH ?
               ORDER BY rank
               LIMIT ? OFFSET ?""",
            (fts_query, per_page, offset),
        ).fetchall()

    except sqlite3.OperationalError:
        # FTS5 query syntax error – fall back to LIKE search.
        like = f"%{query}%"
        total = conn.execute(
            "SELECT COUNT(*) FROM clips WHERE transcript LIKE ?",
            (like,),
        ).fetchone()[0]
        rows = conn.execute(
            """SELECT id, filename, start_time, duration, transcript, created_at,
                      transcript AS snippet
               FROM clips
               WHERE transcript LIKE ?
               ORDER BY start_time DESC
               LIMIT ? OFFSET ?""",
            (like, per_page, offset),
        ).fetchall()

    return {
        "clips": [dict(r) for r in rows],
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": math.ceil(total / per_page) if total > 0 else 0,
    }


def get_dates() -> list:
    """Return all dates that have clips, with clip counts, newest first."""
    conn = _get_conn()
    rows = conn.execute(
        """SELECT date(start_time) AS date, COUNT(*) AS count
           FROM clips
           GROUP BY date(start_time)
           ORDER BY date DESC""",
    ).fetchall()
    return [dict(r) for r in rows]


def get_status() -> dict:
    """Return aggregate statistics."""
    conn = _get_conn()
    total = conn.execute("SELECT COUNT(*) FROM clips").fetchone()[0]
    pending = conn.execute(
        "SELECT COUNT(*) FROM clips WHERE transcript IS NULL"
    ).fetchone()[0]
    return {"total_clips": total, "pending_transcription": pending}


def _build_fts_query(query: str) -> str:
    """
    Convert a plain-text search string into a safe FTS5 query.

    Wraps the input in double-quotes to perform a phrase search, escaping any
    internal double-quote characters. This prevents FTS5 syntax injection while
    still giving useful phrase-match results.
    """
    cleaned = query.strip()
    if not cleaned:
        return '""'
    # Escape double quotes inside the string, then wrap in double quotes
    # to treat the whole input as a phrase.
    escaped = cleaned.replace('"', '""')
    return f'"{escaped}"'
