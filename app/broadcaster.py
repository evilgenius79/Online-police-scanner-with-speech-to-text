"""
Thread-safe audio and event broadcaster for WebSocket clients.

The audio capture thread produces data synchronously; WebSocket handlers
consume it asynchronously inside the asyncio event loop. This module bridges
the two worlds using asyncio.AbstractEventLoop.call_soon_threadsafe(), which
schedules a callback to run inside the event loop from any thread.

Usage:
    broadcaster = AudioBroadcaster()

    # At app startup (inside the running event loop):
    broadcaster.set_loop(asyncio.get_running_loop())

    # When a WebSocket client connects:
    client_id, queue = broadcaster.subscribe()

    # WebSocket handler reads from queue:
    data = await queue.get()   # bytes = audio PCM, str = JSON event

    # Audio capture thread calls:
    broadcaster.broadcast_audio(pcm_bytes)
    broadcaster.broadcast_event({"type": "new_clip", ...})

    # When WebSocket client disconnects:
    broadcaster.unsubscribe(client_id)
"""
import asyncio
import json
import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)


class AudioBroadcaster:
    # Each client queue holds at most this many items.
    # At 30 ms/frame that is roughly 9 seconds of buffered audio.
    # Frames are silently dropped when a slow client fills its queue.
    _QUEUE_MAXSIZE = 300

    def __init__(self) -> None:
        self._clients: dict[int, asyncio.Queue] = {}
        self._next_id: int = 0
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # ──────────────────────────────────────────────────────────────────────────
    # Setup (call once from the asyncio event loop at app startup)
    # ──────────────────────────────────────────────────────────────────────────

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Store a reference to the running asyncio event loop."""
        self._loop = loop

    # ──────────────────────────────────────────────────────────────────────────
    # Client lifecycle (called from asyncio coroutines)
    # ──────────────────────────────────────────────────────────────────────────

    def subscribe(self) -> tuple:
        """
        Register a new WebSocket client.
        Returns (client_id, asyncio.Queue).
        Each item in the queue is either bytes (PCM audio) or str (JSON event).
        """
        q: asyncio.Queue = asyncio.Queue(maxsize=self._QUEUE_MAXSIZE)
        with self._lock:
            client_id = self._next_id
            self._next_id += 1
            self._clients[client_id] = q
        logger.debug("WebSocket client %d subscribed (%d total)", client_id, len(self._clients))
        return client_id, q

    def unsubscribe(self, client_id: int) -> None:
        """Deregister a WebSocket client."""
        with self._lock:
            self._clients.pop(client_id, None)
        logger.debug("WebSocket client %d unsubscribed (%d total)", client_id, len(self._clients))

    def client_count(self) -> int:
        with self._lock:
            return len(self._clients)

    # ──────────────────────────────────────────────────────────────────────────
    # Broadcasting (called from background threads)
    # ──────────────────────────────────────────────────────────────────────────

    def broadcast_audio(self, data: bytes) -> None:
        """
        Send raw PCM audio bytes to every connected client.
        Thread-safe. Called from the audio capture thread.
        """
        if self._loop is None or not self._clients:
            return
        with self._lock:
            queues = list(self._clients.values())
        for q in queues:
            self._loop.call_soon_threadsafe(self._enqueue_bytes, q, data)

    def broadcast_event(self, event: dict) -> None:
        """
        Send a JSON-serialisable event dict to every connected client.
        Thread-safe. Called from any background thread.
        """
        if self._loop is None:
            return
        payload = json.dumps(event)
        with self._lock:
            queues = list(self._clients.values())
        for q in queues:
            self._loop.call_soon_threadsafe(self._enqueue_str, q, payload)

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers (always run inside the asyncio event loop)
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _enqueue_bytes(q: asyncio.Queue, data: bytes) -> None:
        try:
            q.put_nowait(data)
        except asyncio.QueueFull:
            pass  # Drop the frame; the client is too slow.

    @staticmethod
    def _enqueue_str(q: asyncio.Queue, data: str) -> None:
        try:
            q.put_nowait(data)
        except asyncio.QueueFull:
            pass
