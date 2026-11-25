"""Thread-safe event hub: scheduler threads emit, asyncio websocket handlers consume.
Every event is kept, so a client that connects late (or reconnects) gets a full replay."""
from __future__ import annotations

import asyncio
import threading
import time

TERMINAL = {"session_finished", "session_error"}


class _Sub:
    def __init__(self, loop):
        self.loop, self.queue = loop, asyncio.Queue()


class EventHub:
    def __init__(self):
        self._lock = threading.Lock()
        self._events: list[dict] = []
        self._subs: list[_Sub] = []

    def emit(self, event: dict) -> None:
        with self._lock:
            ev = {**event, "id": len(self._events), "ts": time.time()}
            self._events.append(ev)
            subs = list(self._subs)
        for s in subs:
            try:
                s.loop.call_soon_threadsafe(s.queue.put_nowait, ev)
            except RuntimeError:  # that client's event loop is gone
                pass

    def subscribe(self, loop):
        """Atomically snapshot the backlog and register, so no event is missed or duplicated."""
        sub = _Sub(loop)
        with self._lock:
            backlog = list(self._events)
            self._subs.append(sub)
        return sub, backlog

    def unsubscribe(self, sub: _Sub) -> None:
        with self._lock:
            if sub in self._subs:
                self._subs.remove(sub)

    def events(self) -> list[dict]:
        with self._lock:
            return list(self._events)
