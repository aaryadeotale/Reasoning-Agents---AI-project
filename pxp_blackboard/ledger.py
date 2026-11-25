"""Thread-safe token + format-compliance accounting (feeds Milestone 4 cost analysis)."""
from __future__ import annotations

import threading
from collections import defaultdict


class TokenLedger:
    def __init__(self):
        self._lock = threading.Lock()
        self._tok = defaultdict(lambda: {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0})
        self._dec = defaultdict(lambda: {"decisions": 0, "first_try_ok": 0, "repaired": 0, "failed": 0})

    def record(self, agent_id: str, resp) -> None:
        with self._lock:
            t = self._tok[agent_id]
            t["prompt_tokens"] += resp.prompt_tokens
            t["completion_tokens"] += resp.completion_tokens
            t["calls"] += 1

    def note_outcome(self, agent_id: str, attempts: int, ok: bool) -> None:
        with self._lock:
            d = self._dec[agent_id]
            d["decisions"] += 1
            if not ok:
                d["failed"] += 1
            elif attempts == 1:
                d["first_try_ok"] += 1
            else:
                d["repaired"] += 1

    def totals(self) -> dict:
        with self._lock:
            p = sum(t["prompt_tokens"] for t in self._tok.values())
            c = sum(t["completion_tokens"] for t in self._tok.values())
            return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c,
                    "calls": sum(t["calls"] for t in self._tok.values())}

    def per_agent(self) -> dict:
        with self._lock:
            return {a: dict(t) for a, t in self._tok.items()}

    def format_stats(self) -> dict:
        with self._lock:
            n = sum(d["decisions"] for d in self._dec.values())
            first = sum(d["first_try_ok"] for d in self._dec.values())
            rep = sum(d["repaired"] for d in self._dec.values())
            fail = sum(d["failed"] for d in self._dec.values())
            return {"decisions": n, "first_try_ok": first, "repaired": rep, "failed": fail,
                    "first_try_rate": first / n if n else 1.0,
                    "compliance_rate": (first + rep) / n if n else 1.0}
