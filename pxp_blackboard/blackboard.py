"""Thread-safe, append-only blackboard with PXP state transitions."""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Callable, Iterable, Optional

from .models import HYPOTHESIS_TAGS, NEGATIVE_TAGS, Draft, Message, Tag
from .validator import BoardClosedError, validate


class Blackboard:
    """Global shared state. All mutation goes through `post()` under one lock.

    State machine (derived from the log, never set directly):
      head            latest PROPOSE/REVISE message (the live hypothesis)
      stances         each agent's latest tag on the head (author = implicit RATIFY)
      negative_streak consecutive REFUTE/REJECT; RATIFY resets it, PROPOSE/REVISE
                      are transparent (so a reject->revise->reject loop still counts)
    """

    def __init__(self, problem: str, agent_ids: Iterable[str], max_messages: int = 40):
        ids = list(agent_ids)
        if len(ids) < 2 or len(set(ids)) != len(ids):
            raise ValueError("need at least two distinct agent ids")
        self.problem = problem
        self.agents = tuple(ids)
        self.max_messages = max_messages
        self._lock = threading.RLock()
        self._log: list[Message] = []
        self._head: Optional[Message] = None
        self._stances: dict[str, Tag] = {}
        self._streak = 0
        self._closed = False
        self._subs: list[Callable[[Message], None]] = []

    # ---- writes -------------------------------------------------------
    def post(self, draft: Draft, *, _timestamp: Optional[float] = None) -> Message:
        with self._lock:
            if self._closed:
                raise BoardClosedError("board is closed")
            if len(self._log) >= self.max_messages:
                raise BoardClosedError("message limit reached")
            validate(draft, head=self._head, agents=self.agents)
            msg = Message(
                seq=len(self._log) + 1, agent_id=draft.agent_id, tag=draft.tag,
                target_id=draft.target_id, prediction=draft.prediction,
                explanation=draft.explanation,
                timestamp=time.time() if _timestamp is None else _timestamp,
                meta=dict(draft.meta),
            )
            self._log.append(msg)
            self._apply(msg)
            for cb in tuple(self._subs):  # inside lock => subscribers see a total order
                cb(msg)                   # callbacks must be non-blocking
            return msg

    def _apply(self, msg: Message) -> None:
        if msg.tag in HYPOTHESIS_TAGS:
            self._head = msg
            self._stances = {msg.agent_id: Tag.RATIFY}
        else:
            self._stances[msg.agent_id] = msg.tag
        if msg.tag is Tag.RATIFY:
            self._streak = 0
        elif msg.tag in NEGATIVE_TAGS:
            self._streak += 1

    def close(self) -> None:
        with self._lock:
            self._closed = True

    def subscribe(self, callback: Callable[[Message], None]) -> None:
        with self._lock:
            self._subs.append(callback)

    # ---- reads --------------------------------------------------------
    @property
    def head(self) -> Optional[Message]:
        with self._lock:
            return self._head

    @property
    def stances(self) -> dict:
        with self._lock:
            return {a: t.value for a, t in self._stances.items()}

    @property
    def version(self) -> int:
        with self._lock:
            return len(self._log)

    @property
    def negative_streak(self) -> int:
        with self._lock:
            return self._streak

    def is_full(self) -> bool:
        with self._lock:
            return len(self._log) >= self.max_messages

    def is_consensus(self) -> bool:
        with self._lock:
            if self._head is None:
                return False
            return all(a == self._head.agent_id or self._stances.get(a) is Tag.RATIFY
                       for a in self.agents)

    def history(self, since_seq: int = 0) -> list[Message]:
        with self._lock:
            return list(self._log[since_seq:])

    def snapshot(self) -> dict:
        """Plain-dict copy of the full state; safe to hand to agents / UI."""
        with self._lock:
            return {
                "problem": self.problem,
                "agents": list(self.agents),
                "version": len(self._log),
                "max_messages": self.max_messages,
                "closed": self._closed,
                "head": self._head.to_dict() if self._head else None,
                "stances": {a: t.value for a, t in self._stances.items()},
                "negative_streak": self._streak,
                "messages": [m.to_dict() for m in self._log],
            }

    # ---- persistence (JSON now; a Redis-backed store can replace this) --
    def save(self, path: str) -> None:
        tmp = f"{path}.tmp"
        with open(tmp, "w") as f:
            json.dump(self.snapshot(), f, indent=2)
        os.replace(tmp, path)

    @classmethod
    def from_dict(cls, data: dict) -> "Blackboard":
        """Rebuild by replaying the log through post(), so state is re-validated."""
        bb = cls(data["problem"], data["agents"], data["max_messages"])
        for m in data["messages"]:
            bb.post(Draft(m["agent_id"], Tag(m["tag"]), m["target_id"], m["prediction"],
                          m["explanation"], m.get("meta", {})), _timestamp=m["timestamp"])
        if data.get("closed"):
            bb.close()
        return bb

    @classmethod
    def load(cls, path: str) -> "Blackboard":
        with open(path) as f:
            return cls.from_dict(json.load(f))
