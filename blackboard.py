"""Thread-safe blackboard implementing the four PXP action tags."""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Callable, Iterable, Optional

from models import NEGATIVE_TAGS, Draft, Message, Tag
from validator import BoardClosedError, validate


class Blackboard:
    """Global shared state.

    A Milestone-1 board is created with a system-seeded initial hypothesis.
    This seed is not an agent action and therefore does not introduce a fifth
    protocol tag. All agent writes use only RATIFY/REVISE/REFUTE/REJECT.
    """

    INITIAL_SEQ = 0

    def __init__(
        self,
        problem: str,
        agent_ids: Iterable[str],
        max_messages: int = 40,
        initial_prediction: str = "",
        initial_explanation: str = "",
        initial_agent_id: Optional[str] = None,
    ):
        ids = list(agent_ids)
        if len(ids) < 2 or len(set(ids)) != len(ids):
            raise ValueError("need at least two distinct agent ids")
        if initial_agent_id is None:
            initial_agent_id = ids[0]
        if initial_agent_id not in ids:
            raise ValueError("initial_agent_id must be a registered agent")
        if not isinstance(initial_prediction, str) or not initial_prediction.strip():
            raise ValueError("initial_prediction must be non-empty")
        if not isinstance(initial_explanation, str) or not initial_explanation.strip():
            raise ValueError("initial_explanation must be non-empty")

        self.problem = problem
        self.agents = tuple(ids)
        self.max_messages = max_messages
        self._lock = threading.RLock()
        self._log: list[Message] = []
        self._initial_timestamp = time.time()
        self._initial_agent_id = initial_agent_id
        self._initial_prediction = initial_prediction
        self._initial_explanation = initial_explanation
        self._head = Message(
            seq=self.INITIAL_SEQ,
            agent_id=initial_agent_id,
            tag=Tag.RATIFY,
            target_id=None,
            prediction=initial_prediction,
            explanation=initial_explanation,
            timestamp=self._initial_timestamp,
            meta={"system_seed": True},
        )
        self._stances: dict[str, Tag] = {initial_agent_id: Tag.RATIFY}
        self._streak = 0
        self._closed = False
        self._subs: list[Callable[[Message], None]] = []

    def post(self, draft: Draft, *, _timestamp: Optional[float] = None) -> Message:
        with self._lock:
            if self._closed:
                raise BoardClosedError("board is closed")
            if len(self._log) >= self.max_messages:
                raise BoardClosedError("message limit reached")

            validate(draft, head=self._head, agents=self.agents)

            msg = Message(
                seq=len(self._log) + 1,
                agent_id=draft.agent_id,
                tag=draft.tag,
                target_id=draft.target_id,
                prediction=draft.prediction,
                explanation=draft.explanation,
                timestamp=time.time() if _timestamp is None else _timestamp,
                meta=dict(draft.meta),
            )
            self._log.append(msg)
            self._apply(msg)
            for cb in tuple(self._subs):
                cb(msg)
            return msg

    def _apply(self, msg: Message) -> None:
        if msg.tag is Tag.REVISE:
            self._head = msg
            self._stances = {msg.agent_id: Tag.RATIFY}
        else:
            self._stances[msg.agent_id] = msg.tag

        if msg.tag is Tag.RATIFY:
            self._streak = 0
        elif msg.tag in NEGATIVE_TAGS:
            self._streak += 1
        # REVISE is deliberately transparent to the negative streak.

    def close(self) -> None:
        with self._lock:
            self._closed = True

    def subscribe(self, callback: Callable[[Message], None]) -> None:
        with self._lock:
            self._subs.append(callback)

    @property
    def head(self) -> Message:
        with self._lock:
            return self._head

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
            return all(
                a == self._head.agent_id
                or self._stances.get(a) is Tag.RATIFY
                for a in self.agents
            )

    def history(self, since_seq: int = 1) -> list[Message]:
        with self._lock:
            if since_seq <= 1:
                return list(self._log)
            return [m for m in self._log if m.seq >= since_seq]

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "problem": self.problem,
                "agents": list(self.agents),
                "version": len(self._log),
                "max_messages": self.max_messages,
                "closed": self._closed,
                "initial": {
                    "agent_id": self._initial_agent_id,
                    "prediction": self._initial_prediction,
                    "explanation": self._initial_explanation,
                    "timestamp": self._initial_timestamp,
                },
                "head": self._head.to_dict(),
                "stances": {a: t.value for a, t in self._stances.items()},
                "negative_streak": self._streak,
                "messages": [m.to_dict() for m in self._log],
            }

    def save(self, path: str) -> None:
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.snapshot(), f, indent=2)
        os.replace(tmp, path)

    @classmethod
    def from_dict(cls, data: dict) -> "Blackboard":
        initial = data["initial"]
        bb = cls(
            data["problem"],
            data["agents"],
            data["max_messages"],
            initial["prediction"],
            initial["explanation"],
            initial["agent_id"],
        )
        bb._initial_timestamp = initial.get("timestamp", bb._initial_timestamp)
        bb._initial_agent_id = initial["agent_id"]
        bb._initial_prediction = initial["prediction"]
        bb._initial_explanation = initial["explanation"]
        # Restore the exact seed timestamp/state before replaying actions.
        bb._head = Message(
            seq=0,
            agent_id=initial["agent_id"],
            tag=Tag.RATIFY,
            target_id=None,
            prediction=initial["prediction"],
            explanation=initial["explanation"],
            timestamp=bb._initial_timestamp,
            meta={"system_seed": True},
        )
        bb._stances = {initial["agent_id"]: Tag.RATIFY}
        for m in data["messages"]:
            bb.post(
                Draft(
                    m["agent_id"], Tag(m["tag"]), m["target_id"],
                    m["prediction"], m["explanation"], m.get("meta", {})
                ),
                _timestamp=m["timestamp"],
            )
        if data.get("closed"):
            bb.close()
        return bb

    @classmethod
    def load(cls, path: str) -> "Blackboard":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
