"""Agent interface + a scripted stand-in used until real LLM agents arrive (Milestone 2)."""
from __future__ import annotations

from typing import Callable, Optional, Protocol

from .models import Draft


class Agent(Protocol):
    agent_id: str
    priority: float

    def act(self, snapshot: dict) -> Optional[Draft]:
        """Return a Draft to post, or None to pass on this board version."""
        ...


class ScriptedAgent:
    def __init__(self, agent_id: str, fn: Callable[[dict], Optional[Draft]], priority: float = 1.0):
        self.agent_id = agent_id
        self.priority = priority
        self._fn = fn

    def act(self, snapshot: dict) -> Optional[Draft]:
        return self._fn(snapshot)
