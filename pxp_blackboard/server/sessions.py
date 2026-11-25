"""Builds and runs one panel session (board + LLM agents + scheduler) in a background thread."""
from __future__ import annotations

import threading
import uuid
from typing import Literal, Optional

from pydantic import BaseModel, Field

from ..blackboard import Blackboard
from ..counterfactual import CounterfactualSandbox
from ..ledger import TokenLedger
from ..llm.backends import LLMBackend, MockBackend, OpenAICompatBackend
from ..llm.mock import JITTER, ROLE_LATENCY, SCENARIOS, demo_responder, role_for_persona
from ..llm_agent import LLMAgent
from ..prompts import PromptLibrary
from ..scheduler import Scheduler, SchedulerConfig
from .hub import EventHub


class AgentSpec(BaseModel):
    id: str
    persona: str = "generalist"
    priority: float = 1.0
    role: Optional[str] = None       # mock only: analyst | skeptic | auditor (default: from persona)
    counterfactual: bool = False     # Milestone 3: may run counterfactual reflection on deadlock


class BackendSpec(BaseModel):
    type: Literal["mock-demo", "openai"] = "mock-demo"
    base_url: Optional[str] = None   # e.g. http://localhost:8000/v1 (vLLM) or :11434/v1 (Ollama)
    model: Optional[str] = None
    api_key: Optional[str] = None
    flaky: bool = False              # mock only: emit malformed first attempts
    speed: float = Field(0.0, ge=0.0, le=50.0)  # mock only: 0 = instant, 1 = realistic thinking time, 2 = twice as fast


class SessionRequest(BaseModel):
    scenario: str = "revenue"        # scripted story (see llm/mock.py); also supplies default problem/agents
    problem: Optional[str] = None
    agents: Optional[list[AgentSpec]] = None
    backend: BackendSpec = Field(default_factory=BackendSpec)
    max_messages: int = 30
    max_concurrent: int = 2
    impasse_threshold: int = 4
    max_reflections: int = 1         # per counterfactual agent


DEFAULT_AGENTS = [AgentSpec(id="Analyst", persona="data_analyst", priority=5),
                  AgentSpec(id="Skeptic", persona="skeptic"),
                  AgentSpec(id="Auditor", persona="generalist")]


def make_backend(spec: BackendSpec, persona_id: str, role: Optional[str] = None,
                 scenario: str = "revenue") -> LLMBackend:
    if spec.type == "mock-demo":
        role = role or role_for_persona(persona_id)
        latency = ROLE_LATENCY.get(role, 1.5) / spec.speed if spec.speed else 0.0
        return MockBackend(demo_responder(role, flaky=spec.flaky, scenario=scenario), name="mock-demo",
                           latency_s=latency, jitter=JITTER)
    if not (spec.base_url and spec.model):
        raise ValueError("backend 'openai' needs base_url and model")
    return OpenAICompatBackend(spec.base_url, spec.model, spec.api_key)


class Session:
    def __init__(self, req: SessionRequest, library: PromptLibrary):
        self.id = uuid.uuid4().hex[:12]
        self.hub, self.ledger = EventHub(), TokenLedger()
        if req.scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario {req.scenario!r}; available: {sorted(SCENARIOS)}")
        sc = SCENARIOS[req.scenario]
        specs = req.agents or [AgentSpec(id=t[0], persona=t[1], role=t[2], priority=t[3],
                                         counterfactual=bool(t[4]) if len(t) > 4 else False)
                               for t in sc["agents"]]
        problem = req.problem or sc["problem"]
        agents = [LLMAgent(s.id, library.get(s.persona),
                           make_backend(req.backend, s.persona, s.role, req.scenario),
                           priority=s.priority, ledger=self.ledger, counterfactual=s.counterfactual)
                  for s in specs]
        self.specs, self.req = specs, req
        self.library = library
        self.board = Blackboard(problem, [a.agent_id for a in agents], req.max_messages)
        self.board.subscribe(self._on_message)  # callback runs inside the board lock: just emit
        self.scheduler = Scheduler(self.board, agents,
                                   SchedulerConfig(req.max_concurrent, req.impasse_threshold,
                                                   max_reflections=req.max_reflections),
                                   listener=self.hub.emit,
                                   sandbox=CounterfactualSandbox(impasse_threshold=req.impasse_threshold))
        self.status, self.result = "created", None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _on_message(self, m) -> None:
        self.hub.emit({"type": "message", "message": m.to_dict(), "version": self.board.version,
                       "negative_streak": self.board.negative_streak, "stances": self.board.stances})

    def _run(self) -> None:
        self.status = "running"
        r = self.req
        agents = []
        for s in self.specs:
            p = self.library.get(s.persona)
            agents.append({**s.model_dump(), "title": p.title, "domain": p.domain})
        self.hub.emit({"type": "session_started", "problem": self.board.problem, "agents": agents,
                       "scenario": {"id": r.scenario, "title": SCENARIOS[r.scenario]["title"]},
                       "backend": {"type": r.backend.type, "model": r.backend.model or "scripted",
                                   "flaky": r.backend.flaky, "speed": r.backend.speed},
                       "limits": {"max_messages": r.max_messages, "max_concurrent": r.max_concurrent,
                                  "impasse_threshold": r.impasse_threshold}})
        try:
            self.result = self.scheduler.run()
            self.status = "finished"
            self.hub.emit({"type": "session_finished", "result": self.result.to_dict(),
                           "tokens": self.ledger.totals(), "format": self.ledger.format_stats(),
                           "tokens_by_agent": self.ledger.per_agent()})
        except Exception as e:  # noqa: BLE001
            self.status = "error"
            self.hub.emit({"type": "session_error", "error": repr(e)})

    def start(self) -> "Session":
        self._thread.start()
        return self

    def join(self, timeout: Optional[float] = None) -> None:
        self._thread.join(timeout)

    def summary(self) -> dict:
        return {"session_id": self.id, "status": self.status, "problem": self.board.problem,
                "events": len(self.hub.events()),
                "result": self.result.to_dict() if self.result else None}
