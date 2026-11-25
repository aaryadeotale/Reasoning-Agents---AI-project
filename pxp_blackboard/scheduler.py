"""Event-driven priority scheduler and terminal-condition detector."""
from __future__ import annotations

import copy
import queue
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Callable, Iterable, Optional

from .agents import Agent
from .blackboard import Blackboard
from .models import NEGATIVE_TAGS, Message, Tag
from .validator import BoardClosedError, ProtocolError, StaleTargetError

AUTHOR_BOOST = 10.0  # the head's author gets first shot at answering objections


class Outcome(str, Enum):
    CONSENSUS = "consensus"
    IMPASSE = "impasse"                    # negative-tag streak hit the threshold
    MAX_ITERATIONS = "max_iterations"
    NO_ELIGIBLE_AGENTS = "no_eligible_agents"


Policy = Callable[[Agent, dict], Optional[float]]


def default_policy(agent: Agent, snap: dict) -> Optional[float]:
    """Score an agent for the current board; None = nothing for it to do."""
    head = snap["head"]
    if head is None:
        return agent.priority
    aid = agent.agent_id
    last = max((m["seq"] for m in snap["messages"] if m["agent_id"] == aid), default=0)
    if aid == head["agent_id"]:  # author: only acts to answer unanswered objections
        pending = any(m["seq"] > last and m["tag"] in {t.value for t in NEGATIVE_TAGS}
                      for m in snap["messages"])
        return agent.priority + AUTHOR_BOOST if pending else None
    return agent.priority if last < head["seq"] else None  # hasn't seen this head yet


def classify_intelligibility(messages: Iterable[Message], outcome: Outcome) -> str:
    """Simplified: Ultra-Strong = consensus reached AND a REVISE followed a
    REFUTE/REJECT (a productive revise loop); Strong = consensus otherwise."""
    if outcome is not Outcome.CONSENSUS:
        return "Failed"
    seen_negative = False
    for m in messages:
        if m.tag in NEGATIVE_TAGS:
            seen_negative = True
        elif m.tag is Tag.REVISE and seen_negative:
            return "Ultra-Strong"
    return "Strong"


@dataclass
class SchedulerConfig:
    max_concurrent: int = 2       # concurrent agent calls (LLM inference slots)
    impasse_threshold: int = 4    # negative streak that declares an impasse
    max_protocol_errors: int = 3  # strikes before an agent is benched
    max_reflections: int = 1      # counterfactual reflections allowed per CF agent per session
    loop_revisions: int = 3       # REVISEs without consensus that also count as a deadlock loop


@dataclass
class RunResult:
    outcome: Outcome
    iterations: int
    tag_counts: dict
    intelligibility: str
    final_prediction: Optional[str]
    protocol_errors: list
    reflections: int = 0          # counterfactual reflections run (Milestone 3)
    deltas_applied: int = 0       # reflections that changed the live board
    sandbox_tokens: int = 0       # tokens spent inside counterfactual sandboxes

    def to_dict(self) -> dict:
        d = asdict(self)
        d["outcome"] = self.outcome.value
        return d


class Scheduler:
    """Reacts to board events; picks the highest-priority eligible agents,
    caps concurrency, and stops on consensus / impasse / limits."""

    def __init__(self, board: Blackboard, agents: Iterable[Agent],
                 config: Optional[SchedulerConfig] = None, policy: Policy = default_policy,
                 listener: Optional[Callable[[dict], None]] = None, sandbox=None):
        self.board = board
        self.agents = {a.agent_id: a for a in agents}
        if set(self.agents) != set(board.agents):
            raise ValueError("scheduler agents must match the board's registered agents")
        self.cfg = config or SchedulerConfig()
        self.policy = policy
        self._listener = listener  # receives agent_started / agent_finished dicts (UI hook)
        self._events: queue.Queue = queue.Queue()
        self._in_flight: dict = {}
        self._declined: set = set()   # (agent_id, board version) the agent passed on
        self._strikes: Counter = Counter()
        self._benched: set = set()
        self.errors: list = []
        # ---- Milestone 3: counterfactual reflection on deadlock ----
        self.sandbox = sandbox
        self._cf_budget = {aid: self.cfg.max_reflections for aid, a in self.agents.items()
                           if getattr(a, "counterfactual", False)} if sandbox else {}
        self._impasse_at = self.cfg.impasse_threshold  # moves up after an applied delta (grace)
        self.reflections: list = []
        self._deltas = 0
        self._rev_mark = 0            # REVISE count at the last reflection
        board.subscribe(lambda m: self._events.put(("message", m)))

    def _emit(self, **event) -> None:
        if self._listener:
            try:
                self._listener(event)
            except Exception:  # noqa: BLE001 - a broken UI must never stop the run
                pass

    # ---- worker (runs in pool thread; never raises) -------------------
    def _work(self, aid: str, snap: dict):
        try:
            draft = self.agents[aid].act(copy.deepcopy(snap))
        except Exception as e:  # noqa: BLE001 - agent code is untrusted
            return ("error", f"{aid}: agent raised {e!r}")
        if draft is None:
            return ("declined", snap["version"])
        try:
            self.board.post(draft)
            return ("posted", None)
        except StaleTargetError:
            return ("stale", snap["version"])
        except BoardClosedError:
            return ("closed", snap["version"])
        except ProtocolError as e:
            return ("error", f"{aid}: {e}")

    # ---- control loop -------------------------------------------------
    def _terminal(self) -> Optional[Outcome]:
        if self.board.is_consensus():
            return Outcome.CONSENSUS
        if self.board.negative_streak >= self.cfg.impasse_threshold:
            return Outcome.IMPASSE
        if self.board.is_full():
            return Outcome.MAX_ITERATIONS
        return None

    def _dispatch(self, pool: ThreadPoolExecutor) -> None:
        snap = self.board.snapshot()
        slots = self.cfg.max_concurrent - len(self._in_flight)
        if snap["head"] is None:  # exactly one agent may open the board
            slots = min(slots, 1 - len(self._in_flight))
        if slots <= 0:
            return
        scored = []
        for aid, agent in self.agents.items():
            if aid in self._in_flight or aid in self._benched:
                continue
            if (aid, snap["version"]) in self._declined:
                continue
            score = self.policy(agent, snap)
            if score is not None:
                scored.append((-score, aid))
        scored.sort()  # highest score first; agent_id breaks ties deterministically
        for _, aid in scored[:slots]:
            fut = pool.submit(self._work, aid, snap)
            self._in_flight[aid] = fut
            self._emit(type="agent_started", agent_id=aid, version=snap["version"])
            fut.add_done_callback(lambda f, aid=aid: self._events.put(("done", aid, f.result())))

    def _handle(self, event) -> None:
        if event[0] != "done":
            return  # "message" events just wake the loop to re-evaluate
        _, aid, (kind, payload) = event
        self._in_flight.pop(aid, None)
        self._emit(type="agent_finished", agent_id=aid, kind=kind)
        if kind in ("declined", "stale", "closed"):
            # don't re-ask this agent until the board changes (prevents busy-loops)
            self._declined.add((aid, payload))
        elif kind == "error":
            self.errors.append(payload)
            self._strikes[aid] += 1
            if self._strikes[aid] >= self.cfg.max_protocol_errors:
                self._benched.add(aid)

    def _revise_count(self) -> int:
        return sum(1 for m in self.board.history() if m.tag is Tag.REVISE)

    def _can_reflect(self) -> bool:
        return any(n > 0 for n in self._cf_budget.values())

    def _reflect(self) -> bool:
        """Run counterfactual reflection for each CF agent with budget (highest priority first).
        Returns True as soon as one reflection changes the live board."""
        order = sorted(self._cf_budget, key=lambda a: (-self.agents[a].priority, a))
        for aid in order:
            if self._cf_budget[aid] <= 0 or aid in self._benched:
                continue
            self._cf_budget[aid] -= 1
            snap = self.board.snapshot()
            self._emit(type="counterfactual_started", agent_id=aid, version=snap["version"],
                       negative_streak=snap["negative_streak"])
            try:
                ref = self.sandbox.reflect(snap, self.agents[aid], self.agents, self.policy)
            except Exception as e:  # noqa: BLE001 - reflection is best-effort
                self.errors.append(f"{aid}: counterfactual reflection failed: {e!r}")
                continue
            self.reflections.append(ref)
            applied = False
            if ref.delta is not None:
                try:
                    self.board.post(ref.delta)
                    applied = True
                except ProtocolError as e:
                    self.errors.append(f"{aid}: counterfactual delta rejected: {e}")
            if applied:
                if hasattr(self.agents[aid], "learn"):
                    self.agents[aid].learn(ref.best.alternative)
                self._deltas += 1
                # grace: every other agent gets one chance to answer the new move
                self._impasse_at = self.board.negative_streak + len(self.agents) - 1
            self._emit(type="counterfactual_finished", agent_id=aid, applied=applied, reflection=ref.to_dict())
            if applied:
                return True
        return False

    def run(self) -> RunResult:
        with ThreadPoolExecutor(max_workers=self.cfg.max_concurrent) as pool:
            while True:
                if self.board.is_consensus():
                    outcome = Outcome.CONSENSUS
                    break
                deadlocked = self.board.negative_streak >= max(self._impasse_at, self.cfg.impasse_threshold)
                revs = self._revise_count()
                looping = revs - self._rev_mark >= self.cfg.loop_revisions
                if (deadlocked or looping) and self._can_reflect():
                    if self._in_flight:  # drain in-flight work before reflecting
                        self._handle(self._events.get())
                        continue
                    self._rev_mark = revs
                    if self._reflect():
                        continue
                if deadlocked:
                    outcome = Outcome.IMPASSE
                    break
                if self.board.is_full():
                    outcome = Outcome.MAX_ITERATIONS
                    break
                self._dispatch(pool)
                if not self._in_flight:
                    outcome = Outcome.NO_ELIGIBLE_AGENTS
                    break
                self._handle(self._events.get())
            self.board.close()  # late in-flight posts now fail cleanly
        hist = self.board.history()
        head = self.board.head
        return RunResult(
            outcome=outcome, iterations=len(hist),
            tag_counts=dict(Counter(m.tag.value for m in hist)),
            intelligibility=classify_intelligibility(hist, outcome),
            final_prediction=head.prediction if (head and outcome is Outcome.CONSENSUS) else None,
            protocol_errors=list(self.errors),
            reflections=len(self.reflections), deltas_applied=self._deltas,
            sandbox_tokens=sum(r.tokens for r in self.reflections),
        )
