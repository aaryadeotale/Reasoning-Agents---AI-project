"""Event-driven priority scheduler for Milestone 1."""
from __future__ import annotations

import copy
import queue
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Callable, Iterable, Optional

from agents import Agent
from blackboard import Blackboard
from models import NEGATIVE_TAGS, Message, Tag
from validator import BoardClosedError, ProtocolError, StaleTargetError

AUTHOR_BOOST = 10.0


class Outcome(str, Enum):
    CONSENSUS = "consensus"
    IMPASSE = "impasse"
    MAX_ITERATIONS = "max_iterations"
    NO_ELIGIBLE_AGENTS = "no_eligible_agents"

Policy = Callable[[Agent, dict], Optional[float]]


def default_policy(agent: Agent, snap: dict) -> Optional[float]:
    head = snap["head"]
    if head is None:
        return None

    aid = agent.agent_id
    last = max(
        (m["seq"] for m in snap["messages"] if m["agent_id"] == aid),
        default=0,
    )

    # The current stance is the scheduler's source of truth. A non-author
    # acts once per hypothesis; after a REVISE the stance map is reset, so the
    # same agent becomes eligible for the new head.
    stance = snap["stances"].get(aid)

    if aid == head["agent_id"]:
        pending = any(
            m["seq"] > last and m["tag"] in {t.value for t in NEGATIVE_TAGS}
            for m in snap["messages"]
        )
        return agent.priority + AUTHOR_BOOST if pending else None

    return agent.priority if stance != Tag.RATIFY.value else None


def classify_intelligibility(messages: Iterable[Message], outcome: Outcome) -> str:
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
    max_concurrent: int = 2
    impasse_threshold: int = 4
    max_protocol_errors: int = 3


@dataclass
class RunResult:
    outcome: Outcome
    iterations: int
    tag_counts: dict
    intelligibility: str
    final_prediction: Optional[str]
    protocol_errors: list

    def to_dict(self) -> dict:
        d = asdict(self)
        d["outcome"] = self.outcome.value
        return d


class Scheduler:
    def __init__(
        self,
        board: Blackboard,
        agents: Iterable[Agent],
        config: Optional[SchedulerConfig] = None,
        policy: Policy = default_policy,
    ):
        self.board = board
        self.agents = {a.agent_id: a for a in agents}
        if set(self.agents) != set(board.agents):
            raise ValueError("scheduler agents must match the board's registered agents")
        self.cfg = config or SchedulerConfig()
        self.policy = policy
        self._events: queue.Queue = queue.Queue()
        self._in_flight: dict[str, object] = {}
        self._declined: set[tuple[str, int]] = set()
        self._strikes: Counter = Counter()
        self._benched: set[str] = set()
        self.errors: list[str] = []

    def _work(self, aid: str, snap: dict):
        try:
            draft = self.agents[aid].act(copy.deepcopy(snap))
        except Exception as e:
            return "error", f"{aid}: agent raised {e!r}"
        if draft is None:
            return "declined", snap["version"]
        try:
            self.board.post(draft)
            return "posted", None
        except StaleTargetError:
            return "stale", snap["version"]
        except BoardClosedError:
            return "closed", snap["version"]
        except ProtocolError as e:
            return "error", f"{aid}: {e}"

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
        scored.sort()

        for _, aid in scored[:slots]:
            fut = pool.submit(self._work, aid, snap)
            self._in_flight[aid] = fut
            fut.add_done_callback(
                lambda f, aid=aid: self._events.put(("done", aid, f.result()))
            )

    def _handle(self, event) -> None:
        _, aid, (kind, payload) = event
        self._in_flight.pop(aid, None)
        if kind in ("declined", "stale", "closed"):
            self._declined.add((aid, payload))
        elif kind == "error":
            self.errors.append(payload)
            self._strikes[aid] += 1
            if self._strikes[aid] >= self.cfg.max_protocol_errors:
                self._benched.add(aid)

    def run(self) -> RunResult:
        with ThreadPoolExecutor(max_workers=self.cfg.max_concurrent) as pool:
            while True:
                outcome = self._terminal()
                if outcome:
                    break
                self._dispatch(pool)
                if not self._in_flight:
                    outcome = Outcome.NO_ELIGIBLE_AGENTS
                    break
                self._handle(self._events.get())

        self.board.close()
        hist = self.board.history()
        head = self.board.head
        return RunResult(
            outcome=outcome,
            iterations=len(hist),
            tag_counts=dict(Counter(m.tag.value for m in hist)),
            intelligibility=classify_intelligibility(hist, outcome),
            final_prediction=head.prediction if outcome is Outcome.CONSENSUS else None,
            protocol_errors=list(self.errors),
        )
