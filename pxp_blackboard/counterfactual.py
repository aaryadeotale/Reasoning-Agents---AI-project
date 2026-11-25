"""Milestone 3: retrospective counterfactual sandbox (credit assignment on deadlock).

When the live board deadlocks, a counterfactual-enabled agent asks:
  "If I had posted something different at step k, would the panel be closer to consensus now?"

Procedure (the live board is never touched):
  1. Rollback points: the agent's own most recent past messages.
  2. For each point, rebuild an isolated Blackboard from the log *before* that message
     (replaying through post(), so every step is re-validated).
  3. Ask the agent for alternative moves at that point (`propose_alternatives`):
     a different explanation, prediction or tag.
  4. Roll forward: substitute the alternative, then let sandbox copies (`fork()`) of all
     agents respond, using the same scheduling policy, for at most `max_steps` messages.
  5. Score each alternative timeline by its distance to consensus, minus a token penalty.
  6. Apply the delta: the best alternative becomes a legal move on the *current* live board
     (usually a REVISE carrying the improved explanation), and the agent keeps the lesson.

Agents opt in by providing:
  counterfactual: bool                       is this agent allowed to reflect?
  propose_alternatives(snap, own_msg, objections, k) -> (list[dict], tokens_used)
  fork() -> Agent                            an independent copy for sandbox use (optional)
  learn(alternative: dict) -> None           keep the lesson (optional)
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Optional

from .blackboard import Blackboard
from .models import NEGATIVE_TAGS, Draft, Tag
from .validator import ProtocolError

NEG = {t.value for t in NEGATIVE_TAGS}


@dataclass
class Timeline:
    rollback_seq: int                  # the agent's past message that was replaced
    alternative: dict                  # {"tag", "prediction", "explanation", ...}
    steps: list                        # message dicts posted inside the sandbox
    outcome: str                       # consensus | impasse | open | invalid
    score: float
    tokens: int

    def to_dict(self) -> dict:
        return {"rollback_seq": self.rollback_seq, "alternative": self.alternative, "steps": self.steps,
                "outcome": self.outcome, "score": round(self.score, 3), "tokens": self.tokens}


@dataclass
class Reflection:
    agent_id: str
    baseline_score: float
    timelines: list = field(default_factory=list)
    best: Optional[Timeline] = None
    delta: Optional[Draft] = None      # move to post on the live board (None = nothing better found)
    tokens: int = 0                    # all sandbox tokens (alternatives + roll-forwards)

    def to_dict(self) -> dict:
        return {"agent_id": self.agent_id, "baseline_score": round(self.baseline_score, 3),
                "timelines": [t.to_dict() for t in self.timelines],
                "best": self.best.to_dict() if self.best else None,
                "delta": ({"tag": self.delta.tag.value, "prediction": self.delta.prediction,
                           "explanation": self.delta.explanation} if self.delta else None),
                "tokens": self.tokens}


def _draft_tokens(d: Optional[Draft]) -> int:
    if d is None:
        return 0
    return int(d.meta.get("prompt_tokens", 0)) + int(d.meta.get("completion_tokens", 0))


def consensus_distance(snap: dict) -> float:
    """0 at consensus; otherwise the number of non-author agents not ratifying the head,
    plus half the negative streak (so a calmer board scores better)."""
    head = snap["head"]
    if head is None:
        return float(len(snap["agents"]))
    missing = sum(1 for a in snap["agents"]
                  if a != head["agent_id"] and snap["stances"].get(a) != Tag.RATIFY.value)
    return missing + 0.5 * snap["negative_streak"]


class CounterfactualSandbox:
    def __init__(self, max_points: int = 2, max_alternatives: int = 4, max_steps: int = 8,
                 token_penalty: float = 0.5, impasse_threshold: int = 4, seed: int = 0):
        self.max_points = max_points              # how many own past messages to roll back to
        self.max_alternatives = max_alternatives  # alternatives tried per rollback point
        self.max_steps = max_steps                # sandbox roll-forward budget (messages)
        self.token_penalty = token_penalty        # score penalty per 1k sandbox tokens
        self.impasse_threshold = impasse_threshold
        self.rng = random.Random(seed)            # for imperfect peer models

    # ---- helpers ----------------------------------------------------------
    @staticmethod
    def _replay(snapshot: dict, messages: list) -> Blackboard:
        bb = Blackboard(snapshot["problem"], snapshot["agents"], snapshot["max_messages"])
        for m in messages:
            bb.post(Draft(m["agent_id"], Tag(m["tag"]), m["target_id"], m["prediction"],
                          m["explanation"], m.get("meta", {})), _timestamp=m["timestamp"])
        return bb

    def _score(self, snap: dict, tokens: int, alt: dict) -> float:
        base = 10.0 if _is_consensus(snap) else -consensus_distance(snap)
        return base - self.token_penalty * tokens / 1000.0 - float(alt.get("cost", 0.0))

    def _roll_forward(self, bb: Blackboard, agents: dict, policy: Callable) -> tuple[list, int, str]:
        steps, tokens, declined = [], 0, set()
        for _ in range(self.max_steps):
            snap = bb.snapshot()
            if _is_consensus(snap):
                return steps, tokens, "consensus"
            if snap["negative_streak"] >= self.impasse_threshold:
                return steps, tokens, "impasse"
            if bb.is_full():
                return steps, tokens, "open"
            scored = sorted((-s, aid) for aid, a in agents.items()
                            if (aid, snap["version"]) not in declined
                            and (s := policy(a, snap)) is not None)
            if not scored:
                return steps, tokens, "open"
            aid = scored[0][1]
            try:
                draft = agents[aid].act(snap)
            except Exception:  # noqa: BLE001 - sandbox must never crash the live run
                draft = None
            tokens += _draft_tokens(draft)
            if draft is None:
                declined.add((aid, snap["version"]))
                continue
            try:
                steps.append(bb.post(draft).to_dict())
            except ProtocolError:
                declined.add((aid, snap["version"]))
        snap = bb.snapshot()
        return steps, tokens, "consensus" if _is_consensus(snap) else "open"

    # ---- main entry point -------------------------------------------------
    def reflect(self, snapshot: dict, agent, agents: dict, policy: Callable) -> Reflection:
        aid = agent.agent_id
        msgs = snapshot["messages"]
        objections = [m for m in msgs if m["tag"] in NEG]
        own = [m for m in msgs if m["agent_id"] == aid][-self.max_points:]
        ref = Reflection(aid, baseline_score=-consensus_distance(snapshot))
        if not hasattr(agent, "propose_alternatives"):
            return ref
        for point in reversed(own):  # most recent first
            prefix = msgs[:point["seq"] - 1]
            try:
                base = self._replay(snapshot, prefix)
            except ProtocolError:
                continue
            base_snap = base.snapshot()
            alts, t = agent.propose_alternatives(base_snap, point, objections, self.max_alternatives)
            ref.tokens += int(t)
            for alt in alts[: self.max_alternatives]:
                bb = self._replay(snapshot, prefix)
                head = bb.head
                tag = Tag(alt["tag"])
                draft = Draft(aid, tag, None if tag is Tag.PROPOSE else (head.seq if head else None),
                              alt.get("prediction"), alt.get("explanation"),
                              {**alt.get("meta", {}), "counterfactual": True})
                try:
                    first = bb.post(draft).to_dict()
                except ProtocolError:
                    continue
                sandbox_agents = {k: self._model_of(k, a, aid) for k, a in agents.items()}
                if hasattr(sandbox_agents[aid], "learn"):
                    sandbox_agents[aid].learn(alt)
                steps, tok, outcome = self._roll_forward(bb, sandbox_agents, policy)
                ref.tokens += tok
                tl = Timeline(point["seq"], alt, [first] + steps, outcome,
                              self._score(bb.snapshot(), tok, alt), tok)
                ref.timelines.append(tl)
        if ref.timelines:
            ref.best = max(ref.timelines, key=lambda x: x.score)
            if ref.best.score > ref.baseline_score:
                ref.delta = self._live_move(snapshot, aid, ref.best.alternative)
        return ref

    def _model_of(self, k: str, a, reflecting_id: str):
        """Sandbox stand-in for agent k. Peers may expose `peer_model(rng)`, an imperfect model of
        their hidden state; otherwise a plain `fork()` (an LLM agent re-queried with its persona)."""
        if k != reflecting_id and hasattr(a, "peer_model"):
            return a.peer_model(self.rng)
        return a.fork() if hasattr(a, "fork") else a

    @staticmethod
    def _live_move(snapshot: dict, aid: str, alt: dict) -> Optional[Draft]:
        """Translate the best past alternative into a legal move on the current board."""
        head = snapshot["head"]
        if head is None:
            return None
        meta = {**alt.get("meta", {}), "counterfactual": True}
        if alt.get("prediction") and alt.get("explanation"):
            if alt["prediction"] == head["prediction"] and alt["explanation"] == head["explanation"]:
                return None if aid == head["agent_id"] else Draft(aid, Tag.RATIFY, head["seq"], meta=meta)
            return Draft(aid, Tag.REVISE, head["seq"], alt["prediction"], alt["explanation"], meta)
        if alt.get("tag") == Tag.RATIFY.value and aid != head["agent_id"]:
            return Draft(aid, Tag.RATIFY, head["seq"], meta=meta)
        return None


def _is_consensus(snap: dict) -> bool:
    head = snap["head"]
    return head is not None and all(a == head["agent_id"] or snap["stances"].get(a) == Tag.RATIFY.value
                                    for a in snap["agents"])
