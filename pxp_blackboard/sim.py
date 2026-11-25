"""Simulated PEX agents for the Milestone 4 ablation (no LLM needed).

Each task has K candidate answers (one gold) and M evidence *aspects*. An explanation can
cover at most L aspects (the persona's word limit). Every agent has hidden state:
  belief      the answer it thinks is right (correct with probability `competence`)
  confidence  calibrated: higher on average when the belief is correct
  concerns    the aspects it needs to see addressed before it will RATIFY
Agents speak the real PXP protocol through the real Blackboard, validator and Scheduler.
Prediction/explanation are human-readable strings; the structured content (aspects,
confidence) travels in `meta` so that other simulated agents can "read" it.

Standard (myopic) authors answer only the *latest* objection and keep their own pet
aspects pinned, which reproduces the revise-loop / deadlock failure seen with real 7B models.
Counterfactual agents can additionally `propose_alternatives` for past messages and adopt
the best one through the sandbox in `counterfactual.py`.
"""
from __future__ import annotations

import copy
import itertools
import random
from dataclasses import dataclass, field
from typing import Optional

from .models import Draft, Tag

DOMAINS = {
    "automotive": ("Which root cause explains the brake-system recall?",
                   ["pedal-force data", "supplier tolerance", "thermal fade", "ECU logs", "crash-test results",
                    "warranty claims"],
                   ["worn caliper seals", "faulty ECU firmware", "out-of-spec rotor alloy", "hydraulic line leak"]),
    "pharmaceutical": ("Which formulation change explains the failed stability batch?",
                       ["dissolution profile", "excipient interaction", "humidity exposure", "API polymorph",
                        "assay variance", "packaging permeability"],
                       ["moisture ingress", "polymorph conversion", "excipient incompatibility", "assay error"]),
    "energy": ("What is the main driver of the grid frequency excursion?",
               ["load forecast error", "inertia estimate", "renewable ramp rate", "reserve margin",
                "protection settings", "interconnector flow"],
               ["solar ramp-down", "generator trip", "interconnector loss", "under-forecast demand"]),
    "clinical": ("What is the most likely diagnosis for this presentation?",
                 ["ECG findings", "troponin trend", "medication history", "imaging", "vital signs",
                  "risk factors"],
                 ["acute MI", "aortic dissection", "pulmonary embolism", "pericarditis"]),
}
PERSONAS = {"automotive": ("automotive_engineer", "skeptic", "generalist"),
            "pharmaceutical": ("pharmacologist", "skeptic", "data_analyst"),
            "energy": ("energy_analyst", "skeptic", "data_analyst"),
            "clinical": ("clinician", "skeptic", "pharmacologist")}


@dataclass
class Task:
    id: str
    domain: str
    problem: str
    aspects: list
    answers: list
    gold: int
    capacity: int = 3          # max aspects one explanation can cover (word limit)


def make_task(rng: random.Random, idx: int, domain: Optional[str] = None) -> Task:
    domain = domain or rng.choice(sorted(DOMAINS))
    q, aspects, answers = DOMAINS[domain]
    return Task(f"{domain}-{idx}", domain, q, list(aspects), list(answers), rng.randrange(len(answers)))


@dataclass
class SimProfile:
    belief: int
    confidence: float
    concerns: list               # aspect indices this agent needs addressed
    disposition: str             # author | critical (REJECT) | cautious (REFUTE)
    pinned: list = field(default_factory=list)   # aspects the agent always keeps when authoring
    pets: list = field(default_factory=list)     # preferred extra aspects (dropped first when space runs out)


def _tokens(snapshot: dict, completion: int, base: int = 320) -> dict:
    ctx = min(len(snapshot["messages"]), 8)
    return {"prompt_tokens": base + 45 * ctx + 30, "completion_tokens": completion}


class SimAgent:
    """A simulated PEX agent. Deterministic given its profile; forks may be imperfect copies."""

    def __init__(self, agent_id: str, task: Task, profile: SimProfile, *, priority: float = 1.0,
                 counterfactual: bool = False, persona: str = "generalist", fidelity: float = 0.8,
                 belief_cost: float = 4.0, seed: int = 0):
        self.agent_id, self.task, self.p = agent_id, task, profile
        self.priority, self.counterfactual, self.persona = priority, counterfactual, persona
        self.fidelity, self.belief_cost = fidelity, belief_cost
        self.rng = random.Random(seed)
        self._fifo: list = []        # aspects added by myopic revisions (oldest first)
        self._rebuttals = itertools.count(1)
        self.peers: dict = {}        # agent_id -> SimAgent (set by make_panel; used only by fork)

    # ---- text helpers --------------------------------------------------
    def _pred(self, i: int) -> str:
        return self.task.answers[i]

    def _expl(self, pred: int, aspects: list, suffix: str = "") -> str:
        names = ", ".join(self.task.aspects[a] for a in aspects) or "the overall picture"
        return f"{self._pred(pred)} is supported by {names}{suffix}"

    def _hyp(self, tag: Tag, target, pred: int, aspects: list, snapshot: dict, suffix: str = "") -> Draft:
        meta = {"aspects": list(aspects), "conf": round(self.p.confidence, 3), "answer": pred,
                **_tokens(snapshot, 90)}
        return Draft(self.agent_id, tag, target, self._pred(pred), self._expl(pred, aspects, suffix), meta)

    def _own_explanation(self) -> list:
        L = self.task.capacity
        out = list(dict.fromkeys(self.p.pinned + self.p.concerns + self.p.pets))
        for a in self._fifo:
            if a not in out:
                out.append(a)
        while len(out) > L:  # drop the oldest non-pinned aspect
            for i, a in enumerate(out):
                if a not in self.p.pinned:
                    out.pop(i)
                    break
            else:
                out.pop()
        return out

    # ---- policy ----------------------------------------------------------
    def act(self, snap: dict) -> Optional[Draft]:
        head = snap["head"]
        if head is None:
            return self._hyp(Tag.PROPOSE, None, self.p.belief, self._own_explanation(), snap)
        if head["agent_id"] == self.agent_id:
            return self._answer_objection(snap)
        return self._review(snap)

    def _review(self, snap: dict) -> Draft:
        head = snap["head"]
        hm = head.get("meta", {})
        pred, aspects = hm.get("answer"), set(hm.get("aspects", []))
        missing = [c for c in self.p.concerns if c not in aspects]
        if pred != self.p.belief:
            if not missing and hm.get("conf", 0) > self.p.confidence:  # persuaded
                self.p.belief = pred
            else:
                a = missing[0] if missing else None
                why = (f"The evidence points to {self._pred(self.p.belief)}"
                       + (f"; {self.task.aspects[a]} is not addressed" if a is not None else ""))
                return Draft(self.agent_id, Tag.REJECT, head["seq"], self._pred(self.p.belief), why,
                             {"aspect": a, "conf": round(self.p.confidence, 3), "answer": self.p.belief,
                              **_tokens(snap, 60)})
        if missing:
            a = missing[0]
            why = f"The explanation does not address {self.task.aspects[a]}"
            tag = Tag.REJECT if self.p.disposition == "critical" else Tag.REFUTE
            return Draft(self.agent_id, tag, head["seq"], None, why,
                         {"aspect": a, "conf": round(self.p.confidence, 3), **_tokens(snap, 50)})
        return Draft(self.agent_id, Tag.RATIFY, head["seq"], None, None, _tokens(snap, 15))

    def _answer_objection(self, snap: dict) -> Optional[Draft]:
        """Myopic author: react to the latest objection only (no long-horizon credit assignment)."""
        head = snap["head"]
        mine = max(m["seq"] for m in snap["messages"] if m["agent_id"] == self.agent_id)
        objs = [m for m in snap["messages"] if m["seq"] > mine and m["tag"] in ("REJECT", "REFUTE")]
        if not objs:
            return None
        last = objs[-1]["meta"]
        pred = head["meta"].get("answer", self.p.belief)
        if last.get("answer") is not None and last["answer"] != pred and last.get("conf", 0) > self.p.confidence:
            pred = last["answer"]  # defer to a more confident objector
            self.p.belief, self.p.confidence = pred, last["conf"] * 0.95
        if last.get("aspect") is not None and last["aspect"] not in self._own_explanation():
            self._fifo.append(last["aspect"])
        aspects = self._own_explanation()
        suffix = ""
        if pred == head["meta"].get("answer") and aspects == head["meta"].get("aspects"):
            suffix = f" (restated, rebuttal {next(self._rebuttals)})"  # nothing new: argue again
        return self._hyp(Tag.REVISE, head["seq"], pred, aspects, snap, suffix)

    # ---- Milestone 3 hooks ----------------------------------------------
    def fork(self) -> "SimAgent":
        """Sandbox copy. Copies of *other* agents are imperfect models: each hidden concern is
        kept with probability `fidelity` (otherwise replaced by a random aspect)."""
        twin = copy.copy(self)
        twin.p = copy.deepcopy(self.p)
        twin._fifo = list(self._fifo)
        twin._rebuttals = itertools.count(1000)
        twin.rng = random.Random(self.rng.random())
        return twin

    def peer_model(self, rng: random.Random) -> "SimAgent":
        twin = self.fork()
        twin.p.concerns = [c if rng.random() < self.fidelity else rng.randrange(len(self.task.aspects))
                           for c in twin.p.concerns]
        return twin

    def learn(self, alt: dict) -> None:
        meta = alt.get("meta", {})
        if "aspects" in meta:
            self.p.pinned = list(meta["aspects"])
            self.p.pets, self._fifo = [], []
        if "answer" in meta:
            self.p.belief = meta["answer"]

    def propose_alternatives(self, snap: dict, own_msg: dict, objections: list, k: int):
        """Alternatives for one of our past messages, using everything raised in the whole debate."""
        L = self.task.capacity
        raised = [m["meta"].get("aspect") for m in objections if m["meta"].get("aspect") is not None]
        freq = sorted(set(raised), key=lambda a: (-raised.count(a), raised.index(a)))
        counter = sorted({(m["meta"]["answer"], m["meta"].get("conf", 0)) for m in objections
                          if m["meta"].get("answer") is not None}, key=lambda x: -x[1])
        cover = (freq + [a for a in self.p.concerns if a not in freq])[:L]
        preds = [self.p.belief] + [a for a, _ in counter if a != self.p.belief][:1]
        tag = "PROPOSE" if snap["head"] is None else "REVISE"
        alts = []
        for pi, pred in enumerate(preds):
            cost = 0.0 if pi == 0 else self.belief_cost * self.p.confidence
            for asp in (cover, self._own_explanation()):
                alt = {"tag": tag, "prediction": self._pred(pred), "explanation": self._expl(pred, asp, " (reconsidered)"),
                       "cost": cost, "meta": {"aspects": list(asp), "conf": round(self.p.confidence, 3), "answer": pred,
                                             **_tokens(snap, 90)}}
                if alt not in alts:
                    alts.append(alt)
        if snap["head"] is not None and snap["head"]["agent_id"] != self.agent_id:  # concede
            hp = snap["head"]["meta"].get("answer")
            alts.append({"tag": "RATIFY", "prediction": None, "explanation": None,
                         "cost": self.belief_cost * self.p.confidence if hp != self.p.belief else 1.0,
                         "meta": {"answer": hp}})
        ctx = min(len(snap["messages"]), 8)
        tokens = 600 + 45 * ctx + 60 * min(k, len(alts))
        return alts[:k], tokens


def make_panel(task: Task, rng: random.Random, *, cf_mask: list, competence: float = 0.7,
               fidelity: float = 0.8, seed: int = 0) -> list:
    """Author + critical reviewer + cautious reviewer, with hidden beliefs and concerns."""
    roles = [("Author", "author", 5.0), ("Critic", "critical", 1.0), ("Reviewer", "cautious", 1.0)]
    personas = PERSONAS[task.domain]
    agents = []
    for i, (aid, disp, prio) in enumerate(roles):
        correct = rng.random() < competence
        belief = task.gold if correct else rng.choice([j for j in range(len(task.answers)) if j != task.gold])
        conf = rng.uniform(0.55, 0.95) if correct else rng.uniform(0.35, 0.8)
        n = 1 if disp == "author" else rng.choice([1, 2])
        concerns = rng.sample(range(len(task.aspects)), n)
        pinned = list(concerns) if disp == "author" else []
        pets = rng.sample([a for a in range(len(task.aspects)) if a not in concerns], task.capacity - n)
        agents.append(SimAgent(aid, task, SimProfile(belief, conf, concerns, disp, pinned, pets), priority=prio,
                               counterfactual=bool(cf_mask[i]), persona=personas[i], fidelity=fidelity,
                               seed=seed * 7 + i))
    return agents
