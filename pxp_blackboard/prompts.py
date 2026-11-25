"""Prompt matrix library + prompt builder.

A *persona* (prompt_library/*.json) says who the agent is and sets output constraints.
The builder turns (persona, board snapshot) into chat messages that tell the model
exactly which PXP tags are legal right now and what JSON to return.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .models import Tag

LIB_DIR = Path(__file__).parent / "prompt_library"


@dataclass(frozen=True)
class Persona:
    id: str
    title: str
    domain: str
    system: str
    disposition: str = "cooperative"
    explanation_min_words: int = 5
    explanation_max_words: int = 120
    max_prediction_chars: int = 300


class PromptLibrary:
    def __init__(self, personas: list[Persona]):
        self._p = {p.id: p for p in personas}

    @classmethod
    def load(cls, directory: Path = LIB_DIR) -> "PromptLibrary":
        out = []
        for f in sorted(Path(directory).glob("*.json")):
            try:
                out.append(Persona(**json.loads(f.read_text())))
            except (TypeError, ValueError) as e:
                raise ValueError(f"bad persona file {f.name}: {e}") from e
        if not out:
            raise ValueError(f"no personas found in {directory}")
        return cls(out)

    def get(self, persona_id: str) -> Persona:
        if persona_id not in self._p:
            raise KeyError(f"unknown persona {persona_id!r}; available: {sorted(self._p)}")
        return self._p[persona_id]

    def ids(self) -> list[str]:
        return sorted(self._p)


PRIMER = """You take part in a structured expert panel that works on a shared blackboard.
Every message you write has a TAG, a PREDICTION (your answer: the "what") and an EXPLANATION (your reasoning: the "why").
Tags:
- PROPOSE: put the first answer on an empty board.
- RATIFY: you agree with the current answer AND its explanation.
- REVISE: you improve the answer or its explanation; give the full new version.
- REFUTE: you disagree but cannot fix it yourself; explain the roadblock.
- REJECT: you disagree with both the answer and its explanation; explain why.
Reply with ONE JSON object and nothing else (no markdown, no commentary)."""

TAG_RULES = {
    Tag.PROPOSE: "PROPOSE: prediction = your answer, explanation = why.",
    Tag.RATIFY: "RATIFY: prediction = null, explanation = null.",
    Tag.REVISE: "REVISE: prediction = full new answer, explanation = full new reasoning (must differ from the current ones).",
    Tag.REFUTE: "REFUTE: prediction = null, explanation = the roadblock that stops you.",
    Tag.REJECT: "REJECT: explanation = why it is wrong; prediction = your counter-answer or null.",
}


def allowed_tags(snapshot: dict, agent_id: str) -> list[Tag]:
    """Mirror of the validator rules, so the model is only offered legal moves."""
    head = snapshot["head"]
    if head is None:
        return [Tag.PROPOSE]
    if head["agent_id"] == agent_id:
        return [Tag.REVISE]  # authors may only answer objections by revising
    return [Tag.RATIFY, Tag.REVISE, Tag.REFUTE, Tag.REJECT]


def response_schema(allowed: list[Tag], persona: Persona) -> dict:
    return {
        "type": "object",
        "properties": {
            "tag": {"enum": [t.value for t in allowed]},
            "prediction": {"type": ["string", "null"], "maxLength": persona.max_prediction_chars},
            "explanation": {"type": ["string", "null"],
                            "description": f"{persona.explanation_min_words}-{persona.explanation_max_words} words"},
        },
        "required": ["tag", "prediction", "explanation"],
        "additionalProperties": False,
    }


def _board_text(snap: dict, window: int) -> str:
    head = snap["head"]
    if head is None:
        return "(empty - nobody has proposed an answer yet)"
    lines = [f"CURRENT ANSWER (#{head['seq']} by {head['agent_id']}): {head['prediction']}",
             f"REASONING: {head['explanation']}",
             "STANCES: " + ", ".join(f"{a}={t}" for a, t in sorted(snap["stances"].items())),
             "RECENT ACTIVITY:"]
    for m in snap["messages"][-window:]:
        tgt = f" -> #{m['target_id']}" if m["target_id"] else ""
        body = m["explanation"] or ""
        extra = f" (answer: {m['prediction']})" if m["prediction"] and m["tag"] != "PROPOSE" else ""
        lines.append(f"  #{m['seq']} {m['agent_id']} {m['tag']}{tgt}: {body}{extra}")
    return "\n".join(lines)


def build_messages(snapshot: dict, agent_id: str, persona: Persona, *, history_window: int = 8) -> list[dict]:
    allowed = allowed_tags(snapshot, agent_id)
    system = (f"{persona.system}\nYour role: {persona.title}. Your disposition: {persona.disposition}.\n\n{PRIMER}")
    state = {"agent_id": agent_id, "problem": snapshot["problem"], "version": snapshot["version"],
             "allowed_tags": [t.value for t in allowed],
             "head": ({k: snapshot["head"][k] for k in ("seq", "agent_id", "prediction", "explanation")}
                      if snapshot["head"] else None)}
    user = (f"PROBLEM:\n{snapshot['problem']}\n\nYOUR NAME: {agent_id}\n\nBOARD:\n{_board_text(snapshot, history_window)}\n\n"
            f"ALLOWED TAGS NOW: {', '.join(t.value for t in allowed)}\n"
            + "\n".join(TAG_RULES[t] for t in allowed)
            + f"\nExplanations: {persona.explanation_min_words}-{persona.explanation_max_words} words.\n\n"
            f"JSON schema for your reply:\n{json.dumps(response_schema(allowed, persona))}\n\n"
            f"<state>{json.dumps(state)}</state>")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def extract_state(messages: list[dict]) -> dict:
    """Read the machine-readable <state> block back out (used by mock models)."""
    for m in messages:
        if m["role"] == "user":
            hit = re.search(r"<state>(.*?)</state>", m["content"], re.S)
            if hit:
                return json.loads(hit.group(1))
    raise ValueError("no <state> block in prompt")


CF_PRIMER = """You are reflecting privately on a deadlocked panel discussion (a counterfactual replay).
Nobody else sees this. Look at the message you posted earlier and at every objection raised since.
Ask: "If I had posted something different at that point, would the panel be closer to agreement now?"
Propose alternative messages you could have posted instead. Good alternatives address ALL open
objections at once instead of only the latest one. Stay faithful to the evidence; do not concede a
claim you believe is false just to end the debate.
Reply with ONE JSON object: {"alternatives": [{"tag": ..., "prediction": ..., "explanation": ...}, ...]}"""


def build_counterfactual_messages(snapshot: dict, agent_id: str, persona: Persona, own_msg: dict,
                                  objections: list, k: int) -> list[dict]:
    """Prompt for the counterfactual sandbox: what could `agent_id` have posted instead of `own_msg`?"""
    allowed = [t.value for t in allowed_tags(snapshot, agent_id)]
    obj_lines = "\n".join(f"  - #{m['seq']} {m['agent_id']} {m['tag']}: {m['explanation']}"
                          + (f" (counter-answer: {m['prediction']})" if m.get("prediction") else "")
                          for m in objections) or "  (none)"
    state = {"agent_id": agent_id, "rollback_seq": own_msg["seq"], "allowed_tags": allowed, "k": k,
             "objections": [m["explanation"] for m in objections]}
    user = (f"PROBLEM:\n{snapshot['problem']}\n\nYOUR NAME: {agent_id}\n\n"
            f"BOARD JUST BEFORE YOUR MESSAGE #{own_msg['seq']}:\n{_board_text(snapshot, 8)}\n\n"
            f"WHAT YOU ACTUALLY POSTED (#{own_msg['seq']}): {own_msg['tag']} | {own_msg.get('prediction')} | "
            f"{own_msg.get('explanation')}\n\nOBJECTIONS RAISED DURING THE WHOLE DEBATE:\n{obj_lines}\n\n"
            f"Give up to {k} alternatives. Legal tags at that point: {', '.join(allowed)}.\n"
            f"<counterfactual>{json.dumps(state)}</counterfactual>")
    return [{"role": "system", "content": f"{persona.system}\nYour role: {persona.title}.\n\n{CF_PRIMER}"},
            {"role": "user", "content": user}]


def extract_cf_state(messages: list[dict]) -> Optional[dict]:
    for m in messages:
        if m["role"] == "user":
            hit = re.search(r"<counterfactual>(.*?)</counterfactual>", m["content"], re.S)
            if hit:
                return json.loads(hit.group(1))
    return None
