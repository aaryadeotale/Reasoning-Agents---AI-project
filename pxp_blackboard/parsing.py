"""Turn raw model text into a validated Draft, or explain precisely what is wrong."""
from __future__ import annotations

import json
from typing import Optional

from .models import Draft, Message, Tag
from .prompts import Persona
from .validator import ProtocolError, validate


class FormatError(ValueError):
    """Model output is unusable; the message is fed back to the model for repair."""


def extract_json(text: str, key: str = "tag") -> dict:
    """Find the first JSON object that has `key` (default 'tag'); tolerates fences and chatter."""
    dec, seen = json.JSONDecoder(), False
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = dec.raw_decode(text[i:])
        except ValueError:
            continue
        if isinstance(obj, dict):
            seen = True
            if key in obj:
                return obj
    raise FormatError(f"JSON object has no '{key}' key" if seen else "no valid JSON object found in your reply")


def _clean(v) -> Optional[str]:
    if v is None:
        return None
    s = (v if isinstance(v, str) else json.dumps(v) if isinstance(v, (list, dict)) else str(v)).strip()
    return s or None


def head_message(snapshot: dict) -> Optional[Message]:
    h = snapshot["head"]
    return Message(**{**h, "tag": Tag(h["tag"])}) if h else None


def parse_response(text: str, *, agent_id: str, snapshot: dict, allowed: list[Tag], persona: Persona) -> Draft:
    obj = extract_json(text)
    names = [t.value for t in allowed]
    try:
        tag = Tag(str(obj.get("tag", "")).strip().upper())
    except ValueError:
        raise FormatError(f"'tag' must be one of {names}") from None
    if tag not in allowed:
        raise FormatError(f"tag {tag.value} is not allowed now; allowed tags: {names}")

    pred, expl = _clean(obj.get("prediction")), _clean(obj.get("explanation"))
    if tag is Tag.RATIFY:       # small models like to add prose; it is meaningless here
        pred = expl = None
    elif tag is Tag.REFUTE:
        pred = None
    if pred and len(pred) > persona.max_prediction_chars:
        raise FormatError(f"prediction too long ({len(pred)} chars, max {persona.max_prediction_chars})")
    if expl and tag is not Tag.RATIFY:
        words = expl.split()
        if len(words) < persona.explanation_min_words:
            raise FormatError(f"explanation too short ({len(words)} words, min {persona.explanation_min_words})")
        if len(words) > persona.explanation_max_words:
            expl = " ".join(words[:persona.explanation_max_words])

    head = snapshot["head"]
    draft = Draft(agent_id, tag, None if tag is Tag.PROPOSE else head["seq"], pred, expl)
    try:
        validate(draft, head=head_message(snapshot), agents=snapshot["agents"])
    except ProtocolError as e:
        raise FormatError(str(e)) from None
    return draft
