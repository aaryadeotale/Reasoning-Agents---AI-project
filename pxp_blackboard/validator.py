"""PXP message validator: structural schema + state-transition rules."""
from __future__ import annotations

from typing import Iterable, Optional

from .models import Draft, Message, Tag


class ProtocolError(Exception):
    """A draft violates the PXP schema or the current board state."""


class StaleTargetError(ProtocolError):
    """The draft was written against a head that is no longer current (benign race)."""


class BoardClosedError(ProtocolError):
    """The board is closed or its message limit is reached."""


def _require(value: Optional[str], name: str, tag: Tag) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ProtocolError(f"{tag.value} requires a non-empty {name}")


def validate(draft: Draft, *, head: Optional[Message], agents: Iterable[str]) -> None:
    """Raise ProtocolError if `draft` is not a legal move given the current head.

    Rules (simplified PXP):
      PROPOSE: only when the board has no head; needs prediction + explanation.
      REVISE : targets the head; needs a *changed* prediction/explanation.
      RATIFY : targets the head; carries no new content; not by the head's author.
      REFUTE : targets the head; explanation (the roadblock) only; not by the author.
      REJECT : targets the head; explanation required, counter-prediction optional.
    """
    if draft.agent_id not in set(agents):
        raise ProtocolError(f"unknown agent {draft.agent_id!r}")
    if not isinstance(draft.tag, Tag):
        raise ProtocolError(f"invalid tag {draft.tag!r}")
    tag = draft.tag

    if tag is Tag.PROPOSE:
        if head is not None:
            raise StaleTargetError("board already has a hypothesis; use REVISE")
        if draft.target_id is not None:
            raise ProtocolError("PROPOSE must not have a target")
        _require(draft.prediction, "prediction", tag)
        _require(draft.explanation, "explanation", tag)
        return

    if head is None:
        raise ProtocolError(f"{tag.value} requires an existing hypothesis")
    if draft.target_id is None:
        raise ProtocolError(f"{tag.value} requires target_id")
    if draft.target_id != head.seq:
        raise StaleTargetError(f"target {draft.target_id} is not the current head ({head.seq})")

    if tag is Tag.REVISE:
        _require(draft.prediction, "prediction", tag)
        _require(draft.explanation, "explanation", tag)
        if draft.prediction == head.prediction and draft.explanation == head.explanation:
            raise ProtocolError("no-op REVISE: prediction and explanation unchanged")
        return

    if draft.agent_id == head.agent_id:
        raise ProtocolError(f"author cannot {tag.value} its own hypothesis")
    if tag is Tag.RATIFY:
        if draft.prediction or draft.explanation:
            raise ProtocolError("RATIFY carries no new content (use REVISE)")
    elif tag is Tag.REFUTE:
        _require(draft.explanation, "explanation", tag)
        if draft.prediction:
            raise ProtocolError("REFUTE cannot carry a prediction (use REJECT or REVISE)")
    elif tag is Tag.REJECT:
        _require(draft.explanation, "explanation", tag)
