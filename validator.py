"""Validation rules for the four PXP action tags."""
from __future__ import annotations

from typing import Iterable, Optional

from models import Draft, Message, Tag


class ProtocolError(Exception):
    """A draft violates the PXP schema or current board state."""


class StaleTargetError(ProtocolError):
    """A draft targets a board state that is no longer current."""


class BoardClosedError(ProtocolError):
    """The board is closed or its message limit has been reached."""


def _require(value: Optional[str], name: str, tag: Tag) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ProtocolError(f"{tag.value} requires a non-empty {name}")


def validate(draft: Draft, *, head: Optional[Message], agents: Iterable[str]) -> None:
    """Validate one of RATIFY/REVISE/REFUTE/REJECT.

    The board starts with a system-seeded hypothesis (target id 0). After a
    REVISE, the resulting message becomes the new hypothesis and its sequence
    number becomes the target id for subsequent actions.
    """
    if draft.agent_id not in set(agents):
        raise ProtocolError(f"unknown agent {draft.agent_id!r}")
    if not isinstance(draft.tag, Tag):
        raise ProtocolError(f"invalid tag {draft.tag!r}")

    if head is None:
        raise ProtocolError(f"{draft.tag.value} requires an existing hypothesis")

    if draft.target_id is None:
        raise ProtocolError(f"{draft.tag.value} requires target_id")

    if draft.target_id != head.seq:
        raise StaleTargetError(
            f"target {draft.target_id} is not the current head ({head.seq})"
        )

    tag = draft.tag

    if tag is Tag.REVISE:
        _require(draft.prediction, "prediction", tag)
        _require(draft.explanation, "explanation", tag)
        if (
            draft.prediction == head.prediction
            and draft.explanation == head.explanation
        ):
            raise ProtocolError("no-op REVISE: prediction and explanation unchanged")
        return

    if draft.agent_id == head.agent_id:
        raise ProtocolError(f"author cannot {tag.value} its own hypothesis")

    if tag is Tag.RATIFY:
        if draft.prediction is not None or draft.explanation is not None:
            raise ProtocolError("RATIFY carries no new content")

    elif tag is Tag.REFUTE:
        _require(draft.explanation, "explanation", tag)
        if draft.prediction is not None:
            raise ProtocolError("REFUTE cannot carry a prediction")

    elif tag is Tag.REJECT:
        _require(draft.explanation, "explanation", tag)

