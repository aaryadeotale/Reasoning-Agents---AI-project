"""Core data types for the PXP blackboard."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional


class Tag(str, Enum):
    PROPOSE = "PROPOSE"  # simplification: opening move that seeds the board
    RATIFY = "RATIFY"
    REVISE = "REVISE"
    REFUTE = "REFUTE"
    REJECT = "REJECT"


HYPOTHESIS_TAGS = frozenset({Tag.PROPOSE, Tag.REVISE})  # tags that create a new "head"
NEGATIVE_TAGS = frozenset({Tag.REFUTE, Tag.REJECT})


@dataclass(frozen=True)
class Draft:
    """What an agent submits. The board turns it into a Message."""
    agent_id: str
    tag: Tag
    target_id: Optional[int] = None  # seq of the hypothesis being responded to
    prediction: Optional[str] = None  # the "what"
    explanation: Optional[str] = None  # the "why"
    meta: dict = field(default_factory=dict)  # tokens used, latency, etc.


@dataclass(frozen=True)
class Message:
    """An accepted, immutable entry in the append-only blackboard log."""
    seq: int
    agent_id: str
    tag: Tag
    target_id: Optional[int]
    prediction: Optional[str]
    explanation: Optional[str]
    timestamp: float
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["tag"] = self.tag.value
        return d
