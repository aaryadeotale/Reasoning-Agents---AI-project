"""Core data types for the PXP blackboard.

Milestone 1 uses only the four PXP action tags:
RATIFY, REVISE, REFUTE and REJECT.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional


class Tag(str, Enum):
    RATIFY = "RATIFY"
    REVISE = "REVISE"
    REFUTE = "REFUTE"
    REJECT = "REJECT"


NEGATIVE_TAGS = frozenset({Tag.REFUTE, Tag.REJECT})


@dataclass(frozen=True)
class Draft:
    """What an agent submits to the blackboard."""
    agent_id: str
    tag: Tag
    target_id: Optional[int] = None
    prediction: Optional[str] = None
    explanation: Optional[str] = None
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Message:
    """An accepted, immutable PXP action in the blackboard log."""
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
