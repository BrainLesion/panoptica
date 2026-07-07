"""Edge-case policy: what a metric returns when inputs are degenerate.

Governs cases like empty reference, empty prediction, or a TP count of zero —
where a raw metric formula would divide by zero. The handler maps
(metric, situation) -> sentinel value (0.0, NaN, inf).
"""

from __future__ import annotations

from enum import Enum, auto


class EdgeCaseResult(Enum):
    """Sentinel a metric takes in a degenerate situation."""

    NONE = auto()
    ZERO = auto()
    ONE = auto()
    NAN = auto()
    INF = auto()


class EdgeCaseHandler:
    """Resolves degenerate metric situations to sentinel values."""

    def __init__(self) -> None: ...

    def handle_zero_tp(self, metric: str, num_ref: int, num_pred: int) -> float:
        raise NotImplementedError("filled in by owning stream against v1 policy")
