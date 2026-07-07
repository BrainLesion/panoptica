"""Core enums: input types, metric classification, aggregation modes, directions."""

from __future__ import annotations

from enum import Enum, auto


class InputType(Enum):
    """Which pipeline phase the input enters at."""

    SEMANTIC = auto()
    UNMATCHED_INSTANCE = auto()
    MATCHED_INSTANCE = auto()


class MetricType(Enum):
    """Classification of a metric for dispatch/printing."""

    NO_PRINT = auto()
    MATCHING = auto()
    GLOBAL = auto()
    INSTANCE = auto()


class MetricMode(Enum):
    """Aggregation applied to a per-instance metric list."""

    ALL = auto()
    AVG = auto()
    SUM = auto()
    STD = auto()
    MIN = auto()
    MAX = auto()


class Direction(Enum):
    """Whether a higher metric value is better (for decision thresholds/AUTC)."""

    INCREASING = auto()  # higher is better (DSC, IOU, NSD, clDSC)
    DECREASING = auto()  # lower is better (ASSD, HD, HD95, RVD, RVAE, CEDI)
