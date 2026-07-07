"""panoptica v2 — batched, backend-swappable instance segmentation metrics."""

from __future__ import annotations

from panoptica import backends, core
from panoptica.api import EvalResult, Evaluator
from panoptica.core import (
    Direction,
    InputType,
    LabelGroup,
    MetricMode,
    MetricType,
    SegmentationClassGroups,
)
from panoptica.metrics import Metric

__all__ = [
    "Evaluator",
    "EvalResult",
    "Metric",
    "InputType",
    "MetricType",
    "MetricMode",
    "Direction",
    "LabelGroup",
    "SegmentationClassGroups",
    "core",
    "backends",
]
