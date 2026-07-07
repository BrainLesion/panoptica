"""Core: data model, enums, protocols, errors — zero heavy compute deps."""

from panoptica.core.constants import BACKGROUND, CCABackend
from panoptica.core.enums import Direction, InputType, MetricMode, MetricType
from panoptica.core.errors import (
    BackendUnavailable,
    InputValidationError,
    MetricComputeError,
    PanopticaError,
)
from panoptica.core.labels import (
    InstanceLabelMap,
    LabelGroup,
    LabelPartGroup,
    SegmentationClassGroups,
)
from panoptica.core.pairs import (
    EvaluateInstancePair,
    MatchedInstancePair,
    SemanticPair,
    UnmatchedInstancePair,
)
from panoptica.core.protocols import Array, MetricFn, Serializable, Xp

__all__ = [
    "BACKGROUND",
    "CCABackend",
    "Direction",
    "InputType",
    "MetricMode",
    "MetricType",
    "PanopticaError",
    "MetricComputeError",
    "BackendUnavailable",
    "InputValidationError",
    "LabelGroup",
    "LabelPartGroup",
    "SegmentationClassGroups",
    "InstanceLabelMap",
    "SemanticPair",
    "UnmatchedInstancePair",
    "MatchedInstancePair",
    "EvaluateInstancePair",
    "Array",
    "Xp",
    "Serializable",
    "MetricFn",
]
