"""Metrics layer.

Importing this package imports every metric submodule so their ``@register``
decorators populate ``METRIC_REGISTRY`` (and the derived ``Metric`` enum).
"""

from panoptica.metrics import (  # noqa: F401 (registration side-effects)
    center,
    surface,
    topology,
    volumetric,
)
from panoptica.metrics.registry import (
    METRIC_REGISTRY,
    Metric,
    MetricSpec,
    aggregate,
    derive_detection_quality,
    derive_pq,
    derive_sq,
    get_spec,
    register,
)

__all__ = [
    "METRIC_REGISTRY",
    "Metric",
    "MetricSpec",
    "aggregate",
    "get_spec",
    "register",
    "derive_detection_quality",
    "derive_sq",
    "derive_pq",
]
