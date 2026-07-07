"""Parity tolerance table for metric comparison (cpu vs cuda, self-consistency).

These are the measured acceptable deviations per metric family from the phase-0
precision spike. Values are conservative and refined by actual measurement.

Rationale:
- Counts (TP, FP, FN): exact match (integer, no rounding).
- Volumetric (DSC/IOU/RVD/RVAE): rtol=1e-9 (both devices, ratios of integers).
- Surface (ASSD/HD/HD95): atol~1e-5 voxel (mixed precision on GPU, but EDT
  distances are computed from integer squared-distances; surface vector rounding
  is ~1e-6 relative; the threshold is chosen to be safe for device differences).
- NSD: rtol=1e-6 (surface metric, compared in squared-int space to avoid threshold ties).
- clDice: rtol=1e-6 (skeletonize is CPU-only; single precision path).
- CenterDistance: atol=1e-6 voxel.
- RQ/Precision/Recall: rtol=1e-9 (ratios of exact counts).
- SQ: rtol=1e-6 (aggregate of metrics).
- PQ: rtol=1e-6 (product of SQ × RQ).
"""

from __future__ import annotations

import math
from typing import Any

__all__ = ["TOLERANCES", "compare"]

# Metric family -> (metric_type, rtol, atol)
# metric_type: "count" | "volumetric" | "surface" | "topology" | "distance" | "quality"
TOLERANCES: dict[str, tuple[str, float, float]] = {
    # Counts (exact)
    "tp": ("count", 0.0, 0.0),
    "fp": ("count", 0.0, 0.0),
    "fn": ("count", 0.0, 0.0),
    "num_ref_instances": ("count", 0.0, 0.0),
    "num_pred_instances": ("count", 0.0, 0.0),
    # Volumetric metrics
    "dsc": ("volumetric", 1e-9, 0.0),
    "iou": ("volumetric", 1e-9, 0.0),
    "rvd": ("volumetric", 1e-9, 0.0),
    "rvae": ("volumetric", 1e-9, 0.0),
    # Surface distances (voxel units)
    "assd": ("surface", 0.0, 1e-5),
    "hd": ("surface", 0.0, 1e-5),
    "hd95": ("surface", 0.0, 1e-5),
    # Normalized Surface Dice
    "nsd": ("topology", 1e-6, 0.0),
    # Centerline Dice
    "cldice": ("topology", 1e-6, 0.0),
    # Center distance
    "cedi": ("distance", 0.0, 1e-6),
    # Panoptic quality metrics
    "rq": ("quality", 1e-9, 0.0),
    "precision": ("quality", 1e-9, 0.0),
    "recall": ("quality", 1e-9, 0.0),
    # SQ and PQ variants (aggregates)
    "sq": ("quality", 1e-6, 0.0),
    "pq": ("quality", 1e-6, 0.0),
}


def compare(a: Any, b: Any, metric: str) -> bool:
    """Compare two metric values with per-metric tolerance.

    Args:
        a: First value (e.g. cpu, or reference side).
        b: Second value (e.g. cuda, or test side).
        metric: Metric name (lowercase, e.g., "dsc", "assd", "tp").

    Returns:
        True if values match within tolerance, False otherwise.

    Raises:
        ValueError: If metric is unknown or values are not numeric.
    """
    metric_lower = metric.lower()
    if metric_lower not in TOLERANCES:
        raise ValueError(
            f"Unknown metric {metric!r}; known metrics: {sorted(TOLERANCES.keys())}"
        )

    # Handle NaN: both NaN is OK, one NaN is failure
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        if math.isnan(a) or math.isnan(b):
            return False

    # Handle non-numeric (lists, etc.)
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise ValueError(
            f"Non-numeric values for metric {metric}: a={type(a)}, b={type(b)}"
        )

    metric_type, rtol, atol = TOLERANCES[metric_lower]

    # Exact count comparison
    if metric_type == "count":
        return a == b

    # Relative tolerance (for zero, use atol)
    if rtol > 0:
        if abs(a) < 1e-15:
            # a is effectively zero; use atol if available, else exact
            return abs(b) <= atol if atol > 0 else b == 0
        return abs((b - a) / a) <= rtol

    # Absolute tolerance
    if atol > 0:
        return abs(b - a) <= atol

    # Should not reach here if tolerances are well-defined
    raise ValueError(f"No tolerance defined for metric {metric!r}")
