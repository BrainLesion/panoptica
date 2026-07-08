"""Immutable data-model for the processing pairs that flow through the pipeline.

A pair is progressively transformed: SemanticPair -> UnmatchedInstancePair ->
MatchedInstancePair -> EvaluateInstancePair. Arrays are backend arrays (numpy or
cupy); `spacing` is physical voxel spacing (mm) or None (defaults to 1 per axis).

Frozen dataclasses: transformations return new instances rather than mutating,
which keeps kernels/metrics pure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from panoptica.core.protocols import Array


@dataclass(frozen=True)
class SemanticPair:
    """Raw semantic masks; instances not yet extracted."""

    ref: Array
    pred: Array
    spacing: tuple[float, ...] | None = None


@dataclass(frozen=True)
class UnmatchedInstancePair:
    """Instances exist on both sides but labels do not correspond across ref/pred."""

    ref: Array
    pred: Array
    n_ref: int
    n_pred: int
    spacing: tuple[float, ...] | None = None


@dataclass(frozen=True)
class MatchedInstancePair:
    """Reference and prediction instance labels correspond.

    `matched_ids` is a (K, 2) integer array of [ref_id, pred_id] rows for the K
    true-positive matches. Unmatched ref ids (FN) and pred ids (FP) are recorded
    separately so counts are exact.
    """

    ref: Array
    pred: Array
    matched_ids: Array
    unmatched_ref: tuple[int, ...] = ()
    unmatched_pred: tuple[int, ...] = ()
    spacing: tuple[float, ...] | None = None


@dataclass(frozen=True)
class EvaluateInstancePair:
    """Per-instance arrays cropped and aligned, ready for batched metric evaluation.

    `ref_ids`/`pred_ids` are positionally-aligned (K,) arrays: metric k compares
    ref==ref_ids[k] against pred==pred_ids[k].
    """

    ref: Array
    pred: Array
    ref_ids: Array
    pred_ids: Array
    n_ref: int
    n_pred: int
    tp: int
    spacing: tuple[float, ...] | None = None
    meta: dict = field(default_factory=dict)
