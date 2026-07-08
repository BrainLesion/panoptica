"""Volumetric instance metrics: Dice, IoU, RVD, RVAE.

Fully vectorized: instance areas and per-pair intersections are computed for all
matched pairs at once via ``bincount`` (a few full-array passes), then each metric
is an elementwise expression over the resulting K-vectors -- no per-instance
Python loop and no per-instance host sync.
"""

from __future__ import annotations

from panoptica.core.edge_cases import EdgeCaseResult
from panoptica.core.enums import Direction, MetricType
from panoptica.core.errors import InputValidationError
from panoptica.core.protocols import Array, Xp
from panoptica.metrics.registry import ZeroTPPolicy, register


def _check_pairs(ref_ids: Array, pred_ids: Array) -> int:
    n_ref = len(ref_ids)
    n_pred = len(pred_ids)
    if n_ref != n_pred:
        raise InputValidationError(
            f"ref_ids and pred_ids must be positionally aligned (got {n_ref} vs {n_pred})"
        )
    return n_ref


def _gather(arr: Array, idx: Array, xp: Xp) -> Array:
    """arr[idx] as float64, returning 0.0 for out-of-range indices."""
    n = arr.shape[0]
    in_range = idx < n
    safe = xp.where(in_range, idx, 0)
    vals = arr[safe].astype(xp.float64)
    return xp.where(in_range, vals, 0.0)


def volume_stats(
    ref: Array, pred: Array, ref_ids: Array, pred_ids: Array, xp: Xp
) -> tuple[Array, Array, Array]:
    """(ref_area, pred_area, intersection) K-vectors for the matched pairs.

    Areas from one bincount per array; intersections from one bincount over the
    encoded (ref, pred) voxel pairs, gathered at the K matched encodings.
    """
    ref_flat = ref.reshape(-1)
    pred_flat = pred.reshape(-1)
    # Instances are sparse; restrict every bincount/where to foreground voxels
    # (nonzero ref OR pred) so they scan ~1% of the volume, not all of it. Every
    # label>=1 area and intersection is fully contained in this mask.
    fg = (ref_flat != 0) | (pred_flat != 0)
    ref_flat = ref_flat[fg]
    pred_flat = pred_flat[fg]
    if ref_flat.dtype.kind not in "iu":
        ref_flat = ref_flat.astype(xp.int64)
    if pred_flat.dtype.kind not in "iu":
        pred_flat = pred_flat.astype(xp.int64)
    ref_ids = xp.asarray(ref_ids, dtype=xp.int64)
    pred_ids = xp.asarray(pred_ids, dtype=xp.int64)

    ref_area = xp.bincount(ref_flat)
    pred_area = xp.bincount(pred_flat)
    r = _gather(ref_area, ref_ids, xp)
    p = _gather(pred_area, pred_ids, xp)

    if ref_ids.shape[0] and bool((ref_ids == pred_ids).all()):
        # Common case (pipeline): matched pairs share a label, so the
        # intersection of pair L is just voxels where ref == pred == L.
        agree = xp.where(ref_flat == pred_flat, ref_flat, 0)
        inter = _gather(xp.bincount(agree), ref_ids, xp)
    else:
        stride = (int(pred_flat.max()) + 1) if pred_flat.size else 1
        enc = ref_flat.astype(xp.int64) * stride + pred_flat.astype(xp.int64)
        inter = _gather(xp.bincount(enc), ref_ids * stride + pred_ids, xp)
    return r, p, inter


def _stats(ref, pred, ref_ids, pred_ids, xp, params) -> tuple[Array, Array, Array]:
    pre = params.get("_vol_stats")
    if pre is not None:
        return pre
    return volume_stats(ref, pred, ref_ids, pred_ids, xp)


@register(
    id="DSC",
    type=MetricType.INSTANCE,
    direction=Direction.INCREASING,
    long_name="Dice",
    zero_tp=ZeroTPPolicy(default=EdgeCaseResult.ZERO, no_instances=EdgeCaseResult.NAN),
)
def dice_batched(
    ref: Array,
    pred: Array,
    ref_ids: Array,
    pred_ids: Array,
    xp: Xp,
    *,
    spacing: tuple[float, ...] | None = None,
    **params: object,
) -> Array:
    """Dice = 2|ref & pred| / (|ref| + |pred|); 0.0 when both masks are empty."""
    _check_pairs(ref_ids, pred_ids)
    r, p, inter = _stats(ref, pred, ref_ids, pred_ids, xp, params)
    denom = r + p
    return xp.where(denom == 0, 0.0, 2.0 * inter / xp.where(denom == 0, 1.0, denom))


@register(
    id="IOU",
    type=MetricType.INSTANCE,
    direction=Direction.INCREASING,
    long_name="Intersection over Union",
    zero_tp=ZeroTPPolicy(
        default=EdgeCaseResult.ZERO,
        no_instances=EdgeCaseResult.NAN,
        empty_pred=EdgeCaseResult.ZERO,
    ),
)
def iou_batched(
    ref: Array,
    pred: Array,
    ref_ids: Array,
    pred_ids: Array,
    xp: Xp,
    *,
    spacing: tuple[float, ...] | None = None,
    **params: object,
) -> Array:
    """IoU = |ref & pred| / |ref | pred|; 0.0 when the union is empty."""
    _check_pairs(ref_ids, pred_ids)
    r, p, inter = _stats(ref, pred, ref_ids, pred_ids, xp, params)
    union = r + p - inter
    return xp.where(union == 0, 0.0, inter / xp.where(union == 0, 1.0, union))


@register(
    id="RVD",
    type=MetricType.INSTANCE,
    direction=Direction.DECREASING,
    long_name="Relative Volume Difference",
    zero_tp=ZeroTPPolicy(default=EdgeCaseResult.NAN),
)
def rvd_batched(
    ref: Array,
    pred: Array,
    ref_ids: Array,
    pred_ids: Array,
    xp: Xp,
    *,
    spacing: tuple[float, ...] | None = None,
    **params: object,
) -> Array:
    """RVD = (|pred| - |ref|) / |ref|; 0.0 when both masks are empty."""
    _check_pairs(ref_ids, pred_ids)
    r, p, _inter = _stats(ref, pred, ref_ids, pred_ids, xp, params)
    both_empty = (r == 0) & (p == 0)
    val = (p - r) / xp.where(r == 0, xp.nan, r)
    return xp.where(both_empty, 0.0, val)


@register(
    id="RVAE",
    type=MetricType.INSTANCE,
    direction=Direction.DECREASING,
    long_name="Relative Volume Absolute Error",
    zero_tp=ZeroTPPolicy(default=EdgeCaseResult.NAN),
)
def rvae_batched(
    ref: Array,
    pred: Array,
    ref_ids: Array,
    pred_ids: Array,
    xp: Xp,
    *,
    spacing: tuple[float, ...] | None = None,
    **params: object,
) -> Array:
    """RVAE = |(|pred| - |ref|) / |ref||; 0.0 when both masks are empty."""
    _check_pairs(ref_ids, pred_ids)
    r, p, _inter = _stats(ref, pred, ref_ids, pred_ids, xp, params)
    both_empty = (r == 0) & (p == 0)
    val = xp.abs((p - r) / xp.where(r == 0, xp.nan, r))
    return xp.where(both_empty, 0.0, val)
