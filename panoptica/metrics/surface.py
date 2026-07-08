"""Surface-distance instance metrics: ASSD, HD, HD95, NSD.

For a matched pair (ref_id, pred_id):

    ref_border  = surface_border(ref == ref_id)
    pred_border = surface_border(pred == pred_id)
    sd_ref  = edt(~pred_border)[ref_border]   # ref-border points -> nearest pred-border point
    sd_pred = edt(~ref_border)[pred_border]   # pred-border points -> nearest ref-border point

ASSD/HD/HD95/NSD share this pair to avoid recomputing the EDT four times.

`edt`/`surface_border` come from `panoptica.kernels` and are imported *lazily* inside
each function body -- importing this module must never require kernels/ to
exist yet.
"""

from __future__ import annotations

import sys

from panoptica.core.edge_cases import EdgeCaseResult
from panoptica.core.enums import Direction, MetricType
from panoptica.core.errors import InputValidationError, MetricComputeError
from panoptica.core.protocols import Array, Xp
from panoptica.metrics.registry import ZeroTPPolicy, register


def _kernels():
    try:
        from panoptica.kernels import edt, surface_border
    except ImportError as e:  # pragma: no cover - depends on kernels stream landing
        raise MetricComputeError(
            "panoptica.kernels.edt/surface_border are not available yet "
            "(kernels stream has not landed)"
        ) from e
    return edt, surface_border


def _check_pairs(ref_ids: Array, pred_ids: Array) -> int:
    n_ref = len(ref_ids)
    n_pred = len(pred_ids)
    if n_ref != n_pred:
        raise InputValidationError(
            f"ref_ids and pred_ids must be positionally aligned (got {n_ref} vs {n_pred})"
        )
    return n_ref


def _padded_union_box(ref_mask: Array, pred_mask: Array, xp: Xp):
    """Bounding box of (ref | pred) padded by 1 voxel per axis (clamped).

    Cropping to the instance's own bbox turns the O(voxels) EDT from a
    full-volume cost into a tiny-crop cost. The 1-voxel pad is required:
    surface_border = mask XOR erosion(mask) treats the array edge as
    background, so a mask touching the crop face would gain spurious border
    voxels — a background margin reproduces the full-volume border exactly.
    """
    union = xp.logical_or(ref_mask, pred_mask)
    if not bool(union.any()):
        return None
    coords = xp.where(union)
    box = []
    for axis, c in enumerate(coords):
        lo = int(c.min()) - 1
        hi = int(c.max()) + 2  # +1 for inclusive, +1 for pad
        box.append(slice(max(lo, 0), min(hi, ref_mask.shape[axis])))
    return tuple(box)


def _surface_distance_pair(
    ref_mask: Array,
    pred_mask: Array,
    xp: Xp,
    spacing: tuple[float, ...] | None,
    already_boxed: bool = False,
) -> tuple[Array, Array]:
    """One (sd_ref, sd_pred) directional surface-distance pair for one instance.

    The masks are cropped to their padded union bbox first so the EDT runs on
    a small volume instead of the whole image. ``already_boxed`` skips that crop
    when the caller (``surface_distances_from_crops``) already passes the padded
    union crop — the re-box is not only redundant work but, on GPU, its
    ``int(min())``/``bool(any())`` are per-instance device→host syncs that stall
    the stream. Skipping them is bit-identical (the crop already carries the
    1-voxel background margin ``surface_border`` needs).
    """
    edt, surface_border = _kernels()
    if not already_boxed:
        box = _padded_union_box(ref_mask, pred_mask, xp)
        if box is None:
            empty = xp.zeros((0,), dtype=xp.float64)
            return empty, empty
        ref_mask = ref_mask[box]
        pred_mask = pred_mask[box]
    elif ref_mask.size == 0:
        empty = xp.zeros((0,), dtype=xp.float64)
        return empty, empty
    ref_border = surface_border(ref_mask, xp)
    pred_border = surface_border(pred_mask, xp)
    dt_from_pred = edt(xp.logical_not(pred_border), xp, spacing=spacing)
    dt_from_ref = edt(xp.logical_not(ref_border), xp, spacing=spacing)
    sd_ref = dt_from_pred[ref_border]
    sd_pred = dt_from_ref[pred_border]
    return sd_ref, sd_pred


# Sentinel for a degenerate instance with an empty ref- or pred-border.
_EMPTY_SURFACE = float("nan")


def _empty_surface(sd_ref: Array, sd_pred: Array) -> bool:
    """True if either directional surface-distance array has no elements."""
    return sd_ref.shape[0] == 0 or sd_pred.shape[0] == 0


def compute_surface_distances(
    ref: Array,
    pred: Array,
    ref_ids: Array,
    pred_ids: Array,
    xp: Xp,
    spacing: tuple[float, ...] | None,
) -> list[tuple[Array, Array]]:
    """All per-instance (sd_ref, sd_pred) pairs, computed ONCE.

    ASSD/HD/HD95/NSD all reduce the same distance pairs; evaluate() computes them
    once via this function and passes them to each reducer as `_sd_pairs` so the
    (expensive) EDT is not repeated 4x per instance.
    """
    k = _check_pairs(ref_ids, pred_ids)
    return [
        _surface_distance_pair(ref == ref_ids[i], pred == pred_ids[i], xp, spacing)
        for i in range(k)
    ]


def surface_distances_from_crops(
    crops: list[tuple[Array, Array]],
    xp: Xp,
    spacing: tuple[float, ...] | None,
    n_jobs: int = 1,
) -> list[tuple[Array, Array]]:
    """Surface-distance pairs from precomputed (ref_bool, pred_bool) crops."""
    from panoptica.metrics._parallel import parallel_list

    jobs = 1 if xp.__name__ == "cupy" else n_jobs
    return parallel_list(
        lambda c: _surface_distance_pair(c[0], c[1], xp, spacing, already_boxed=True),
        crops,
        jobs,
    )


def _pairs(
    ref, pred, ref_ids, pred_ids, xp, spacing, params
) -> list[tuple[Array, Array]]:
    """Precomputed `_sd_pairs` from params, or compute them (standalone call).

    Falls back to `_crops` (shared cropped masks) if present, else full arrays.
    """
    pre = params.get("_sd_pairs")
    if pre is not None:
        return pre
    crops = params.get("_crops")
    if crops is not None:
        return surface_distances_from_crops(crops, xp, spacing)
    return compute_surface_distances(ref, pred, ref_ids, pred_ids, xp, spacing)


@register(
    id="ASSD",
    type=MetricType.INSTANCE,
    direction=Direction.DECREASING,
    long_name="Average Symmetric Surface Distance",
    cpu_only=False,
    zero_tp=ZeroTPPolicy(default=EdgeCaseResult.INF, no_instances=EdgeCaseResult.NAN),
)
def assd_batched(
    ref: Array,
    pred: Array,
    ref_ids: Array,
    pred_ids: Array,
    xp: Xp,
    *,
    spacing: tuple[float, ...] | None = None,
    **params: object,
) -> Array:
    """ASSD = mean(mean(sd_ref), mean(sd_pred)) -- final reduction in float64."""
    k = _check_pairs(ref_ids, pred_ids)
    pairs = _pairs(ref, pred, ref_ids, pred_ids, xp, spacing, params)
    out = xp.empty((k,), dtype=xp.float64)
    for i in range(k):
        sd_ref, sd_pred = pairs[i]
        if _empty_surface(sd_ref, sd_pred):
            out[i] = _EMPTY_SURFACE
            continue
        mean_ref = xp.mean(xp.asarray(sd_ref, dtype=xp.float64))
        mean_pred = xp.mean(xp.asarray(sd_pred, dtype=xp.float64))
        out[i] = (mean_ref + mean_pred) / 2.0
    return out


@register(
    id="HD",
    type=MetricType.INSTANCE,
    direction=Direction.DECREASING,
    long_name="Hausdorff Distance",
    zero_tp=ZeroTPPolicy(default=EdgeCaseResult.INF, no_instances=EdgeCaseResult.NAN),
)
def hd_batched(
    ref: Array,
    pred: Array,
    ref_ids: Array,
    pred_ids: Array,
    xp: Xp,
    *,
    spacing: tuple[float, ...] | None = None,
    **params: object,
) -> Array:
    """HD = max(max(sd_ref), max(sd_pred))."""
    k = _check_pairs(ref_ids, pred_ids)
    pairs = _pairs(ref, pred, ref_ids, pred_ids, xp, spacing, params)
    out = xp.empty((k,), dtype=xp.float64)
    for i in range(k):
        sd_ref, sd_pred = pairs[i]
        if _empty_surface(sd_ref, sd_pred):
            out[i] = _EMPTY_SURFACE
            continue
        # Stay on-device (no float()): converting per instance would force a
        # host sync each iteration and serialize the GPU.
        out[i] = xp.maximum(
            xp.max(xp.asarray(sd_ref, dtype=xp.float64)),
            xp.max(xp.asarray(sd_pred, dtype=xp.float64)),
        )
    return out


@register(
    id="HD95",
    type=MetricType.INSTANCE,
    direction=Direction.DECREASING,
    long_name="Hausdorff Distance 95",
    zero_tp=ZeroTPPolicy(default=EdgeCaseResult.INF, no_instances=EdgeCaseResult.NAN),
)
def hd95_batched(
    ref: Array,
    pred: Array,
    ref_ids: Array,
    pred_ids: Array,
    xp: Xp,
    *,
    spacing: tuple[float, ...] | None = None,
    **params: object,
) -> Array:
    """HD95 = 95th percentile of the pooled (sd_ref, sd_pred) distances."""
    k = _check_pairs(ref_ids, pred_ids)
    pairs = _pairs(ref, pred, ref_ids, pred_ids, xp, spacing, params)
    out = xp.empty((k,), dtype=xp.float64)
    for i in range(k):
        sd_ref, sd_pred = pairs[i]
        if _empty_surface(sd_ref, sd_pred):
            out[i] = _EMPTY_SURFACE
            continue
        pooled = xp.concatenate(
            (
                xp.asarray(sd_ref, dtype=xp.float64).reshape(-1),
                xp.asarray(sd_pred, dtype=xp.float64).reshape(-1),
            )
        )
        out[i] = xp.percentile(pooled, 95)  # on-device; no per-instance sync
    return out


@register(
    id="NSD",
    type=MetricType.INSTANCE,
    # NSD is marked decreasing despite the score being a Dice-like overlap fraction.
    direction=Direction.DECREASING,
    long_name="Normalized Surface Dice",
    zero_tp=ZeroTPPolicy(default=EdgeCaseResult.ZERO, no_instances=EdgeCaseResult.NAN),
)
def nsd_batched(
    ref: Array,
    pred: Array,
    ref_ids: Array,
    pred_ids: Array,
    xp: Xp,
    *,
    spacing: tuple[float, ...] | None = None,
    threshold: float | None = None,
    **params: object,
) -> Array:
    """Normalized Surface Dice at `threshold` (default: min(spacing), unit if None).

    When no spacing is given the grid is treated as unit-isotropic, so the
    threshold defaults to min((1, ..., 1)) = 1.0 -- matching v1, whose evaluator
    substitutes a default voxel spacing of (1.0, ...) when none is supplied.

    Counts are decided by comparing *squared* distance against *squared*
    threshold (never sqrt-then-compare) so the count is device independent;
    this is mathematically identical to comparing the raw distances (a
    monotone transform on non-negative values).
    """
    k = _check_pairs(ref_ids, pred_ids)
    thr = threshold if threshold is not None else (min(spacing) if spacing else 1.0)
    thr2 = thr * thr
    pairs = _pairs(ref, pred, ref_ids, pred_ids, xp, spacing, params)
    out = xp.empty((k,), dtype=xp.float64)
    for i in range(k):
        sd_ref, sd_pred = pairs[i]
        if _empty_surface(sd_ref, sd_pred):
            out[i] = _EMPTY_SURFACE
            continue
        sd_ref64 = xp.asarray(sd_ref, dtype=xp.float64)
        sd_pred64 = xp.asarray(sd_pred, dtype=xp.float64)
        numel_a = sd_ref64.shape[0]
        numel_b = sd_pred64.shape[0]
        # On-device scalars (no float()): numel_* are host ints from .shape.
        tp_a = xp.sum(sd_ref64 * sd_ref64 <= thr2) / numel_a
        tp_b = xp.sum(sd_pred64 * sd_pred64 <= thr2) / numel_b
        fp = xp.sum(sd_ref64 * sd_ref64 > thr2) / numel_a
        fn = xp.sum(sd_pred64 * sd_pred64 > thr2) / numel_b
        out[i] = (tp_a + tp_b) / (tp_a + tp_b + fp + fn + sys.float_info.min)
    return out
