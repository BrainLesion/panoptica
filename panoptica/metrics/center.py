"""Center-of-mass distance metric (CEDI).

Center of mass of a binary mask is the mean coordinate of its True voxels,
computed without mutating `ref`/`pred`.
"""

from __future__ import annotations

from panoptica.core.edge_cases import EdgeCaseResult
from panoptica.core.enums import Direction, MetricType
from panoptica.core.errors import InputValidationError
from panoptica.core.protocols import Array, Xp
from panoptica.metrics._masks import instance_masks
from panoptica.metrics.registry import ZeroTPPolicy, register


def _center_of_mass(mask: Array, xp: Xp) -> Array:
    """Mean coordinate of the True voxels of `mask`, per axis, float64.

    Pure: never mutates `mask`. Returns a (ndim,) float64 array; all-NaN if the
    mask is empty (mirrors scipy.ndimage.center_of_mass's behavior of dividing
    by zero total mass).
    """
    idx = xp.nonzero(mask)
    if idx[0].shape[0] == 0:
        return xp.full((mask.ndim,), xp.nan, dtype=xp.float64)
    return xp.asarray(
        [xp.mean(xp.asarray(axis_idx, dtype=xp.float64)) for axis_idx in idx],
        dtype=xp.float64,
    )


def _label_centroids(labels: Array, xp: Xp) -> Array:
    """``(max_label+1, ndim)`` centroid per label, NaN rows for absent labels.

    ``centroid[L, a] = mean over voxels of (labels==L) of coordinate a`` =
    ``bincount(lab, weights=coord_a) / bincount(lab)`` — one foreground pass per
    axis, no per-instance loop. Absent labels get count 0 → 0/0 → NaN, matching
    the per-instance "empty mask → NaN center" behaviour.
    """
    ndim = labels.ndim
    n = int(labels.max())
    nz = xp.nonzero(labels)  # ndim arrays of the foreground voxel coordinates
    lab = labels[nz].astype(xp.int64)
    count = xp.bincount(lab, minlength=n + 1).astype(xp.float64)
    cent = xp.empty((n + 1, ndim), dtype=xp.float64)
    for a in range(ndim):
        cent[:, a] = xp.bincount(lab, weights=nz[a].astype(xp.float64), minlength=n + 1)
    return cent / count[:, None]  # 0/0 → NaN for absent labels


def batched_center_distance(
    ref: Array,
    pred: Array,
    ref_ids: Array,
    pred_ids: Array,
    xp: Xp,
    spacing: tuple[float, ...] | None,
) -> Array:
    """Vectorized CEDI ``(K,)`` for the GPU: batched centroids + gather, no loop.

    Bit-equivalent to the per-instance path (same mean-coordinate centroids,
    same spacing scaling, same NaN-on-empty), but the two centroid tables are
    each built in one foreground pass and the K distances are a single gather +
    reduction instead of K crop iterations.
    """
    ndim = ref.ndim
    if spacing is not None and len(spacing) != ndim:
        raise InputValidationError(
            f"spacing must have same dimensionality as the input "
            f"(len(spacing)={len(spacing)}, ndim={ndim})"
        )
    ref_cent = _label_centroids(ref, xp)
    pred_cent = _label_centroids(pred, xp)
    r_idx = xp.asarray(ref_ids, dtype=xp.int64)
    p_idx = xp.asarray(pred_ids, dtype=xp.int64)
    diff = pred_cent[p_idx] - ref_cent[r_idx]  # (K, ndim); NaN if either absent
    if spacing is not None:
        diff = diff * xp.asarray(spacing, dtype=xp.float64)
    return xp.sqrt(xp.sum(diff * diff, axis=1))


@register(
    id="CEDI",
    type=MetricType.INSTANCE,
    direction=Direction.DECREASING,
    long_name="Center Distance",
    zero_tp=ZeroTPPolicy(default=EdgeCaseResult.NAN),
)
def center_distance_batched(
    ref: Array,
    pred: Array,
    ref_ids: Array,
    pred_ids: Array,
    xp: Xp,
    *,
    spacing: tuple[float, ...] | None = None,
    **params: object,
) -> Array:
    """Euclidean distance between the ref/pred centers of mass, per matched pair.

    NaN when a mask is empty (its center of mass is undefined).
    """
    if len(ref_ids) != len(pred_ids):
        raise InputValidationError(
            f"ref_ids and pred_ids must be positionally aligned "
            f"(got {len(ref_ids)} vs {len(pred_ids)})"
        )
    k = len(ref_ids)
    masks = instance_masks(ref, pred, ref_ids, pred_ids, xp, params)
    out = xp.empty((k,), dtype=xp.float64)
    for i in range(k):
        r, p = masks[i]
        ref_com = _center_of_mass(r, xp)
        pred_com = _center_of_mass(p, xp)
        diff = pred_com - ref_com
        if spacing is not None:
            if len(spacing) != diff.shape[0]:
                raise InputValidationError(
                    f"spacing must have same dimensionality as the input "
                    f"(len(spacing)={len(spacing)}, ndim={diff.shape[0]})"
                )
            diff = diff * xp.asarray(spacing, dtype=xp.float64)
        out[i] = xp.sqrt(xp.sum(diff * diff))
    return out
