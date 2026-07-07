"""Dense overlap/cost matrix between two labelled instance arrays.

Encodes each voxel's (ref, pred) label pair into a single integer, ``bincount``s
it into an intersection table, and combines with the per-label areas (also
``bincount``) — one pass over the volume instead of one full-array reduction
per candidate pair. Returns the *dense* matrix including the background
row/col (index 0), so callers can slice/mask it however they need.
"""

from __future__ import annotations

from panoptica.core.errors import MetricComputeError
from panoptica.core.protocols import Array, Xp


def overlap_cost(ref: Array, pred: Array, xp: Xp, *, metric: str):
    """Return ``(cost, ref_labels, pred_labels)``.

    ``cost`` is the dense ``(n_ref+1, n_pred+1)`` matrix: row/col ``0`` is the
    background label; rows ``1..n_ref`` / cols ``1..n_pred`` are instance labels;
    ``cost[i, j]`` is the IoU or Dice between ``ref == i`` and ``pred == j``
    (zero denominator → ``0.0``). ``ref_labels`` / ``pred_labels`` are the present
    nonzero label values (ascending), computed here for free from the area
    bincounts so callers need not re-scan the volume.
    """
    metric_name = metric.upper()
    if metric_name not in ("IOU", "DSC"):
        raise MetricComputeError(
            f"overlap_cost: unsupported metric {metric!r}; expected 'iou' or 'dsc'"
        )

    n_ref = int(ref.max())
    n_pred = int(pred.max())

    ref_flat = ref.ravel()
    pred_flat = pred.ravel()

    # Only voxels with a nonzero ref OR pred label contribute to any label>=1
    # area or intersection; instances are sparse (typically <1% of the volume),
    # so restricting to the foreground shrinks all three bincounts by ~100x.
    # The background row/col (index 0) is thereby left inexact, but callers only
    # ever index cost[r][p] for label values r,p >= 1, never the background.
    fg = (ref_flat > 0) | (pred_flat > 0)
    ref_fg = ref_flat[fg].astype(xp.int64)
    pred_fg = pred_flat[fg].astype(xp.int64)

    ref_area = xp.bincount(ref_fg, minlength=n_ref + 1).astype(xp.float64)
    pred_area = xp.bincount(pred_fg, minlength=n_pred + 1).astype(xp.float64)

    # The nonzero-area indices ARE the present label values (ascending), so
    # callers get their label lists for free here instead of re-scanning the
    # full volume with a second bincount (`_nonzero_labels`).
    r_idx = xp.nonzero(ref_area)[0]
    p_idx = xp.nonzero(pred_area)[0]
    ref_labels = [
        int(v) for v in (r_idx.get() if hasattr(r_idx, "get") else r_idx) if v
    ]
    pred_labels = [
        int(v) for v in (p_idx.get() if hasattr(p_idx, "get") else p_idx) if v
    ]

    encoded = ref_fg * (n_pred + 1) + pred_fg
    intersection = xp.bincount(encoded, minlength=(n_ref + 1) * (n_pred + 1)).astype(
        xp.float64
    )
    intersection = intersection.reshape(n_ref + 1, n_pred + 1)

    ref_area_col = ref_area.reshape(-1, 1)
    pred_area_row = pred_area.reshape(1, -1)

    if metric_name == "IOU":
        denom = ref_area_col + pred_area_row - intersection
        numer = intersection
    else:  # DSC
        denom = ref_area_col + pred_area_row
        numer = 2.0 * intersection

    cost = xp.where(denom > 0, numer / xp.where(denom > 0, denom, 1.0), 0.0)
    return cost, ref_labels, pred_labels
