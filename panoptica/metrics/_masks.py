"""Shared per-instance cropped masks — computed once, reused by every metric.

``precompute_crops`` derives, for each matched pair, the two boolean masks and
crops both to their padded union bbox, so every downstream metric operates on
a tiny sub-volume instead of re-deriving ``ref == id`` / ``pred == id`` over
the full volume. Metrics pull the result from ``params['_crops']`` via
``instance_masks``; when it is absent (standalone metric call / unit test)
they fall back to deriving masks from the full arrays.
"""

from __future__ import annotations

from panoptica.core.protocols import Array, Xp


def _find_objects(labels: Array, xp: Xp):
    """All per-label bounding boxes in ONE pass (scipy/cupyx find_objects)."""
    if xp.__name__ == "cupy":
        from cupyx.scipy.ndimage import find_objects
    else:
        from scipy.ndimage import find_objects
    return find_objects(labels)


def _union_padded(box_r, box_p, shape) -> tuple:
    """Union of two per-axis slice tuples, padded 1 voxel, clamped to shape."""
    out = []
    for axis in range(len(shape)):
        los, his = [], []
        for box in (box_r, box_p):
            if box is not None:
                los.append(box[axis].start)
                his.append(box[axis].stop)
        lo = min(los) - 1
        hi = max(his) + 1
        out.append(slice(max(lo, 0), min(hi, shape[axis])))
    return tuple(out)


def precompute_crops(
    ref: Array, pred: Array, ref_ids: Array, pred_ids: Array, xp: Xp
) -> list[tuple[Array, Array]]:
    """One (ref_bool_crop, pred_bool_crop) per matched instance, cropped to bbox.

    Bounding boxes for every label come from a single ``find_objects`` pass over
    each label array, so we never scan the full volume per instance -- only the
    small box region is compared (``ref[box] == id``). A pair with no voxels
    yields two empty arrays.
    """
    ref_objs = _find_objects(ref, xp)
    pred_objs = _find_objects(pred, xp)
    shape = ref.shape
    empty = xp.zeros((0,), dtype=bool)
    out: list[tuple[Array, Array]] = []
    for r_id, p_id in zip(ref_ids, pred_ids):
        box_r = ref_objs[int(r_id) - 1] if int(r_id) - 1 < len(ref_objs) else None
        box_p = pred_objs[int(p_id) - 1] if int(p_id) - 1 < len(pred_objs) else None
        if box_r is None and box_p is None:
            out.append((empty, empty))
            continue
        box = _union_padded(box_r, box_p, shape)
        out.append((ref[box] == r_id, pred[box] == p_id))
    return out


def instance_masks(
    ref: Array, pred: Array, ref_ids: Array, pred_ids: Array, xp: Xp, params
) -> list[tuple[Array, Array]]:
    """Precomputed crops from params, or full-volume masks (fallback)."""
    crops = params.get("_crops")
    if crops is not None:
        return crops
    return [(ref == ref_ids[i], pred == pred_ids[i]) for i in range(len(ref_ids))]
