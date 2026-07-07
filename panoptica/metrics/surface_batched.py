"""Batched GPU surface distance (ASSD/HD/HD95/NSD).

The per-instance path (`surface.py`) runs, for every matched pair, an erosion +
EDT + boolean gather on a tiny crop. On GPU that is K serial launches of tiny
kernels — launch-bound, and the reason the GPU *loses* on many-small-instance
cases. This module removes the K loop entirely:

  A  global border extraction — one shift-based pass over the whole label array
     each for ref/pred (`labels != shift_zerofill(labels)` & foreground). This
     is bit-identical to the per-instance `mask XOR erosion` border (a fg voxel
     is a border voxel iff a face-neighbour carries a different label, i.e.
     background or another instance — exactly what erosion removes).
  B  directed nearest-neighbour by matmul — bucket instances by border size,
     pad to `(B, cap, ndim)`, and compute
     `d2 = |A|² + |B|² - 2·A·Bᵀ`; `sd_ref = sqrt(min_j d2)`, `sd_pred = sqrt(min_i d2)`.
     This *is* the EDT-gather distance (nearest opposite-border voxel), just
     re-expressed — exact for integer/isotropic grids, <1e-14 for float spacing.
  C  segment reductions — ASSD/HD/HD95/NSD computed over the padded bucket
     tensors with validity masks, no per-instance kernel.

Everything is gated behind ``xp.__name__ == "cupy"`` by the caller; the CPU
path keeps its crop-once loop (already the fastest CPU formulation).
"""

from __future__ import annotations

import sys

from panoptica.core.protocols import Array, Xp

# Match surface.py's degenerate sentinel (empty ref- or pred-border → NaN).
_EMPTY_SURFACE = float("nan")

_SURFACE_IDS = ("ASSD", "HD", "HD95", "NSD")


def _shift_zerofill(a: Array, axis: int, shift: int, xp: Xp) -> Array:
    """``a`` shifted by ``shift`` along ``axis``, vacated cells filled with 0."""
    out = xp.zeros_like(a)
    src: list = [slice(None)] * a.ndim
    dst: list = [slice(None)] * a.ndim
    if shift > 0:
        dst[axis] = slice(shift, None)
        src[axis] = slice(None, -shift)
    else:
        dst[axis] = slice(None, shift)
        src[axis] = slice(-shift, None)
    out[tuple(dst)] = a[tuple(src)]
    return out


def _border_points(labels: Array, xp: Xp, scale: Array):
    """Return ``(sorted_labels, sorted_points)`` for all border voxels.

    ``border = fg & OR_axes,±(labels != shift_zerofill(labels))`` — the global
    equivalent of per-instance ``mask XOR erosion``. Points are voxel coords
    scaled by voxel spacing (float32), sorted by label so a single
    ``searchsorted`` yields every instance's contiguous point block.
    """
    fg = labels != 0
    border = xp.zeros_like(fg)
    for axis in range(labels.ndim):
        for shift in (1, -1):
            border |= labels != _shift_zerofill(labels, axis, shift, xp)
    border &= fg

    coords = xp.nonzero(border)
    lbls = labels[border].astype(xp.int64)
    pts = xp.stack([c.astype(xp.float32) for c in coords], axis=1)
    pts = pts * scale  # (N, ndim), spacing baked in
    order = xp.argsort(lbls, kind="stable")
    return lbls[order], pts[order]


def _label_spans(sorted_lbls: Array, ids: Array, xp: Xp):
    """``(start, count)`` int64 arrays into ``sorted_*`` for each label in ``ids``."""
    lo = xp.searchsorted(sorted_lbls, ids, side="left")
    hi = xp.searchsorted(sorted_lbls, ids, side="right")
    return lo.astype(xp.int64), (hi - lo).astype(xp.int64)


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _pad_stack(
    sorted_pts: Array, starts: Array, counts: Array, cap: int, xp: Xp
) -> Array:
    """Gather ragged per-row point blocks into a padded ``(B, cap, ndim)`` tensor.

    Row ``i`` gets ``counts[i]`` real points (from ``sorted_pts[starts[i]:...]``)
    in positions ``0..counts[i]-1``; the rest are 0 (finite pad — validity is
    tracked by a separate mask, never by the coordinate value, so the matmul
    stays finite).
    """
    b = int(starts.shape[0])
    ndim = int(sorted_pts.shape[1])
    total = int(counts.sum())
    # Expand ragged rows without repeat(array) (unsupported on cupy): the row a
    # flat position belongs to is searchsorted into the count prefix-sum.
    csum = xp.cumsum(counts)
    row_start = csum - counts
    t = xp.arange(total)
    row_id = xp.searchsorted(csum, t, side="right")
    within = t - row_start[row_id]
    src = starts[row_id] + within
    dest = row_id * cap + within
    out = xp.zeros((b * cap, ndim), dtype=sorted_pts.dtype)
    out[dest] = sorted_pts[src]
    return out.reshape(b, cap, ndim)


def _bucketize(n_arr, m_arr, xp):
    """Group valid instances by ``pow2(max(n,m))``; return ``{cap: host_index_list}``.

    Instances with an empty ref- or pred-border (``n==0 or m==0``) are excluded
    (their result stays the NaN sentinel). One device→host sync total (both count
    vectors are pulled in a single stacked transfer).
    """
    nm = xp.stack((n_arr, m_arr))
    nm_host = nm.get() if hasattr(nm, "get") else nm
    n_host, m_host = nm_host[0], nm_host[1]
    buckets: dict[int, list[int]] = {}
    for k in range(len(n_host)):
        n, m = int(n_host[k]), int(m_host[k])
        if n == 0 or m == 0:
            continue
        buckets.setdefault(_next_pow2(max(n, m)), []).append(k)
    return buckets


def _percentile95(vals: Array, counts: Array, xp: Xp) -> Array:
    """Per-row 95th percentile (numpy 'linear' interpolation) over valid entries.

    ``vals`` is ``(B, W)`` with invalid slots set to +inf; ``counts`` the valid
    length per row. Sorting pushes the +inf pads to the end, so index ``r`` into
    the sorted row is the ``r``-th smallest real value.
    """
    s = xp.sort(vals, axis=1)
    c = counts.astype(xp.float64)
    rank = 0.95 * (c - 1.0)
    lo = xp.floor(rank).astype(xp.int64)
    hi = xp.ceil(rank).astype(xp.int64)
    frac = rank - lo.astype(xp.float64)
    rows = xp.arange(s.shape[0])
    v_lo = s[rows, lo]
    v_hi = s[rows, hi]
    return v_lo + frac * (v_hi - v_lo)


def batched_surface_scalars(
    ref: Array,
    pred: Array,
    ref_ids: Array,
    pred_ids: Array,
    xp: Xp,
    spacing: tuple[float, ...] | None,
    metrics,
    *,
    nsd_threshold: float | None = None,
) -> dict[str, Array]:
    """Batched ``{id: (K,) values}`` for the requested surface metrics (GPU).

    ``metrics`` is filtered to the surface family; the returned arrays align with
    ``ref_ids``/``pred_ids`` and carry NaN for degenerate (empty-border) pairs —
    identical semantics to the per-instance reducers in ``surface.py``.
    """
    want = [m for m in (mm.upper() for mm in metrics) if m in _SURFACE_IDS]
    k = int(ref_ids.shape[0])
    out = {m: xp.full((k,), _EMPTY_SURFACE, dtype=xp.float64) for m in want}
    if k == 0:
        return out

    ndim = ref.ndim
    sp = tuple(spacing) if spacing else (1.0,) * ndim
    scale = xp.asarray(sp, dtype=xp.float32).reshape(1, ndim)

    r_lbls, r_pts = _border_points(ref, xp, scale)
    p_lbls, p_pts = _border_points(pred, xp, scale)

    r_ids = xp.asarray(ref_ids, dtype=xp.int64)
    p_ids = xp.asarray(pred_ids, dtype=xp.int64)
    r_start, r_cnt = _label_spans(r_lbls, r_ids, xp)
    p_start, p_cnt = _label_spans(p_lbls, p_ids, xp)

    thr = nsd_threshold if nsd_threshold is not None else (min(sp) if spacing else 1.0)
    thr2 = float(thr) * float(thr)

    buckets = _bucketize(r_cnt, p_cnt, xp)
    for cap, idx_list in buckets.items():
        idx = xp.asarray(idx_list, dtype=xp.int64)
        _reduce_bucket(
            idx,
            cap,
            r_pts,
            p_pts,
            r_start,
            r_cnt,
            p_start,
            p_cnt,
            want,
            out,
            thr2,
            xp,
        )
    return out


def _reduce_bucket(
    idx, cap, r_pts, p_pts, r_start, r_cnt, p_start, p_cnt, want, out, thr2, xp
):
    """Stage B+C for one size bucket; scatter per-metric scalars into ``out``."""
    n = r_cnt[idx]
    m = p_cnt[idx]
    a = _pad_stack(r_pts, r_start[idx], n, cap, xp)  # (B, cap, ndim)
    b = _pad_stack(p_pts, p_start[idx], m, cap, xp)

    col = xp.arange(cap)
    valid_a = col.reshape(1, -1) < n.reshape(-1, 1)  # (B, cap)
    valid_b = col.reshape(1, -1) < m.reshape(-1, 1)

    a2 = (a * a).sum(axis=2)  # (B, cap)
    b2 = (b * b).sum(axis=2)
    cross = xp.matmul(a, xp.swapaxes(b, 1, 2))  # (B, cap, cap)
    d2 = a2[:, :, None] + b2[:, None, :] - 2.0 * cross
    pair_valid = valid_a[:, :, None] & valid_b[:, None, :]
    d2 = xp.where(pair_valid, d2, xp.inf)

    # Clip after the min (floor fp noise at 0), so the O(B·cap²) tensor is scanned
    # once for the reduction instead of an extra full clip pass; sqrt only ever
    # sees the per-row minimum, so min-then-clip == clip-then-min here.
    sd_ref = xp.sqrt(xp.clip(d2.min(axis=2), 0.0, None)).astype(xp.float64)  # (B, cap)
    sd_pred = xp.sqrt(xp.clip(d2.min(axis=1), 0.0, None)).astype(xp.float64)
    nf = n.astype(xp.float64)
    mf = m.astype(xp.float64)

    if "ASSD" in want:
        mean_ref = xp.where(valid_a, sd_ref, 0.0).sum(axis=1) / nf
        mean_pred = xp.where(valid_b, sd_pred, 0.0).sum(axis=1) / mf
        out["ASSD"][idx] = (mean_ref + mean_pred) / 2.0
    if "HD" in want:
        max_ref = xp.where(valid_a, sd_ref, -xp.inf).max(axis=1)
        max_pred = xp.where(valid_b, sd_pred, -xp.inf).max(axis=1)
        out["HD"][idx] = xp.maximum(max_ref, max_pred)
    if "NSD" in want:
        ra2 = sd_ref * sd_ref
        rb2 = sd_pred * sd_pred
        tp_a = (valid_a & (ra2 <= thr2)).sum(axis=1) / nf
        tp_b = (valid_b & (rb2 <= thr2)).sum(axis=1) / mf
        fp = (valid_a & (ra2 > thr2)).sum(axis=1) / nf
        fn = (valid_b & (rb2 > thr2)).sum(axis=1) / mf
        out["NSD"][idx] = (tp_a + tp_b) / (tp_a + tp_b + fp + fn + sys.float_info.min)
    if "HD95" in want:
        pooled = xp.concatenate(
            (xp.where(valid_a, sd_ref, xp.inf), xp.where(valid_b, sd_pred, xp.inf)),
            axis=1,
        )
        out["HD95"][idx] = _percentile95(pooled, n + m, xp)
