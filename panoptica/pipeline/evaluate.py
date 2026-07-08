"""Batched instance evaluation: MatchedInstancePair -> flat ResultDict.

Every requested metric's ``*_batched`` function is called exactly once over
*all* matched pairs (a ``(K,)`` array in, ``(K,)`` array out), instead of once
per instance per metric. Aggregation (``MetricMode``), detection quality
(TP/FP/FN/prec/rec/RQ), and per-metric SQ/PQ all consume that single batched
result.
"""

from __future__ import annotations

from panoptica.core.enums import MetricMode
from panoptica.core.pairs import EvaluateInstancePair, MatchedInstancePair
from panoptica.core.protocols import Array, Xp
from panoptica.metrics.registry import (
    aggregate,
    derive_detection_quality,
    derive_pq,
    derive_sq,
    get_spec,
)

#: Aggregation modes captured for every requested instance metric.
_MODES = (
    MetricMode.ALL,
    MetricMode.AVG,
    MetricMode.STD,
    MetricMode.MIN,
    MetricMode.MAX,
    MetricMode.SUM,
)


def _build_evaluate_pair(pair: MatchedInstancePair, xp: Xp) -> EvaluateInstancePair:
    k = int(pair.matched_ids.shape[0]) if pair.matched_ids is not None else 0
    if k > 0:
        matched = xp.asarray(pair.matched_ids)
        ref_ids = matched[:, 0]
        pred_ids = matched[:, 1]
    else:
        ref_ids = xp.zeros((0,), dtype=xp.int64)
        pred_ids = xp.zeros((0,), dtype=xp.int64)

    tp = k
    fn = len(pair.unmatched_ref)
    fp = len(pair.unmatched_pred)
    n_ref = tp + fn
    n_pred = tp + fp

    return EvaluateInstancePair(
        ref=pair.ref,
        pred=pair.pred,
        ref_ids=ref_ids,
        pred_ids=pred_ids,
        n_ref=n_ref,
        n_pred=n_pred,
        tp=tp,
        spacing=pair.spacing,
    )


_GLOBAL_VOLUMETRIC = {"DSC", "IOU", "RVD", "RVAE"}
_SURFACE = {"ASSD", "HD", "HD95", "NSD"}
_VOL = {"DSC", "IOU", "RVD", "RVAE"}


def _global_bin_metric(
    pair: MatchedInstancePair, xp: Xp, metric_id: str, spacing
) -> float:
    """Whole-volume binarized metric (entire ref/pred as a single instance)."""
    spec = get_spec(metric_id)
    ref_b = pair.ref != 0
    pred_b = pair.pred != 0
    mid = metric_id.upper()

    # Volumetric global metrics reduce to a few count_nonzero passes -- no need to
    # materialize label arrays or run bincount over the whole volume.
    if mid in _GLOBAL_VOLUMETRIC:
        r = float(xp.count_nonzero(ref_b))
        p = float(xp.count_nonzero(pred_b))
        if r == 0.0 and p == 0.0:
            return spec.zero_tp.resolve(n_ref=0, n_pred=0)
        if mid == "RVD":
            return (p - r) / r
        if mid == "RVAE":
            return abs((p - r) / r)
        inter = float(xp.count_nonzero(ref_b & pred_b))
        if mid == "DSC":
            denom = r + p
            return 0.0 if denom == 0 else 2.0 * inter / denom
        union = r + p - inter
        return 0.0 if union == 0 else inter / union

    if not bool(xp.any(ref_b)) and not bool(xp.any(pred_b)):
        return spec.zero_tp.resolve(n_ref=0, n_pred=0)
    ref_bin = ref_b.astype(xp.int64)
    pred_bin = pred_b.astype(xp.int64)
    ref_ids = xp.asarray([1], dtype=xp.int64)
    pred_ids = xp.asarray([1], dtype=xp.int64)
    values = spec.fn(ref_bin, pred_bin, ref_ids, pred_ids, xp, spacing=spacing)
    return float(values[0])


def instance_values(
    eval_pair: EvaluateInstancePair,
    metrics: list[str],
    xp: Xp,
    *,
    spacing: tuple[float, ...] | None = None,
    n_jobs: int = 1,
    metric_params: dict[str, dict[str, object]] | None = None,
) -> dict[str, Array]:
    """Per-instance metric value arrays, one batched ``spec.fn`` call per metric.

    Returns ``{metric_id: values (K,)}`` aligned with ``eval_pair.ref_ids``.
    The expensive shared precomputes (crop-once, EDT distance pairs, the
    vectorized volumetric bincount) are done ONCE here and reused across every
    requested metric -- and, in the grouped path, across every group.
    """
    metric_params = metric_params or {}
    out: dict[str, Array] = {}
    if not metrics or eval_pair.tp <= 0:
        for metric_id in metrics:
            values = get_spec(metric_id).fn(
                eval_pair.ref,
                eval_pair.pred,
                eval_pair.ref_ids,
                eval_pair.pred_ids,
                xp,
                spacing=spacing,
                **dict(metric_params.get(metric_id, {})),
            )
            out[metric_id] = values
        return out

    from panoptica.metrics._masks import precompute_crops

    crops = precompute_crops(
        eval_pair.ref, eval_pair.pred, eval_pair.ref_ids, eval_pair.pred_ids, xp
    )
    surface_ids = [m for m in metrics if m.upper() in _SURFACE]
    # GPU: the per-instance EDT/erosion loop is launch-bound (see
    # surface_batched). Replace it with one batched matmul-NN
    # pass that yields the final (K,) scalars directly. CPU keeps the crop loop.
    surface_scalars = None
    shared_sd = None
    if surface_ids and xp.__name__ == "cupy":
        from panoptica.metrics.surface_batched import batched_surface_scalars

        nsd_thr = dict(metric_params.get("NSD", {})).get("threshold")
        surface_scalars = batched_surface_scalars(
            eval_pair.ref,
            eval_pair.pred,
            eval_pair.ref_ids,
            eval_pair.pred_ids,
            xp,
            spacing,
            surface_ids,
            nsd_threshold=nsd_thr,  # pyrefly: ignore
        )
    elif surface_ids:
        from panoptica.metrics.surface import surface_distances_from_crops

        shared_sd = surface_distances_from_crops(crops, xp, spacing, n_jobs=n_jobs)

    # GPU: CEDI's per-instance crop loop is launch-bound too; batch the centroids
    # (one bincount per axis) into the (K,) distances directly. CPU keeps its loop.
    cedi_values = None
    if any(m.upper() == "CEDI" for m in metrics) and xp.__name__ == "cupy":
        from panoptica.metrics.center import batched_center_distance

        cedi_values = batched_center_distance(
            eval_pair.ref,
            eval_pair.pred,
            eval_pair.ref_ids,
            eval_pair.pred_ids,
            xp,
            spacing,
        )

    vol_stats = None
    if any(m.upper() in _VOL for m in metrics):
        from panoptica.metrics.volumetric import volume_stats

        vol_stats = volume_stats(
            eval_pair.ref, eval_pair.pred, eval_pair.ref_ids, eval_pair.pred_ids, xp
        )

    for metric_id in metrics:
        mid = metric_id.upper()
        # GPU batched paths already hold the final (K,) array — assign it directly
        # rather than laundering it back through the reducer.
        if surface_scalars is not None and mid in _SURFACE:
            out[metric_id] = surface_scalars[mid]
            continue
        if cedi_values is not None and mid == "CEDI":
            out[metric_id] = cedi_values
            continue
        params = dict(metric_params.get(metric_id, {}))
        params["_crops"] = crops
        if shared_sd is not None and mid in _SURFACE:
            params["_sd_pairs"] = shared_sd
        if vol_stats is not None and mid in _VOL:
            params["_vol_stats"] = vol_stats
        if mid == "CLDSC":
            params["_n_jobs"] = n_jobs
        out[metric_id] = get_spec(metric_id).fn(
            eval_pair.ref,
            eval_pair.pred,
            eval_pair.ref_ids,
            eval_pair.pred_ids,
            xp,
            spacing=spacing,
            **params,
        )
    return out


def aggregate_metric(
    result: dict,
    metric_id: str,
    values,
    xp: Xp,
    *,
    tp: int,
    n_ref: int,
    n_pred: int,
    rq: float,
) -> None:
    """Write ``<id>_{all,avg,std,...}`` and SQ/PQ for one metric into ``result``."""
    key_prefix = metric_id.lower()
    for mode in _MODES:
        result[f"{key_prefix}_{mode.name.lower()}"] = aggregate(values, mode, xp)

    sq = derive_sq(metric_id, values, xp, tp=tp, n_ref=n_ref, n_pred=n_pred)
    pq = derive_pq(sq, rq)
    sq_std = result[f"{key_prefix}_std"]

    if metric_id.upper() == "IOU":
        result["sq"] = sq
        result["sq_std"] = sq_std
        result["pq"] = pq
    else:
        result[f"sq_{key_prefix}"] = sq
        result[f"sq_{key_prefix}_std"] = sq_std
        result[f"pq_{key_prefix}"] = pq


def detection_result(tp: int, fp: int, fn: int) -> dict:
    """Base result dict with the detection-quality block filled in."""
    dq = derive_detection_quality(tp, fp, fn)
    return {
        "n_ref_instances": tp + fn,
        "n_pred_instances": tp + fp,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "prec": dq.prec,
        "rec": dq.rec,
        "rq": dq.rq,
    }


def evaluate(
    pair: MatchedInstancePair,
    metrics: list[str],
    xp: Xp,
    *,
    spacing: tuple[float, ...] | None = None,
    global_metrics: list[str] | None = None,
    metric_params: dict[str, dict[str, object]] | None = None,
    **cfg: object,
) -> dict:
    """Evaluate `pair` over `metrics`, returning a flat dict.

    Keys emitted: ``n_ref_instances``, ``n_pred_instances``, ``tp``, ``fp``,
    ``fn``, ``prec``, ``rec``, ``rq``; per requested metric ``<id>_all/avg/
    sum/std/min/max``; per-metric detection-derived ``sq``/``sq_std``/``pq``
    (unsuffixed for the default ``IOU`` metric) or
    ``sq_<id>``/``sq_<id>_std``/``pq_<id>`` for every other metric; and, for
    each id in `global_metrics`, ``global_bin_<id>``.
    """
    spacing = spacing if spacing is not None else pair.spacing
    metric_params = metric_params or {}
    n_jobs = int(cfg.get("n_jobs", 1) or 1)  # pyrefly: ignore

    eval_pair = _build_evaluate_pair(pair, xp)
    tp, fp, fn = eval_pair.tp, len(pair.unmatched_pred), len(pair.unmatched_ref)
    n_ref, n_pred = eval_pair.n_ref, eval_pair.n_pred

    result = detection_result(tp, fp, fn)
    rq = result["rq"]

    values_by_metric = instance_values(
        eval_pair,
        metrics,
        xp,
        spacing=spacing,
        n_jobs=n_jobs,
        metric_params=metric_params,
    )
    for metric_id in metrics:
        aggregate_metric(
            result,
            metric_id,
            values_by_metric[metric_id],
            xp,
            tp=tp,
            n_ref=n_ref,
            n_pred=n_pred,
            rq=rq,
        )

    for gid in global_metrics or []:
        result[f"global_bin_{gid.lower()}"] = _global_bin_metric(pair, xp, gid, spacing)

    return result


__all__ = [
    "evaluate",
    "instance_values",
    "aggregate_metric",
    "detection_result",
    "_build_evaluate_pair",
    "_global_bin_metric",
]
