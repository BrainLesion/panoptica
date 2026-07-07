"""Vectorized multi-group (SegmentationClassGroups) evaluation.

v1 evaluates each class group by masking the volume to the group's labels and
re-running the *whole* pipeline per group. That repeats connected components,
the distance transforms and the volumetric passes once per group.

This module does it the vectorized way instead:

1. Derive instances ONCE over the whole volume (CC for SEMANTIC input; the
   given labels otherwise) and tag every instance with its class group.
2. Match ONCE with a same-group constraint (:class:`match._GroupConstraint`).
   Because cross-group overlap scores are dropped, the per-group matchings are
   independent and identical to matching each group-masked volume separately,
   so v1 parity holds.
3. Evaluate ALL matched instances in a single batched pass -- the crop-once
   masks, shared EDT distance pairs and the volumetric bincount are computed
   over every group's instances together.
4. Reduce per group: partition the per-instance value arrays by group tag and
   aggregate (detection quality, SQ/PQ, global binarized metrics) for each.

Returns ``{group_name: result-dict}``. ``single_instance`` groups (v1) match on
any overlap (threshold 0); they only occur with instance input (no CC), so the
group's single label is one instance per side.
"""

from __future__ import annotations

import warnings

from panoptica.core.enums import InputType
from panoptica.core.labels import SegmentationClassGroups
from panoptica.core.pairs import (
    MatchedInstancePair,
    SemanticPair,
    UnmatchedInstancePair,
)
from panoptica.core.protocols import Array, Xp
from panoptica.pipeline.approximate import approximate
from panoptica.pipeline.evaluate import (
    _build_evaluate_pair,
    _global_bin_metric,
    aggregate_metric,
    detection_result,
    instance_values,
)
from panoptica.pipeline.match import _GroupConstraint, match


def _group_list(
    sg: SegmentationClassGroups,
) -> list[tuple[str, tuple[int, ...], bool]]:
    """Ordered ``(lowercased-name, value_labels, single_instance)`` tuples."""
    out: list[tuple[str, tuple[int, ...], bool]] = []
    for name, lg in sg.groups.items():
        out.append(
            (
                str(name).lower(),
                tuple(int(x) for x in lg.value_labels),
                lg.single_instance,
            )
        )
    return out


def _label_to_group_dense(
    groups: list[tuple[str, tuple[int, ...], bool]], size: int, xp: Xp
) -> Array:
    """Dense array indexed by ORIGINAL label -> 1-based group index (0 = none)."""
    dense = xp.zeros(size, dtype=xp.int64)
    for gi, (_name, labels, _single) in enumerate(groups, start=1):
        for label in labels:
            if 0 <= label < size:
                dense[label] = gi
    return dense


def _tags_from_scatter(inst: Array, group_per_voxel: Array, xp: Xp) -> Array:
    """Group tag per instance label: scatter each voxel's group onto its label.

    Every instance is single-labelled (CC respects label boundaries; instance
    input keeps original labels), so any voxel of an instance carries the
    instance's group -- last-write-wins is exact.
    """
    size = int(inst.max()) + 1
    tags = xp.zeros(size, dtype=xp.int64)
    tags[inst.reshape(-1)] = group_per_voxel
    return tags


def _host(arr: Array) -> list[int]:
    a = arr.get() if hasattr(arr, "get") else arr
    return [int(x) for x in a.tolist()]


def _matched_from_correspondence(
    ref: Array, pred: Array, xp: Xp, spacing
) -> MatchedInstancePair:
    """MATCHED input: labels already correspond, so pair by identical id."""
    from panoptica.pipeline.match import _nonzero_labels

    rset = set(_nonzero_labels(ref, xp))
    pset = set(_nonzero_labels(pred, xp))
    both = sorted(rset & pset)
    matched = (
        xp.asarray([[i, i] for i in both], dtype=xp.int64)
        if both
        else xp.zeros((0, 2), dtype=xp.int64)
    )
    return MatchedInstancePair(
        ref=ref,
        pred=pred,
        matched_ids=matched,
        unmatched_ref=tuple(sorted(rset - pset)),
        unmatched_pred=tuple(sorted(pset - rset)),
        spacing=spacing,
    )


def evaluate_grouped(
    ref: Array,
    pred: Array,
    xp: Xp,
    *,
    input_type: InputType,
    groups: SegmentationClassGroups,
    instance_metrics: list[str],
    global_metrics: list[str],
    matcher: str,
    matching_metric: str,
    matching_threshold: float,
    connectivity: int | None,
    spacing: tuple[float, ...] | None,
    n_jobs: int,
    strict_threshold: bool = False,
    skip_groups: set[str] | None = None,
) -> dict[str, dict]:
    """Evaluate every class group and return ``{group_name: result-dict}``.

    ``skip_groups`` (lowercased names) are neither reduced nor returned.
    """
    skip = skip_groups or set()
    glist = _group_list(groups)
    force_groups = {gi for gi, (_n, _l, single) in enumerate(glist, start=1) if single}

    max_label = int(max(int(ref.max()), int(pred.max()))) + 1
    for _n, labels, _s in glist:
        max_label = max(max_label, (max(labels) + 1) if labels else 0)
    label_to_group = _label_to_group_dense(glist, max_label, xp)

    if input_type is InputType.SEMANTIC:
        if force_groups:
            warnings.warn(
                "single_instance groups are not supported with SEMANTIC input in panoptica; "
                "the group is connected-component labelled like any other."
            )
        unmatched = approximate(
            SemanticPair(ref=ref, pred=pred, spacing=spacing),
            xp,
            connectivity=connectivity,
        )
        ref_match, pred_match = unmatched.ref, unmatched.pred
        ref_tags = _tags_from_scatter(ref_match, label_to_group[ref.reshape(-1)], xp)
        pred_tags = _tags_from_scatter(pred_match, label_to_group[pred.reshape(-1)], xp)
    else:
        # Instance input: instances ARE the original labels (no CC).
        ref_match, pred_match = ref, pred
        ref_tags = label_to_group
        pred_tags = label_to_group

    if input_type is InputType.MATCHED_INSTANCE:
        m = _matched_from_correspondence(ref_match, pred_match, xp, spacing)
    else:
        n_ref = int(xp.count_nonzero(xp.bincount(ref_match.reshape(-1))[1:] > 0))
        n_pred = int(xp.count_nonzero(xp.bincount(pred_match.reshape(-1))[1:] > 0))
        constraint = _GroupConstraint(_host(ref_tags), _host(pred_tags), force_groups)
        m = match(
            UnmatchedInstancePair(
                ref=ref_match,
                pred=pred_match,
                n_ref=n_ref,
                n_pred=n_pred,
                spacing=spacing,
            ),
            xp,
            algorithm=matcher,
            matching_metric=matching_metric,
            matching_threshold=matching_threshold,
            groups=constraint,
            strict=strict_threshold,
        )

    # Group of every FINAL label. ref is never relabelled, so ref_tags applies
    # directly. pred is relabelled by matching -> recover each final pred label's
    # group by scattering the pre-relabel (matching) pred's group per voxel.
    grp_final_ref = ref_tags
    grp_pred_voxel = pred_tags[pred_match.reshape(-1)]
    grp_final_pred = _tags_from_scatter(m.pred, grp_pred_voxel, xp)

    eval_pair = _build_evaluate_pair(m, xp)
    values_by_metric = instance_values(
        eval_pair,
        instance_metrics,
        xp,
        spacing=spacing,
        n_jobs=n_jobs,
    )

    # Host-side group tags for the cheap scalar reductions.
    ref_ids_host = _host(eval_pair.ref_ids)
    grp_ref_host = _host(grp_final_ref)
    grp_pred_host = _host(grp_final_pred)
    pair_group = [grp_ref_host[r] for r in ref_ids_host]

    results: dict[str, dict] = {}
    for gi, (name, labels, _single) in enumerate(glist, start=1):
        if name in skip:
            continue
        mask = xp.asarray([pg == gi for pg in pair_group], dtype=bool)
        tp_g = int(mask.sum()) if mask.size else 0
        fn_g = sum(1 for r in m.unmatched_ref if grp_ref_host[r] == gi)
        fp_g = sum(1 for p in m.unmatched_pred if grp_pred_host[p] == gi)

        result = detection_result(tp_g, fp_g, fn_g)
        rq = result["rq"]
        n_ref_g, n_pred_g = tp_g + fn_g, tp_g + fp_g
        for metric_id in instance_metrics:
            values_g = values_by_metric[metric_id]
            values_g = values_g[mask] if getattr(values_g, "size", 0) else values_g
            aggregate_metric(
                result,
                metric_id,
                values_g,
                xp,
                tp=tp_g,
                n_ref=n_ref_g,
                n_pred=n_pred_g,
                rq=rq,
            )

        group_labels = xp.asarray(labels, dtype=xp.int64)
        bin_pair = MatchedInstancePair(
            ref=xp.isin(ref, group_labels),
            pred=xp.isin(pred, group_labels),
            matched_ids=eval_pair.ref_ids,
            spacing=spacing,
        )
        for gid in global_metrics or []:
            result[f"global_bin_{gid.lower()}"] = _global_bin_metric(
                bin_pair, xp, gid, spacing
            )
        results[name] = result

    return results


__all__ = ["evaluate_grouped"]
