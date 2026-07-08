"""Instance matching: UnmatchedInstancePair -> MatchedInstancePair.

Three matching algorithms:

- ``naive``: greedy, highest-score-first, one-to-one unless
  ``allow_many_to_one``.
- ``bipartite``: optimal one-to-one assignment via the Hungarian algorithm
  (``scipy.optimize.linear_sum_assignment``; small matrices, lazily imported).
- ``merge``: many-to-one, merges an extra prediction into an already-matched
  reference whenever the *combined* overlap score exceeds the reference's
  current best score.

All three build their candidate scores from the dense
``kernels.overlap_cost`` matrix rather than a per-pair Python metric call.
"""

from __future__ import annotations

from panoptica.core.errors import InputValidationError
from panoptica.core.pairs import MatchedInstancePair, UnmatchedInstancePair
from panoptica.core.protocols import Array, Xp
from panoptica.kernels.overlap import overlap_cost
from panoptica.kernels.relabel import map_labels

#: Direction of each matching metric this stage supports.
_METRIC_INCREASING: dict[str, bool] = {"IOU": True, "DSC": True}


def _increasing(metric: str) -> bool:
    return _METRIC_INCREASING.get(metric.upper(), True)


def _score_beats_threshold(
    metric: str, score: float, threshold: float, strict: bool = False
) -> bool:
    """Whether ``score`` passes ``threshold`` for this metric's direction.

    ``strict`` uses ``>`` / ``<`` instead of ``>=`` / ``<=``, so a score exactly
    equal to the threshold does NOT pass — e.g. ``threshold=0`` then means "any
    real overlap" rather than "everything".
    """
    if _increasing(metric):
        return score > threshold if strict else score >= threshold
    return score < threshold if strict else score <= threshold


def _nonzero_labels(arr: Array, xp: Xp) -> list[int]:
    """Nonzero instance label values, ascending. Labels may be non-contiguous.

    Uses bincount (linear) rather than unique (which sorts the whole array).
    """
    counts = xp.bincount(arr.reshape(-1))
    idx = xp.nonzero(counts)[0]
    idx = idx[idx != 0]
    return [int(v) for v in (idx.tolist() if hasattr(idx, "tolist") else idx)]


def _candidate_pairs(
    cost: Array,
    ref_labels: list[int],
    pred_labels: list[int],
    groups: _GroupConstraint | None = None,
) -> list[tuple[float, int, int]]:
    """Dense cost matrix -> sorted (score, ref_id, pred_id) candidates, score > 0 only.

    ``cost`` is indexed by actual label value (``overlap_cost`` sizes it to
    ``max_label + 1``), so iteration ranges over the real label lists. When a
    ``groups`` constraint is given, cross-group pairs are dropped so matching a
    reference instance only ever considers predictions in the *same* class
    group (the vectorized multi-group path relies on this).
    """
    # Pull the whole cost matrix to host ONCE. Indexing ``float(cost[r][p])`` per
    # pair would otherwise trigger a device->host sync per candidate (thousands
    # on GPU); the values are identical either way.
    cost = cost.get() if hasattr(cost, "get") else cost
    pairs: list[tuple[float, int, int]] = []
    for r in ref_labels:
        row = cost[r]
        gr = groups.ref[r] if groups is not None else 0
        for p in pred_labels:
            if groups is not None and gr != groups.pred[p]:
                continue
            score = float(row[p])
            if score > 0.0:
                pairs.append((score, r, p))
    pairs.sort(key=lambda t: t[0], reverse=True)
    return pairs


class _GroupConstraint:
    """Per-label class-group tags used to keep matching within a class group.

    ``ref``/``pred`` are dense host lists indexed by label value (0 = background
    / ungrouped); ``force`` is the set of group indices whose matching threshold
    is overridden to 0 (v1 ``single_instance`` groups match on any overlap).
    """

    __slots__ = ("ref", "pred", "force")

    def __init__(self, ref: list[int], pred: list[int], force: set[int]) -> None:
        self.ref = ref
        self.pred = pred
        self.force = force

    def threshold(self, ref_label: int, default: float) -> float:
        return 0.0 if self.ref[ref_label] in self.force else default


def _build_matched_pair(
    pair: UnmatchedInstancePair,
    xp: Xp,
    matched_ids: list[tuple[int, int]],
    ref_labels: list[int] | None = None,
    pred_labels: list[int] | None = None,
) -> MatchedInstancePair:
    """Relabel `pair.pred` so matched pred ids equal their ref partner's id.

    Matched pred labels are remapped to their ref id; leftover (unmatched)
    pred labels are assigned new labels sequentially after the highest ref
    label so no collisions occur. Operates on the ACTUAL (possibly
    non-contiguous) label values, not on instance counts.
    """
    if ref_labels is None:
        ref_labels = _nonzero_labels(pair.ref, xp)
    if pred_labels is None:
        pred_labels = _nonzero_labels(pair.pred, xp)

    matched_ref_ids = {r for r, _p in matched_ids}
    matched_pred_ids = {p for _r, p in matched_ids}

    unmatched_ref = tuple(sorted(set(ref_labels) - matched_ref_ids))
    unmatched_pred_orig = sorted(set(pred_labels) - matched_pred_ids)

    mapping: dict[int, int] = {p: r for r, p in matched_ids}
    # Next free label above the highest reference id, so remapped FP preds
    # never collide with a ref id.
    label_counter = (max(ref_labels) + 1) if ref_labels else 1
    for p in unmatched_pred_orig:
        mapping[p] = label_counter
        label_counter += 1

    pred_relabeled = (
        map_labels(pair.pred, mapping, xp)
        if mapping
        else xp.array(pair.pred, copy=True)
    )
    unmatched_pred = tuple(mapping[p] for p in unmatched_pred_orig)

    if matched_ids:
        # Reference the relabeled pred id (mapping[p]), not the original p.
        remapped = sorted((r, mapping[p]) for r, p in matched_ids)
        matched_ids_arr = xp.asarray(remapped, dtype=xp.int64)
    else:
        matched_ids_arr = xp.zeros((0, 2), dtype=xp.int64)

    return MatchedInstancePair(
        ref=pair.ref,
        pred=pred_relabeled,
        matched_ids=matched_ids_arr,
        unmatched_ref=unmatched_ref,
        unmatched_pred=unmatched_pred,
        spacing=pair.spacing,
    )


def _naive_match(
    pair: UnmatchedInstancePair,
    xp: Xp,
    metric: str,
    threshold: float,
    allow_many_to_one: bool,
    groups: _GroupConstraint | None = None,
    strict: bool = False,
) -> MatchedInstancePair:
    cost, ref_labels, pred_labels = overlap_cost(pair.ref, pair.pred, xp, metric=metric)
    candidates = _candidate_pairs(cost, ref_labels, pred_labels, groups)

    matched_pred: set[int] = set()
    matched_ref: set[int] = set()
    matched_ids: list[tuple[int, int]] = []

    for score, r, p in candidates:
        if p in matched_pred:
            continue
        if r in matched_ref and not allow_many_to_one:
            continue
        thr = groups.threshold(r, threshold) if groups is not None else threshold
        if _score_beats_threshold(metric, score, thr, strict):
            matched_ids.append((r, p))
            matched_pred.add(p)
            matched_ref.add(r)

    return _build_matched_pair(pair, xp, matched_ids, ref_labels, pred_labels)


def _bipartite_match(
    pair: UnmatchedInstancePair,
    xp: Xp,
    metric: str,
    threshold: float,
    groups: _GroupConstraint | None = None,
    strict: bool = False,
) -> MatchedInstancePair:
    if pair.n_ref == 0 or pair.n_pred == 0:
        return _build_matched_pair(pair, xp, [])

    import numpy as np
    from scipy.optimize import linear_sum_assignment

    cost, ref_labels, pred_labels = overlap_cost(pair.ref, pair.pred, xp, metric=metric)
    cost_host = cost.get() if hasattr(cost, "get") else np.asarray(cost)
    # Rows/cols of the assignment matrix index into the ACTUAL label values, so
    # gaps in the label numbering can never drop a real candidate.
    default_cost = 1.0 + 1e-6
    c = np.full((len(ref_labels), len(pred_labels)), default_cost, dtype=np.float64)
    increasing = _increasing(metric)
    for i, r in enumerate(ref_labels):
        gr = groups.ref[r] if groups is not None else 0
        thr = groups.threshold(r, threshold) if groups is not None else threshold
        for j, p in enumerate(pred_labels):
            if groups is not None and gr != groups.pred[p]:
                continue  # cross-group blocked -> stays at default (unassignable)
            score = float(cost_host[r, p])
            if _score_beats_threshold(metric, score, thr, strict):
                c[i, j] = 1.0 - score if increasing else score

    row_idx, col_idx = linear_sum_assignment(c)
    matched_ids = [
        (ref_labels[int(i)], pred_labels[int(j)])
        for i, j in zip(row_idx, col_idx)
        if c[i, j] < 1.0
    ]
    return _build_matched_pair(pair, xp, matched_ids, ref_labels, pred_labels)


def _combined_score(
    pair: UnmatchedInstancePair, xp: Xp, metric: str, ref_id: int, pred_ids: list[int]
) -> float:
    """Overlap score of `ref_id` against the union of `pred_ids`."""
    ref_mask = pair.ref == ref_id
    pred_mask = xp.isin(pair.pred, xp.asarray(pred_ids))
    r_sum = float(xp.sum(ref_mask))
    p_sum = float(xp.sum(pred_mask))
    inter = float(xp.sum(xp.logical_and(ref_mask, pred_mask)))
    if metric.upper() == "DSC":
        denom = r_sum + p_sum
        return 0.0 if denom == 0.0 else 2.0 * inter / denom
    union = r_sum + p_sum - inter
    return 0.0 if union == 0.0 else inter / union


def _merge_match(
    pair: UnmatchedInstancePair,
    xp: Xp,
    metric: str,
    threshold: float,
    groups: _GroupConstraint | None = None,
    strict: bool = False,
) -> MatchedInstancePair:
    cost, ref_labels, pred_labels = overlap_cost(pair.ref, pair.pred, xp, metric=metric)
    candidates = _candidate_pairs(cost, ref_labels, pred_labels, groups)

    matched_pred: set[int] = set()
    ref_to_preds: dict[int, list[int]] = {}
    ref_score: dict[int, float] = {}

    for score, r, p in candidates:
        if p in matched_pred:
            continue
        thr = groups.threshold(r, threshold) if groups is not None else threshold
        if r in ref_to_preds:
            candidate_preds = [*ref_to_preds[r], p]
            new_score = _combined_score(pair, xp, metric, r, candidate_preds)
            if new_score > ref_score[r]:
                ref_to_preds[r] = candidate_preds
                ref_score[r] = new_score
                matched_pred.add(p)
        elif _score_beats_threshold(metric, score, thr, strict):
            ref_to_preds[r] = [p]
            ref_score[r] = score
            matched_pred.add(p)

    matched_ids = [(r, p) for r, preds in ref_to_preds.items() for p in preds]
    return _build_matched_pair(pair, xp, matched_ids, ref_labels, pred_labels)


def match(
    pair: UnmatchedInstancePair,
    xp: Xp,
    *,
    algorithm: str = "naive",
    matching_metric: str = "IOU",
    matching_threshold: float = 0.5,
    allow_many_to_one: bool = False,
    groups: _GroupConstraint | None = None,
    strict: bool = False,
    **cfg: object,
) -> MatchedInstancePair:
    """Match `pair`'s reference and prediction instances.

    `algorithm`: ``"naive"`` (default, greedy threshold matching), ``"bipartite"``
    (Hungarian algorithm), or ``"merge"`` (many-to-one merge matching).

    ``strict`` uses a strict threshold comparison (``>`` / ``<`` instead of
    ``>=`` / ``<=``), so ``matching_threshold=0`` rejects zero-overlap pairs.

    When ``groups`` is given, matching is confined to within class groups (the
    vectorized multi-group path). Because cross-group scores are dropped, the
    per-group matchings decompose independently and equal running each matcher
    on a group-masked volume, so v1 parity is preserved.
    """
    algo = algorithm.lower()
    if algo == "naive":
        return _naive_match(
            pair,
            xp,
            matching_metric,
            matching_threshold,
            allow_many_to_one,
            groups,
            strict,
        )
    if algo in ("bipartite", "max_bipartite", "hungarian"):
        return _bipartite_match(
            pair, xp, matching_metric, matching_threshold, groups, strict
        )
    if algo in ("merge", "maximize_merge"):
        return _merge_match(
            pair, xp, matching_metric, matching_threshold, groups, strict
        )
    raise InputValidationError(f"Unknown matcher algorithm: {algorithm!r}")


__all__ = ["match", "_GroupConstraint"]
