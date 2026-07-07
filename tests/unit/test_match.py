"""Hand-computed tests for pipeline.match (naive threshold + bipartite matching)."""

from __future__ import annotations

import pytest

from panoptica.core.pairs import UnmatchedInstancePair
from panoptica.pipeline.match import match


def _two_instance_pair(xp):
    # ref: instance 1 top-left 2x2, instance 2 bottom-right 2x2.
    ref = xp.zeros((6, 6), dtype=xp.int64)
    ref[0:2, 0:2] = 1
    ref[4:6, 4:6] = 2
    # pred: instance 1 overlaps ref-1 fully (IoU=1.0), instance 2 overlaps ref-2
    # in only 1 of its 4 voxels (IoU = 1/4 = 0.25, below the 0.5 default threshold).
    pred = xp.zeros((6, 6), dtype=xp.int64)
    pred[0:2, 0:2] = 1
    pred[4:5, 4:5] = 2
    return UnmatchedInstancePair(ref=ref, pred=pred, n_ref=2, n_pred=2, spacing=None)


def test_naive_threshold_matching(xp):
    pair = _two_instance_pair(xp)
    out = match(
        pair, xp, algorithm="naive", matching_metric="IOU", matching_threshold=0.5
    )

    matched = {(int(r), int(p)) for r, p in out.matched_ids.tolist()}
    assert matched == {(1, 1)}
    assert out.unmatched_ref == (2,)
    assert set(out.unmatched_pred) == {2} or len(out.unmatched_pred) == 1
    # matched pred relabeled to its ref id
    assert bool(xp.all(out.pred[0:2, 0:2] == 1))


def test_bipartite_matching_same_result_on_simple_case(xp):
    pair = _two_instance_pair(xp)
    out = match(
        pair, xp, algorithm="bipartite", matching_metric="IOU", matching_threshold=0.5
    )
    matched = {(int(r), int(p)) for r, p in out.matched_ids.tolist()}
    assert matched == {(1, 1)}
    assert out.unmatched_ref == (2,)


def test_no_overlap_no_matches(xp):
    ref = xp.zeros((4, 4), dtype=xp.int64)
    ref[0:2, 0:2] = 1
    pred = xp.zeros((4, 4), dtype=xp.int64)
    pred[2:4, 2:4] = 1
    pair = UnmatchedInstancePair(ref=ref, pred=pred, n_ref=1, n_pred=1, spacing=None)

    out = match(pair, xp, algorithm="naive")
    assert out.matched_ids.shape[0] == 0
    assert out.unmatched_ref == (1,)
    assert len(out.unmatched_pred) == 1


def _exact_half_iou_pair(xp):
    # ref: 3-voxel L (area 3). pred: 3-voxel row sharing 2 -> inter 2, union 4,
    # IoU = 0.5 exactly, so it sits right on the default threshold.
    ref = xp.zeros((4, 4), dtype=xp.int64)
    ref[0, 0] = ref[0, 1] = ref[1, 0] = 1
    pred = xp.zeros((4, 4), dtype=xp.int64)
    pred[0, 0] = pred[0, 1] = pred[0, 2] = 1
    return UnmatchedInstancePair(ref=ref, pred=pred, n_ref=1, n_pred=1, spacing=None)


@pytest.mark.parametrize("algo", ["naive", "bipartite", "merge"])
def test_strict_threshold_rejects_score_equal_to_threshold(xp, algo):
    # At IoU == threshold, lenient (>=) matches, strict (>) does not — for all matchers.
    pair = _exact_half_iou_pair(xp)
    lenient = match(pair, xp, algorithm=algo, matching_threshold=0.5, strict=False)
    strict = match(pair, xp, algorithm=algo, matching_threshold=0.5, strict=True)
    assert lenient.matched_ids.shape[0] == 1
    assert strict.matched_ids.shape[0] == 0


def test_merge_matching_combines_two_fragments(xp):
    # One ref block; pred split into two fragments each covering half of it.
    # Each fragment alone is IoU 0.5; merged they cover the ref (IoU 1.0), so the
    # merge matcher attaches both — where naive (one-to-one) keeps only one.
    ref = xp.zeros((4, 6), dtype=xp.int64)
    ref[0:4, 0:4] = 1
    pred = xp.zeros((4, 6), dtype=xp.int64)
    pred[0:4, 0:2] = 1
    pred[0:4, 2:4] = 2
    pair = UnmatchedInstancePair(ref=ref, pred=pred, n_ref=1, n_pred=2, spacing=None)

    merged = match(
        pair, xp, algorithm="merge", matching_metric="IOU", matching_threshold=0.5
    )
    naive = match(
        pair, xp, algorithm="naive", matching_metric="IOU", matching_threshold=0.5
    )

    # merge attaches both fragments to ref 1 (both relabelled to id 1); naive one.
    assert merged.matched_ids.shape[0] == 2
    assert {int(r) for r, _ in merged.matched_ids.tolist()} == {1}
    assert not merged.unmatched_pred
    assert naive.matched_ids.shape[0] == 1
