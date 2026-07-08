"""Hand-computed tests for pipeline.approximate (connected-components extraction)."""

from __future__ import annotations

from panoptica.core.labels import LabelGroup
from panoptica.core.pairs import SemanticPair
from panoptica.pipeline.approximate import approximate


def test_two_blob_mask_yields_two_instances(xp):
    # Two disjoint 2x2 blobs in a 6x6 grid -> 2 connected components each side.
    ref = xp.zeros((6, 6), dtype=xp.int64)
    ref[0:2, 0:2] = 1
    ref[4:6, 4:6] = 1
    pred = xp.zeros((6, 6), dtype=xp.int64)
    pred[0:2, 0:2] = 1
    pred[4:6, 4:6] = 1

    pair = SemanticPair(ref=ref, pred=pred, spacing=None)
    out = approximate(pair, xp)

    assert out.n_ref == 2
    assert out.n_pred == 2
    assert int(xp.max(out.ref)) == 2
    assert int(xp.max(out.pred)) == 2


def test_empty_side_yields_zero_instances(xp):
    ref = xp.zeros((4, 4), dtype=xp.int64)
    pred = xp.zeros((4, 4), dtype=xp.int64)
    pred[1:3, 1:3] = 1

    pair = SemanticPair(ref=ref, pred=pred, spacing=None)
    out = approximate(pair, xp)

    assert out.n_ref == 0
    assert out.n_pred == 1
    assert bool(xp.all(out.ref == 0))


def test_label_group_restricts_foreground(xp):
    # Semantic mask with two label values; only label 2 belongs to the group.
    ref = xp.zeros((5, 5), dtype=xp.int64)
    ref[0:2, 0:2] = 1  # not in group
    ref[3:5, 3:5] = 2  # in group
    pred = xp.zeros((5, 5), dtype=xp.int64)
    pred[3:5, 3:5] = 2

    pair = SemanticPair(ref=ref, pred=pred, spacing=None)
    group = LabelGroup(value_labels=(2,))
    out = approximate(pair, xp, label_group=group)

    assert out.n_ref == 1
    assert out.n_pred == 1
    # The label-1 blob must not have been labelled as an instance.
    assert bool(xp.all(out.ref[0:2, 0:2] == 0))


def test_connectivity_controls_diagonal_merge(xp):
    # Two 2x2 blocks touching only at a diagonal corner: face-connectivity (1)
    # keeps them separate, full connectivity (2) merges them into one instance.
    ref = xp.zeros((4, 4), dtype=xp.int64)
    ref[0:2, 0:2] = 1
    ref[2:4, 2:4] = 1
    pair = SemanticPair(ref=ref, pred=ref, spacing=None)

    c1 = approximate(pair, xp, connectivity=1)
    assert c1.n_ref == 2
    c2 = approximate(pair, xp, connectivity=2)
    assert c2.n_ref == 1
