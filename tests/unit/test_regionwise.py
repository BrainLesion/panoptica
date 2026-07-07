"""Test for pipeline.regionwise per-region (Voronoi) evaluation orchestration."""

from __future__ import annotations

from panoptica.core.pairs import UnmatchedInstancePair
from panoptica.pipeline.regionwise import evaluate_regionwise


def test_regionwise_splits_by_nearest_ref_instance(xp):
    # Two well-separated ref instances; one FP pred voxel-blob sits strictly
    # closer to ref instance 2 than to ref instance 1.
    ref = xp.zeros((10, 10), dtype=xp.int64)
    ref[0:2, 0:2] = 1
    ref[8:10, 8:10] = 2

    pred = xp.zeros((10, 10), dtype=xp.int64)
    pred[0:2, 0:2] = 1  # matches ref 1
    pred[7:9, 7:9] = 3  # FP, nearer to ref instance 2

    pair = UnmatchedInstancePair(ref=ref, pred=pred, n_ref=2, n_pred=2, spacing=None)

    out = evaluate_regionwise(pair, metrics=["DSC"], xp=xp, global_metrics=["DSC"])

    assert set(out["regions"].keys()) == {1, 2}
    # Region 1 (ref instance 1) has its perfect match, no FP.
    assert out["regions"][1]["tp"] == 1
    assert out["regions"][1]["fp"] == 0
    # Region 2 (ref instance 2) is unmatched and picks up the nearby FP pred blob.
    assert out["regions"][2]["fn"] == 1
    assert out["regions"][2]["fp"] == 1
    # region_avg key present for the requested global metric.
    assert "region_avg_global_bin_dsc" in out


def test_regionwise_two_perfect_matches(xp):
    # Two well-separated ref instances, each perfectly matched -> two regions,
    # each tp=1 with no FP/FN.
    ref = xp.zeros((10, 10), dtype=xp.int64)
    ref[0:2, 0:2] = 1
    ref[8:10, 8:10] = 2
    pred = xp.zeros((10, 10), dtype=xp.int64)
    pred[0:2, 0:2] = 1
    pred[8:10, 8:10] = 2
    pair = UnmatchedInstancePair(ref=ref, pred=pred, n_ref=2, n_pred=2, spacing=None)

    out = evaluate_regionwise(pair, metrics=["DSC"], xp=xp, global_metrics=["DSC"])
    assert set(out["regions"]) == {1, 2}
    for rid in (1, 2):
        assert out["regions"][rid]["tp"] == 1
        assert out["regions"][rid]["fp"] == 0
        assert out["regions"][rid]["fn"] == 0
