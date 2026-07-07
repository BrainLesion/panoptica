"""End-to-end test for pipeline.evaluate on a tiny hand-computed MatchedInstancePair."""

from __future__ import annotations

import math

from panoptica.core.pairs import MatchedInstancePair
from panoptica.pipeline.evaluate import evaluate


def _tiny_pair(xp):
    # ref: instance 1 (2x2, perfectly matched by pred instance 1), instance 2
    # (2x2, unmatched -> FN). pred: instance 1 (matches ref 1), instance 3
    # (2x2, unmatched -> FP).
    ref = xp.zeros((8, 8), dtype=xp.int64)
    ref[0:2, 0:2] = 1
    ref[4:6, 4:6] = 2

    pred = xp.zeros((8, 8), dtype=xp.int64)
    pred[0:2, 0:2] = 1
    pred[6:8, 6:8] = 3

    matched_ids = xp.asarray([[1, 1]], dtype=xp.int64)
    return MatchedInstancePair(
        ref=ref,
        pred=pred,
        matched_ids=matched_ids,
        unmatched_ref=(2,),
        unmatched_pred=(3,),
        spacing=None,
    )


def test_evaluate_counts_and_dsc(xp):
    pair = _tiny_pair(xp)
    result = evaluate(pair, metrics=["DSC"], xp=xp, global_metrics=["DSC"])

    assert result["n_ref_instances"] == 2
    assert result["n_pred_instances"] == 2
    assert result["tp"] == 1
    assert result["fp"] == 1
    assert result["fn"] == 1
    assert result["prec"] == 0.5
    assert result["rec"] == 0.5
    assert result["rq"] == 0.5

    assert result["dsc_all"].tolist() == [1.0]
    assert result["dsc_avg"] == 1.0
    assert result["dsc_std"] == 0.0
    assert result["dsc_min"] == 1.0
    assert result["dsc_max"] == 1.0

    assert result["sq_dsc"] == 1.0
    assert result["sq_dsc_std"] == 0.0
    assert result["pq_dsc"] == 0.5

    # global_bin_dsc: ref has 8 fg voxels, pred has 8 fg voxels, overlap 4.
    assert math.isclose(result["global_bin_dsc"], 0.5)


def test_evaluate_iou_uses_unsuffixed_keys(xp):
    pair = _tiny_pair(xp)
    result = evaluate(pair, metrics=["IOU"], xp=xp)

    assert result["iou_all"].tolist() == [1.0]
    assert result["sq"] == 1.0
    assert result["sq_std"] == 0.0
    assert result["pq"] == 0.5


def test_evaluate_zero_tp_edge_case(xp):
    ref = xp.zeros((4, 4), dtype=xp.int64)
    ref[0:2, 0:2] = 1
    pred = xp.zeros((4, 4), dtype=xp.int64)
    pred[2:4, 2:4] = 1

    pair = MatchedInstancePair(
        ref=ref,
        pred=pred,
        matched_ids=xp.zeros((0, 2), dtype=xp.int64),
        unmatched_ref=(1,),
        unmatched_pred=(2,),
        spacing=None,
    )
    result = evaluate(pair, metrics=["DSC"], xp=xp)
    assert result["tp"] == 0
    assert result["fp"] == 1
    assert result["fn"] == 1
    # DSC zero_tp default policy is EdgeCaseResult.ZERO (v1 parity).
    assert result["sq_dsc"] == 0.0
