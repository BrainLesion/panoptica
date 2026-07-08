from __future__ import annotations

import math

import pytest

from panoptica.core.errors import MetricComputeError
from panoptica.kernels.overlap import overlap_cost


def _ref_pred(xp):
    # ref:            pred:
    # 1 1 0           1 1 0
    # 0 2 2           0 0 2
    ref = xp.array([[1, 1, 0], [0, 2, 2]], dtype=xp.int32)
    pred = xp.array([[1, 1, 0], [0, 0, 2]], dtype=xp.int32)
    return ref, pred


def test_overlap_cost_shape(xp):
    ref, pred = _ref_pred(xp)
    cost, _, _ = overlap_cost(ref, pred, xp, metric="iou")
    assert cost.shape == (3, 3)  # (n_ref+1, n_pred+1) = (2+1, 2+1)


def test_overlap_cost_iou_hand_computed(xp):
    ref, pred = _ref_pred(xp)
    cost, _, _ = overlap_cost(ref, pred, xp, metric="iou")
    # ref=1 area=2, pred=1 area=2, intersection=2 -> IoU = 2/(2+2-2) = 1.0
    assert math.isclose(float(cost[1, 1]), 1.0, rel_tol=1e-9)
    # ref=2 area=2, pred=2 area=1, intersection=1 -> IoU = 1/(2+1-1) = 0.5
    assert math.isclose(float(cost[2, 2]), 0.5, rel_tol=1e-9)
    # no overlap between ref=1/pred=2 and ref=2/pred=1
    assert float(cost[1, 2]) == 0.0
    assert float(cost[2, 1]) == 0.0


def test_overlap_cost_dsc_hand_computed(xp):
    ref, pred = _ref_pred(xp)
    cost, _, _ = overlap_cost(ref, pred, xp, metric="dsc")
    # DSC(1,1) = 2*2/(2+2) = 1.0
    assert math.isclose(float(cost[1, 1]), 1.0, rel_tol=1e-9)
    # DSC(2,2) = 2*1/(2+1) = 0.6666...
    assert math.isclose(float(cost[2, 2]), 2.0 / 3.0, rel_tol=1e-9)


def test_overlap_cost_case_insensitive(xp):
    ref, pred = _ref_pred(xp)
    a, _, _ = overlap_cost(ref, pred, xp, metric="IOU")
    b, _, _ = overlap_cost(ref, pred, xp, metric="iou")
    assert bool((a == b).all())


def test_overlap_cost_unsupported_metric_raises(xp):
    ref, pred = _ref_pred(xp)
    with pytest.raises(MetricComputeError):
        overlap_cost(ref, pred, xp, metric="bogus")
