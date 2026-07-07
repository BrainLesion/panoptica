"""Smoke/behavior tests for the public Evaluator (S5)."""

from __future__ import annotations

import numpy as np

from panoptica import EvalResult, Evaluator, InputType


def _two_instance_pair():
    ref = np.zeros((12, 12), dtype=np.uint32)
    pred = np.zeros((12, 12), dtype=np.uint32)
    ref[1:4, 1:4] = 1
    pred[1:5, 1:4] = 1  # instance 1 over-segmented
    ref[7:10, 7:10] = 2
    pred[7:10, 7:10] = 2  # instance 2 perfect
    return ref, pred


def test_matched_input_counts_and_dsc(device):
    ref, pred = _two_instance_pair()
    r = Evaluator(InputType.MATCHED_INSTANCE, device=device).evaluate(pred, ref)
    assert isinstance(r, EvalResult)
    assert (r.get("tp"), r.get("fp"), r.get("fn")) == (2, 0, 0)
    # inst1 dsc=18/21, inst2 dsc=1 -> avg
    assert abs(r.get("dsc_avg") - (18 / 21 + 1.0) / 2) < 1e-6
    assert abs(r.get("iou_avg") - (0.75 + 1.0) / 2) < 1e-6


def test_semantic_input_approximates_and_matches(device):
    sref = np.zeros((20, 20), dtype=np.uint32)
    spred = np.zeros((20, 20), dtype=np.uint32)
    sref[2:6, 2:6] = 1
    sref[12:16, 12:16] = 1
    spred[2:6, 2:6] = 1
    spred[12:16, 12:16] = 1
    r = Evaluator(InputType.SEMANTIC, device=device).evaluate(spred, sref)
    assert r.get("n_ref_instances") == 2
    assert (r.get("tp"), r.get("fp"), r.get("fn")) == (2, 0, 0)
    assert abs(r.get("dsc_avg") - 1.0) < 1e-9


def test_result_serializable_and_frame():
    ref, pred = _two_instance_pair()
    r = Evaluator(InputType.MATCHED_INSTANCE, device="cpu").evaluate(pred, ref)
    d = r.to_dict()
    assert d["tp"] == 2
    frame = r.to_frame()
    assert frame.shape[0] == 1
