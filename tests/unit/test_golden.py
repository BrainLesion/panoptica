"""Golden / absolute-correctness anchor.

Self-consistency (cpu==cuda, grouped==masked) can't catch an error present in
*both* backends. These cases pin panoptica to values computed by hand from the
definitions, so a systematic drift in the math is caught. Everything asserted
here is integer-exact or an exact ratio of integers; surface-distance regression
snapshots live in ``bench/golden/`` (drift detection, not first-principles).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from panoptica import Evaluator, InputType

ALL = ["DSC", "IOU", "RVD", "RVAE", "ASSD", "HD", "HD95"]

_ROOT = Path(__file__).resolve().parents[2]
_GOLDEN = _ROOT / "bench" / "golden" / "golden.json"


def _eval(pred, ref, it, **kw):
    return Evaluator(it, instance_metrics=ALL, device="cpu", **kw).evaluate(pred, ref)


def test_identical_is_perfect():
    """ref == pred, one instance: DSC=IOU=1, all distances 0, counts 1/0/0."""
    ref = np.zeros((10, 10, 10), np.uint32)
    ref[2:5, 2:5, 2:5] = 1  # 3^3 = 27 voxels
    pred = ref.copy()
    r = _eval(pred, ref, InputType.MATCHED_INSTANCE)
    assert r.get("tp") == 1 and r.get("fp") == 0 and r.get("fn") == 0
    assert r.get("dsc_avg") == pytest.approx(1.0)
    assert r.get("iou_avg") == pytest.approx(1.0)
    assert r.get("assd_avg") == pytest.approx(0.0)
    assert r.get("hd_avg") == pytest.approx(0.0)
    assert r.get("hd95_avg") == pytest.approx(0.0)
    assert r.get("rvd_avg") == pytest.approx(0.0)
    assert r.get("rvae_avg") == pytest.approx(0.0)


def test_detection_counts_tp_fp_fn():
    """One matched box (IoU=1 -> TP), one ref-only (FN), one pred-only (FP)."""
    ref = np.zeros((16, 16, 16), np.uint32)
    pred = np.zeros((16, 16, 16), np.uint32)
    ref[2:6, 2:6, 2:6] = 1        # matched
    pred[2:6, 2:6, 2:6] = 1       # exact overlap -> TP
    ref[2:6, 10:14, 2:6] = 2      # ref-only -> FN
    pred[10:14, 10:14, 2:6] = 2   # pred-only -> FP
    r = _eval(pred, ref, InputType.UNMATCHED_INSTANCE,
              matcher="naive", matching_threshold=0.5)
    assert r.get("tp") == 1
    assert r.get("fp") == 1
    assert r.get("fn") == 1
    assert r.get("prec") == pytest.approx(0.5)  # tp/(tp+fp)
    assert r.get("rec") == pytest.approx(0.5)   # tp/(tp+fn)


def test_partial_overlap_known_dsc_iou():
    """Two equal 4^3 boxes offset by 2 voxels along axis 0.

    |ref|=|pred|=64, overlap = 2*4*4 = 32.
    DSC = 2*32/(64+64) = 0.5 ; IOU = 32/(128-32) = 1/3 ; RVD = 0 (equal volumes).
    """
    ref = np.zeros((16, 16, 16), np.uint32)
    pred = np.zeros((16, 16, 16), np.uint32)
    ref[4:8, 4:8, 4:8] = 1
    pred[6:10, 4:8, 4:8] = 1
    r = _eval(pred, ref, InputType.MATCHED_INSTANCE)
    assert r.get("dsc_avg") == pytest.approx(0.5, rel=1e-9)
    assert r.get("iou_avg") == pytest.approx(1.0 / 3.0, rel=1e-9)
    assert r.get("rvd_avg") == pytest.approx(0.0, abs=1e-12)
    assert r.get("rvae_avg") == pytest.approx(0.0, abs=1e-12)


def test_known_volume_ratio_rvd():
    """pred larger than ref by a known amount: RVD = (|pred|-|ref|)/|ref|.

    ref = 4^3 = 64 ; pred = 4x4x6 = 96 (extended along axis 2, ref fully inside).
    RVD = (96-64)/64 = 0.5 ; RVAE = |96-64|/64 = 0.5.
    """
    ref = np.zeros((16, 16, 16), np.uint32)
    pred = np.zeros((16, 16, 16), np.uint32)
    ref[4:8, 4:8, 4:8] = 1     # 64
    pred[4:8, 4:8, 4:10] = 1   # 4*4*6 = 96, superset of ref
    r = _eval(pred, ref, InputType.MATCHED_INSTANCE)
    assert r.get("rvd_avg") == pytest.approx(0.5, rel=1e-9)
    assert r.get("rvae_avg") == pytest.approx(0.5, rel=1e-9)
    # DSC = 2*64/(64+96) = 128/160 = 0.8 ; IOU = 64/96 = 2/3 (ref ⊂ pred)
    assert r.get("dsc_avg") == pytest.approx(0.8, rel=1e-9)
    assert r.get("iou_avg") == pytest.approx(2.0 / 3.0, rel=1e-9)


def test_hausdorff_known_distance():
    """Two unit-spaced boxes offset by 2 along axis 0, equal size.

    Every ref-surface voxel's nearest pred-surface voxel and vice-versa: the
    max surface-to-surface distance (HD) is the axis-0 offset = 2.0 voxels.
    ASSD > 0, HD == HD95 == 2.0 here (the offset faces dominate).
    """
    ref = np.zeros((16, 16, 16), np.uint32)
    pred = np.zeros((16, 16, 16), np.uint32)
    ref[4:8, 4:8, 4:8] = 1
    pred[6:10, 4:8, 4:8] = 1
    r = _eval(pred, ref, InputType.MATCHED_INSTANCE)
    assert r.get("hd_avg") == pytest.approx(2.0, abs=1e-9)
    assert r.get("assd_avg") > 0.0
    assert r.get("assd_avg") <= 2.0


# ---- regression snapshots on seeded realistic pairs -------------------------

_ALL10 = ["DSC", "IOU", "ASSD", "RVD", "RVAE", "CEDI", "HD", "HD95", "NSD", "clDSC"]


def _load_golden():
    if not _GOLDEN.exists():
        pytest.skip(f"{_GOLDEN} missing; run bench/golden/generate_golden.py")
    return json.loads(_GOLDEN.read_text())


@pytest.mark.parametrize(
    "cell", ["small/MATCHED_INSTANCE", "small/UNMATCHED_INSTANCE", "small/SEMANTIC",
             "ellipsoids/MATCHED_INSTANCE", "ellipsoids/UNMATCHED_INSTANCE",
             "ellipsoids/SEMANTIC"],
)
def test_golden_snapshot(cell):
    """Current output must match the frozen snapshot (drift detection).

    Complements the hand-verified cases above: catches a regression on realistic
    seeded data even where a value is too complex to compute by hand.
    """
    sys.path.insert(0, str(_ROOT / "bench"))
    from datasets.generate import (
        generate_matched_instance_pair,
        generate_semantic_pair,
        generate_unmatched_instance_pair,
    )

    golden = _load_golden()[cell]
    size, it = cell.split("/")
    if it == "MATCHED_INSTANCE":
        ref, pred, _ = generate_matched_instance_pair(size=size)
    elif it == "UNMATCHED_INSTANCE":
        ref, pred = generate_unmatched_instance_pair(size=size)
    else:
        ref, pred = generate_semantic_pair(size=size)

    d = Evaluator(getattr(InputType, it), instance_metrics=_ALL10,
                  global_metrics=["DSC"], device="cpu").evaluate(pred, ref).to_dict()
    for k, expected in golden.items():
        v = d.get(k)
        if v is None:
            continue
        got = float(np.asarray(v.get() if hasattr(v, "get") else v))
        if np.isnan(expected) and np.isnan(got):
            continue
        # counts exact; everything else tight relative tolerance
        tol = dict(abs=0) if k in ("tp", "fp", "fn") else dict(rel=1e-9, abs=1e-9)
        assert got == pytest.approx(expected, **tol), f"{cell} {k}: {got} != {expected}"
