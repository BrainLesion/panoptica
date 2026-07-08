"""Unit tests for the metric registry: METRIC_REGISTRY, Metric enum,
MetricMode aggregation, and TP/FP/FN -> precision/recall/RQ/SQ/PQ derivation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from panoptica.core.enums import Direction, MetricMode, MetricType
from panoptica.metrics.registry import (
    METRIC_REGISTRY,
    Metric,
    aggregate,
    derive_detection_quality,
    derive_pq,
    derive_sq,
    get_spec,
)


def _np(arr):
    """Duck-typed device->host transfer for assertions (cupy has `.get()`)."""
    getter = getattr(arr, "get", None)
    return getter() if callable(getter) else np.asarray(arr)


EXPECTED_IDS = {
    "DSC",
    "IOU",
    "RVD",
    "RVAE",
    "ASSD",
    "HD",
    "HD95",
    "NSD",
    "clDSC",
    "CEDI",
}


def test_registry_has_all_ten_metrics():
    assert set(METRIC_REGISTRY.keys()) == EXPECTED_IDS
    assert {m.value for m in Metric} == EXPECTED_IDS


@pytest.mark.parametrize(
    ("metric_id", "expected_direction"),
    [
        ("DSC", Direction.INCREASING),
        ("IOU", Direction.INCREASING),
        ("clDSC", Direction.INCREASING),
        ("RVD", Direction.DECREASING),
        ("RVAE", Direction.DECREASING),
        ("ASSD", Direction.DECREASING),
        ("HD", Direction.DECREASING),
        ("HD95", Direction.DECREASING),
        ("NSD", Direction.DECREASING),
        ("CEDI", Direction.DECREASING),
    ],
)
def test_registry_directions_match_v1(metric_id, expected_direction):
    spec = get_spec(metric_id)
    assert spec.direction is expected_direction
    assert spec.type is MetricType.INSTANCE
    assert callable(spec.fn)


def test_metric_enum_members_resolve_to_spec():
    spec = get_spec(Metric.DSC)
    assert spec.id == "DSC"


def test_get_spec_unknown_raises():
    from panoptica.core.errors import InputValidationError

    with pytest.raises(InputValidationError):
        get_spec("NOT_A_METRIC")


def test_aggregate_all_modes(xp):
    values = xp.asarray([1.0, 2.0, 3.0, 4.0])
    assert list(_np(aggregate(values, MetricMode.ALL, xp))) == [1.0, 2.0, 3.0, 4.0]
    assert aggregate(values, MetricMode.AVG, xp) == pytest.approx(2.5)
    assert aggregate(values, MetricMode.SUM, xp) == pytest.approx(10.0)
    assert aggregate(values, MetricMode.MIN, xp) == pytest.approx(1.0)
    assert aggregate(values, MetricMode.MAX, xp) == pytest.approx(4.0)
    assert aggregate(values, MetricMode.STD, xp) == pytest.approx(
        float(np.std([1, 2, 3, 4]))
    )


def test_aggregate_empty_list(xp):
    values = xp.asarray([], dtype=xp.float64)
    assert math.isnan(aggregate(values, MetricMode.AVG, xp))
    assert math.isnan(aggregate(values, MetricMode.SUM, xp))
    assert math.isnan(aggregate(values, MetricMode.MIN, xp))
    assert math.isnan(aggregate(values, MetricMode.MAX, xp))
    assert math.isnan(
        aggregate(values, MetricMode.STD, xp)
    )  # default empty_list_std=NaN


def test_aggregate_empty_list_custom_std(xp):
    values = xp.asarray([], dtype=xp.float64)
    assert aggregate(values, MetricMode.STD, xp, empty_list_std=0.0) == 0.0


def test_derive_detection_quality_normal():
    # tp=3, fp=1, fn=2 -> prec=3/4=0.75, rec=3/5=0.6, RQ=3/(3+0.5+1)=3/4.5
    dq = derive_detection_quality(tp=3, fp=1, fn=2)
    assert dq.prec == pytest.approx(0.75)
    assert dq.rec == pytest.approx(0.6)
    assert dq.rq == pytest.approx(3 / 4.5)


def test_derive_detection_quality_zero_tp_with_instances():
    # tp=0 but instances exist on both sides -> RQ = 0.0 (v1: panoptica_result.rq)
    dq = derive_detection_quality(tp=0, fp=2, fn=3)
    assert dq.rq == 0.0
    assert dq.prec == 0.0
    assert dq.rec == 0.0


def test_derive_detection_quality_zero_tp_no_instances():
    # tp=0, fp=0, fn=0 (n_pred=n_ref=0) -> RQ = NaN
    dq = derive_detection_quality(tp=0, fp=0, fn=0)
    assert math.isnan(dq.rq)
    assert math.isnan(dq.prec)
    assert math.isnan(dq.rec)


def test_derive_sq_normal_case(xp):
    tp_values = xp.asarray([0.5, 0.7, 0.9])
    sq = derive_sq(Metric.DSC, tp_values, xp, tp=3, n_ref=3, n_pred=3)
    assert sq == pytest.approx(0.7)


def test_derive_sq_zero_tp_uses_zero_tp_policy(xp):
    # DSC zero_tp policy: default=ZERO, no_instances=NAN
    tp_values = xp.asarray([], dtype=xp.float64)
    sq_normal = derive_sq(Metric.DSC, tp_values, xp, tp=0, n_ref=2, n_pred=2)
    assert sq_normal == 0.0
    sq_no_instances = derive_sq(Metric.DSC, tp_values, xp, tp=0, n_ref=0, n_pred=0)
    assert math.isnan(sq_no_instances)


def test_derive_pq_multiplies_sq_and_rq():
    assert derive_pq(sq=0.8, rq=0.5) == pytest.approx(0.4)
    assert math.isnan(derive_pq(sq=float("nan"), rq=0.0))


def test_full_pq_pipeline_zero_tp(xp):
    # tp=0, both sides have instances -> RQ=0.0, DSC SQ policy default=ZERO
    # -> PQ = 0.0 * 0.0 = 0.0
    dq = derive_detection_quality(tp=0, fp=2, fn=2)
    sq = derive_sq(
        Metric.DSC, xp.asarray([], dtype=xp.float64), xp, tp=0, n_ref=2, n_pred=2
    )
    pq = derive_pq(sq, dq.rq)
    assert pq == 0.0
