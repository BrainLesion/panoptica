import numpy as np

from panoptica.metrics.aggregated_jaccard_index import (
    _compute_aggregated_jaccard_index,
    _compute_aggregated_jaccard_index_plus,
)
from panoptica import Panoptica_Evaluator
from panoptica.metrics import Metric


def test_aji_and_aji_plus_perfect_match():
    reference = np.array(
        [
            [1, 1, 0],
            [0, 2, 2],
        ],
        dtype=np.uint8,
    )
    prediction = reference.copy()

    assert _compute_aggregated_jaccard_index(reference, prediction) == 1.0
    assert _compute_aggregated_jaccard_index_plus(reference, prediction) == 1.0


def test_aji_and_aji_plus_empty_empty_is_one():
    reference = np.zeros((4, 4), dtype=np.uint8)
    prediction = np.zeros((4, 4), dtype=np.uint8)

    assert _compute_aggregated_jaccard_index(reference, prediction) == 1.0
    assert _compute_aggregated_jaccard_index_plus(reference, prediction) == 1.0


def test_aji_and_aji_plus_empty_vs_non_empty_is_zero():
    reference = np.zeros((4, 4), dtype=np.uint8)
    prediction = np.zeros((4, 4), dtype=np.uint8)
    prediction[1:3, 1:3] = 1

    assert _compute_aggregated_jaccard_index(reference, prediction) == 0.0
    assert _compute_aggregated_jaccard_index_plus(reference, prediction) == 0.0


def test_aji_and_aji_plus_no_overlap_is_zero():
    reference = np.array([[1, 0]], dtype=np.uint8)
    prediction = np.array([[0, 1]], dtype=np.uint8)

    assert _compute_aggregated_jaccard_index(reference, prediction) == 0.0
    assert _compute_aggregated_jaccard_index_plus(reference, prediction) == 0.0


def test_aji_plus_enforces_one_to_one_matching():
    # Two GT instances overlap the same single predicted instance.
    # Classical AJI can reuse the prediction for both GT instances.
    # AJI+ must choose only one GT-pred pair.
    reference = np.array([[1, 0, 2]], dtype=np.uint8)
    prediction = np.array([[1, 1, 1]], dtype=np.uint8)

    assert (
        _compute_aggregated_jaccard_index(reference, prediction) == 1.0 / 3.0
    )
    assert (
        _compute_aggregated_jaccard_index_plus(reference, prediction)
        == 1.0 / 4.0
    )


def test_aji_and_aji_plus_are_exposed_in_panoptica_result():
    reference = np.array([[1, 0, 2]], dtype=np.uint8)
    prediction = np.array([[1, 1, 1]], dtype=np.uint8)

    evaluator = Panoptica_Evaluator(
        instance_metrics=[Metric.IOU],
        global_metrics=[],
    )

    result = evaluator.evaluate(
        prediction_arr=prediction,
        reference_arr=reference,
    )["ungrouped"]

    result_dict = result.to_dict()

    assert result.aji == 1.0 / 3.0
    assert result.aji_plus == 1.0 / 4.0
    assert result_dict["aji"] == 1.0 / 3.0
    assert result_dict["aji_plus"] == 1.0 / 4.0
