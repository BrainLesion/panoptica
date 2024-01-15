import concurrent.futures
import gc
from multiprocessing import Pool

import numpy as np

from panoptica.metrics import (
    _MatchingMetric,
)
from panoptica.panoptic_result import PanopticaResult
from panoptica.timing import measure_time
from panoptica.utils import EdgeCaseHandler
from panoptica.utils.processing_pair import MatchedInstancePair
from panoptica.metrics import Metrics


def evaluate_matched_instance(
    matched_instance_pair: MatchedInstancePair,
    eval_metrics: list[_MatchingMetric] = [Metrics.DSC, Metrics.IOU, Metrics.ASSD],
    decision_metric: _MatchingMetric | None = Metrics.IOU,
    decision_threshold: float | None = None,
    edge_case_handler: EdgeCaseHandler | None = None,
    **kwargs,
) -> PanopticaResult:
    """
    Map instance labels based on the provided labelmap and create a MatchedInstancePair.

    Args:
        processing_pair (UnmatchedInstancePair): The unmatched instance pair containing original labels.
        labelmap (Instance_Label_Map): The instance label map obtained from instance matching.

    Returns:
        MatchedInstancePair: The result of mapping instance labels.

    Example:
    >>> unmatched_instance_pair = UnmatchedInstancePair(...)
    >>> labelmap = [([1, 2], [3, 4]), ([5], [6])]
    >>> result = map_instance_labels(unmatched_instance_pair, labelmap)
    """
    if edge_case_handler is None:
        edge_case_handler = EdgeCaseHandler()
    if decision_metric is not None:
        assert decision_metric.name in [
            v.name for v in eval_metrics
        ], "decision metric not contained in eval_metrics"
        assert decision_threshold is not None, "decision metric set but no threshold"
    # Initialize variables for True Positives (tp)
    tp = len(matched_instance_pair.matched_instances)
    score_dict: dict[str | _MatchingMetric, list[float]] = {
        m.name: [] for m in eval_metrics
    }

    reference_arr, prediction_arr = (
        matched_instance_pair._reference_arr,
        matched_instance_pair._prediction_arr,
    )
    ref_matched_labels = matched_instance_pair.matched_instances

    instance_pairs = [
        (reference_arr, prediction_arr, ref_idx, eval_metrics)
        for ref_idx in ref_matched_labels
    ]
    with Pool() as pool:
        metric_dicts = pool.starmap(_evaluate_instance, instance_pairs)

    for metric_dict in metric_dicts:
        if decision_metric is None or (
            decision_threshold is not None
            and decision_metric.score_beats_threshold(
                metric_dict[decision_metric.name], decision_threshold
            )
        ):
            for k, v in metric_dict.items():
                score_dict[k].append(v)

    # Create and return the PanopticaResult object with computed metrics
    return PanopticaResult(
        num_ref_instances=matched_instance_pair.n_reference_instance,
        num_pred_instances=matched_instance_pair.n_prediction_instance,
        tp=tp,
        list_metrics=score_dict,
        edge_case_handler=edge_case_handler,
    )


def _evaluate_instance(
    reference_arr: np.ndarray,
    prediction_arr: np.ndarray,
    ref_idx: int,
    eval_metrics: list[_MatchingMetric],
) -> dict[str, float]:
    """
    Evaluate a single instance.

    Args:
        ref_labels (np.ndarray): Reference instance segmentation mask.
        pred_labels (np.ndarray): Predicted instance segmentation mask.
        ref_idx (int): The label of the current instance.
        iou_threshold (float): The IoU threshold for considering a match.

    Returns:
        Tuple[int, float, float]: Tuple containing True Positives (int), Dice coefficient (float), and IoU (float).
    """
    ref_arr = reference_arr == ref_idx
    pred_arr = prediction_arr == ref_idx
    result: dict[str, float] = {}
    if ref_arr.sum() == 0 or pred_arr.sum() == 0:
        return result
    else:
        for metric in eval_metrics:
            value = metric._metric_function(ref_arr, pred_arr)
            result[metric.name] = value

    return result
