import concurrent.futures
from panoptica.utils.datatypes import MatchedInstancePair
from panoptica.result import PanopticaResult
import numpy as np
from panoptica.utils.metrics import _compute_iou, _compute_dice_coefficient


def evaluate_matched_instance(matched_instance_pair: MatchedInstancePair, iou_threshold: float, **kwargs) -> PanopticaResult:
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
    # Initialize variables for True Positives (tp)
    tp, dice_list, iou_list = 0, [], []

    reference_arr, prediction_arr = matched_instance_pair.reference_arr, matched_instance_pair.prediction_arr
    ref_labels = matched_instance_pair.ref_labels

    # Use concurrent.futures.ThreadPoolExecutor for parallelization
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                _evaluate_instance,
                reference_arr,
                prediction_arr,
                ref_idx,
                iou_threshold,
            )
            for ref_idx in ref_labels
        ]

        for future in concurrent.futures.as_completed(futures):
            tp_i, dice_i, iou_i = future.result()
            tp += tp_i
            if dice_i is not None or iou_i is not None:
                dice_list.append(dice_i)
                iou_list.append(iou_i)
    # Create and return the PanopticaResult object with computed metrics
    return PanopticaResult(
        num_ref_instances=matched_instance_pair.n_reference_instance,
        num_pred_instances=matched_instance_pair.n_prediction_instance,
        tp=tp,
        dice_list=dice_list,
        iou_list=iou_list,
    )


def _evaluate_instance(
    reference_arr: np.ndarray,
    prediction_arr: np.ndarray,
    ref_idx: int,
    iou_threshold: float,
) -> tuple[int, float | None, float | None]:
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
    iou: float | None = _compute_iou(
        reference=reference_arr == ref_idx,
        prediction=prediction_arr == ref_idx,
    )
    if iou > iou_threshold:
        tp = 1
        dice = _compute_dice_coefficient(
            reference=reference_arr == ref_idx,
            prediction=prediction_arr == ref_idx,
        )
    else:
        tp = 0
        dice = None
        iou = None

    return tp, dice, iou
