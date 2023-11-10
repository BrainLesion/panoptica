import concurrent.futures
from panoptica.utils.datatypes import MatchedInstancePair
from panoptica.result import PanopticaResult
import numpy as np
from panoptica.utils.metrics import _compute_iou, _compute_dice_coefficient


def evaluate_matched_instance(semantic_pair: MatchedInstancePair, iou_threshold: float, **kwargs) -> PanopticaResult:
    # Initialize variables for True Positives (tp)
    tp, dice_list, iou_list = 0, [], []

    reference_arr, prediction_arr = semantic_pair.reference_arr, semantic_pair.prediction_arr
    ref_labels = semantic_pair.ref_labels

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
            if dice_i is not None:
                dice_list.append(dice_i)
            if iou_i is not None:
                iou_list.append(iou_i)
    # Create and return the PanopticaResult object with computed metrics
    return PanopticaResult(
        num_ref_instances=semantic_pair.n_reference_instance,
        num_pred_instances=semantic_pair.n_prediction_instance,
        tp=tp,
        dice_list=dice_list,
        iou_list=iou_list,
    )


def _evaluate_instance(
    ref_labels: np.ndarray,
    pred_labels: np.ndarray,
    ref_idx: int,
    iou_threshold: float,
) -> tuple[int, float, float]:
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
    iou = _compute_iou(
        reference=ref_labels == ref_idx,
        prediction=pred_labels == ref_idx,
    )
    if iou > iou_threshold:
        tp = 1
        dice = _compute_dice_coefficient(
            reference=ref_labels == ref_idx,
            prediction=pred_labels == ref_idx,
        )
    else:
        tp = 0
        dice = None

    return tp, dice, iou
