import concurrent.futures
from panoptica.utils.datatypes import MatchedInstancePair
from panoptica.result import PanopticaResult
from panoptica.metrics import _compute_iou, _compute_dice_coefficient, _average_symmetric_surface_distance
from panoptica.timing import measure_time
import numpy as np
import gc
from multiprocessing import Pool


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
    tp, dice_list, iou_list, assd_list = 0, [], [], []

    reference_arr, prediction_arr = matched_instance_pair._reference_arr, matched_instance_pair._prediction_arr
    ref_labels = matched_instance_pair._ref_labels

    # instance_pairs = _calc_overlapping_labels(
    #    prediction_arr=prediction_arr,
    #    reference_arr=reference_arr,
    #    ref_labels=ref_labels,
    # )
    # instance_pairs = [(ra, pa, rl, iou_threshold) for (ra, pa, rl, pl) in instance_pairs]

    instance_pairs = [(reference_arr, prediction_arr, ref_idx, iou_threshold) for ref_idx in ref_labels]
    with Pool() as pool:
        metric_values = pool.starmap(_evaluate_instance, instance_pairs)

    for tp_i, dice_i, iou_i, assd_i in metric_values:
        tp += tp_i
        if dice_i is not None and iou_i is not None and assd_i is not None:
            dice_list.append(dice_i)
            iou_list.append(iou_i)
            assd_list.append(assd_i)

    # Use concurrent.futures.ThreadPoolExecutor for parallelization
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #    futures = [
    #        executor.submit(
    #            _evaluate_instance,
    #            reference_arr,
    #            prediction_arr,
    #            ref_idx,
    #            iou_threshold,
    #        )
    #        for ref_idx in ref_labels
    #    ]
    #
    #        for future in concurrent.futures.as_completed(futures):
    #            tp_i, dice_i, iou_i, assd_i = future.result()
    #            tp += tp_i
    #            if dice_i is not None and iou_i is not None and assd_i is not None:
    #                dice_list.append(dice_i)
    #                iou_list.append(iou_i)
    #                assd_list.append(assd_i)
    #            del future
    #            gc.collect()
    # Create and return the PanopticaResult object with computed metrics
    return PanopticaResult(
        num_ref_instances=matched_instance_pair.n_reference_instance,
        num_pred_instances=matched_instance_pair.n_prediction_instance,
        tp=tp,
        dice_list=dice_list,
        iou_list=iou_list,
        assd_list=assd_list,
    )


def _evaluate_instance(
    reference_arr: np.ndarray,
    prediction_arr: np.ndarray,
    ref_idx: int,
    iou_threshold: float,
) -> tuple[int, float | None, float | None, float | None]:
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
    if ref_arr.sum() == 0 or pred_arr.sum() == 0:
        tp = 0
        dice = None
        iou = None
        assd = None
    else:
        iou: float | None = _compute_iou(
            reference=ref_arr,
            prediction=pred_arr,
        )
        if iou > iou_threshold:
            tp = 1
            dice = _compute_dice_coefficient(
                reference=ref_arr,
                prediction=pred_arr,
            )
            assd = _average_symmetric_surface_distance(pred_arr, ref_arr)
        else:
            tp = 0
            dice = None
            iou = None
            assd = None

    return tp, dice, iou, assd
