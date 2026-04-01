import numpy as np


def _compute_instance_iou(
    reference_arr: np.ndarray,
    prediction_arr: np.ndarray,
    ref_instance_idx: int | None = None,
    pred_instance_idx: int | None = None,
) -> float:
    """
    Compute Intersection over Union (IoU) between a specific pair of reference and prediction instances.

    Args:
        ref_labels (np.ndarray): Reference instance labels.
        pred_labels (np.ndarray): Prediction instance labels.
        ref_instance_idx (int): Index of the reference instance.
        pred_instance_idx (int): Index of the prediction instance.

    Returns:
        float: IoU between the specified instances.
    """
    if ref_instance_idx is None and pred_instance_idx is None:
        return _compute_iou(
            reference_arr=reference_arr,
            prediction_arr=prediction_arr,
        )
    ref_instance_mask = reference_arr == ref_instance_idx
    pred_instance_mask = prediction_arr == pred_instance_idx
    return _compute_iou(ref_instance_mask, pred_instance_mask)


def _compute_iou(
    reference_arr: np.ndarray,
    prediction_arr: np.ndarray,
    *args,
) -> float:
    """
    Compute Intersection over Union (IoU) between two masks.

    Args:
        reference (np.ndarray): Reference mask.
        prediction (np.ndarray): Prediction mask.

    Returns:
        float: IoU between the two masks. A value between 0 and 1, where higher values
        indicate better overlap and similarity between masks.
    """
    intersection = np.logical_and(reference_arr, prediction_arr)
    union = np.logical_or(reference_arr, prediction_arr)

    union_sum = np.sum(union)

    # Handle division by zero
    if union_sum == 0:
        return 0.0

    iou = np.sum(intersection) / union_sum
    return iou
