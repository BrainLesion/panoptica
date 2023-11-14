import numpy as np


def _compute_instance_iou(
    reference_arr: np.ndarray,
    prediction_arr: np.ndarray,
    ref_instance_idx: int,
    pred_instance_idx: int,
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
    ref_instance_mask = reference_arr == ref_instance_idx
    pred_instance_mask = prediction_arr == pred_instance_idx
    intersection = np.logical_and(ref_instance_mask, pred_instance_mask)
    union = np.logical_or(ref_instance_mask, pred_instance_mask)

    union_sum = np.sum(union)
    # Handle division by zero
    if union_sum == 0:
        return 0.0

    iou = np.sum(intersection) / union_sum
    return iou


def _compute_iou(reference: np.ndarray, prediction: np.ndarray) -> float:
    """
    Compute Intersection over Union (IoU) between two masks.

    Args:
        reference (np.ndarray): Reference mask.
        prediction (np.ndarray): Prediction mask.

    Returns:
        float: IoU between the two masks. A value between 0 and 1, where higher values
        indicate better overlap and similarity between masks.
    """
    intersection = np.logical_and(reference, prediction)
    union = np.logical_or(reference, prediction)

    union_sum = np.sum(union)

    # Handle division by zero
    if union_sum == 0:
        return 0.0

    iou = np.sum(intersection) / union_sum
    return iou
