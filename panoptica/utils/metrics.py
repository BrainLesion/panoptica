import numpy as np


def _compute_instance_volumetric_dice(
    ref_labels: np.ndarray,
    pred_labels: np.ndarray,
    ref_instance_idx: int,
    pred_instance_idx: int,
) -> float:
    """
    Compute the Dice coefficient between a specific pair of instances.

    The Dice coefficient measures the similarity or overlap between two binary masks representing instances.
    It is defined as:

    Dice = (2 * intersection) / (ref_area + pred_area)

    Args:
        ref_labels (np.ndarray): Reference instance labels.
        pred_labels (np.ndarray): Prediction instance labels.
        ref_instance_idx (int): Index of the reference instance.
        pred_instance_idx (int): Index of the prediction instance.

    Returns:
        float: Dice coefficient between the specified instances. A value between 0 and 1, where higher values
        indicate better overlap and similarity between instances.
    """
    ref_instance_mask = ref_labels == ref_instance_idx
    pred_instance_mask = pred_labels == pred_instance_idx
    intersection = np.logical_and(ref_instance_mask, pred_instance_mask)
    ref_area = np.sum(ref_instance_mask)
    pred_area = np.sum(pred_instance_mask)

    # Calculate Dice coefficient
    dice = 2 * np.sum(intersection) / (ref_area + pred_area)

    return dice


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


def _compute_dice_coefficient(
    reference: np.ndarray,
    prediction: np.ndarray,
) -> float:
    """
    Compute the Dice coefficient between two binary masks.

    The Dice coefficient measures the similarity or overlap between two binary masks.
    It is defined as:

    Dice = (2 * intersection) / (area_mask1 + area_mask2)

    Args:
        reference (np.ndarray): Reference binary mask.
        prediction (np.ndarray): Prediction binary mask.

    Returns:
        float: Dice coefficient between the two binary masks. A value between 0 and 1, where higher values
        indicate better overlap and similarity between masks.
    """
    intersection = np.logical_and(reference, prediction)
    reference_mask = np.sum(reference)
    prediction_mask = np.sum(prediction)

    # Handle division by zero
    if reference_mask == 0 and prediction_mask == 0:
        return 0.0

    # Calculate Dice coefficient
    dice = 2 * np.sum(intersection) / (reference_mask + prediction_mask)
    return dice
