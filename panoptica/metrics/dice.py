import numpy as np


def _compute_instance_volumetric_dice(
    ref_labels: np.ndarray,
    pred_labels: np.ndarray,
    ref_instance_idx: int | None = None,
    pred_instance_idx: int | None = None,
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
    if ref_instance_idx is None and pred_instance_idx is None:
        return _compute_dice_coefficient(
            reference=ref_labels,
            prediction=pred_labels,
        )
    ref_instance_mask = ref_labels == ref_instance_idx
    pred_instance_mask = pred_labels == pred_instance_idx
    return _compute_dice_coefficient(
        reference=ref_instance_mask,
        prediction=pred_instance_mask,
    )


def _compute_dice_coefficient(
    reference: np.ndarray,
    prediction: np.ndarray,
    *args,
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
