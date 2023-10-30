import warnings
from typing import Tuple

import cc3d
import numpy as np
from scipy import ndimage


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
    ref_labels: np.ndarray,
    pred_labels: np.ndarray,
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
    ref_instance_mask = ref_labels == ref_instance_idx
    pred_instance_mask = pred_labels == pred_instance_idx
    intersection = np.logical_and(ref_instance_mask, pred_instance_mask)
    union = np.logical_or(ref_instance_mask, pred_instance_mask)

    union_sum = np.sum(union)
    # Handle division by zero
    if union_sum == 0:
        return 0.0

    iou = np.sum(intersection) / union_sum
    return iou


def _label_instances(mask: np.ndarray, cca_backend: str) -> Tuple[np.ndarray, int]:
    """
    Label connected components in a segmentation mask.

    Args:
        mask (np.ndarray): segmentation mask (2D or 3D array).
        cca_backend (str): Backend for connected components labeling. Should be "cc3d" or "scipy".

    Returns:
        Tuple[np.ndarray, int]:
            - Labeled mask with instances
            - Number of instances found
    """
    if cca_backend == "cc3d":
        labeled, num_instances = cc3d.connected_components(mask, return_N=True)
    elif cca_backend == "scipy":
        labeled, num_instances = ndimage.label(mask)
    else:
        raise NotImplementedError(f"Unsupported cca_backend: {cca_backend}")
    return labeled, num_instances


def _count_unique_without_zeros(arr: np.ndarray) -> int:
    """
    Count the number of unique elements in the input NumPy array, excluding zeros.

    Args:
        arr (np.ndarray): Input array.

    Returns:
        int: Number of unique elements excluding zeros.
    """
    if np.any(arr < 0):
        warnings.warn("Negative values are present in the input array.")

    unique_elements = np.unique(arr)
    if 0 in unique_elements:
        return len(unique_elements) - 1
    else:
        return len(unique_elements)
