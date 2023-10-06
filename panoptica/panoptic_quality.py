import numpy as np
from scipy.optimize import linear_sum_assignment
from multiprocessing import Pool
import warnings

from typing import Tuple

import cc3d


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
    iou = np.sum(intersection) / np.sum(union)
    return iou


def _label_instances(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Label connected components in a segmentation mask.

    Args:
        mask (np.ndarray): segmentation mask (2D or 3D array).

    Returns:
        Tuple[np.ndarray, int]:
            - Labeled mask with instances
            - Number of instances found
    """
    labeled, num_instances = cc3d.connected_components(mask, return_N=True)
    return labeled, num_instances


def count_unique_without_zeros(arr: np.ndarray) -> int:
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


def panoptic_quality(
    ref_mask: np.ndarray,
    pred_mask: np.ndarray,
    cc: bool = False,
    iou_threshold: float = 0.5,
) -> Tuple[float, float, float, int, int, int]:
    """
    Compute Panoptic Quality (PQ), Segmentation Quality (SQ), and Recognition Quality (RQ) for binary masks.

    Args:
        ref_mask (np.ndarray): Reference mask (2D or 3D array).
        pred_mask (np.ndarray): Predicted mask (2D or 3D array).
        cc (bool, optional): Whether to conduct connected component analysis on masks. Defaults to False.
        iou_threshold (float, optional): IoU threshold for considering a match. Defaults to 0.5.

    Returns:
        Tuple[float, float, float, int, int, int]:
            - Panoptic Quality (PQ)
            - Segmentation Quality (SQ)
            - Recognition Quality (RQ)
            - True Positives (tp)
            - False Positives (fp)
            - False Negatives (fn)
    """

    if cc == True:
        # Perform connected component analysis on masks
        ref_labels, num_ref_instances = _label_instances(ref_mask)
        pred_labels, num_pred_instances = _label_instances(pred_mask)
    else:
        # Use masks directly without connected component analysis
        ref_labels = ref_mask
        num_ref_instances = count_unique_without_zeros(ref_mask)

        pred_labels = pred_mask
        num_pred_instances = count_unique_without_zeros(ref_mask)

    # Handle edge cases
    if num_ref_instances == 0 and num_pred_instances == 0:
        return 1.0, 1.0, 1.0, 0, 0, 0
    elif num_ref_instances == 0:
        return 0.0, 0.0, 0.0, 0, num_pred_instances, 0
    elif num_pred_instances == 0:
        return 0.0, 0.0, 0.0, 0, 0, num_ref_instances

    # Create a pool of worker processes
    pool = Pool()

    # Create a list of tuples containing pairs of instance indices to compute IoU for
    instance_pairs = []
    for ref_instance_idx in range(1, num_ref_instances + 1):
        for pred_instance_idx in range(1, num_pred_instances + 1):
            instance_pairs.append(
                (ref_labels, pred_labels, ref_instance_idx, pred_instance_idx)
            )

    # Calculate IoU for all instance pairs in parallel using starmap
    iou_values = pool.starmap(_compute_instance_iou, instance_pairs)

    # Close the pool of worker processes
    pool.close()
    pool.join()

    # Reshape the resulting IoU values into a matrix
    iou_matrix = np.array(iou_values).reshape((num_ref_instances, num_pred_instances))

    # Use the Hungarian algorithm for optimal instance matching
    ref_indices, pred_indices = linear_sum_assignment(-iou_matrix)

    # Initialize variables for Panoptic Quality (PQ), Segmentation Quality (SQ), and Recognition Quality (RQ)
    tp = 0  # True positives (correctly matched instances)
    fp = 0  # False positives (extra predicted instances)
    fn = 0  # False negatives (missing reference instances)

    for ref_idx, pred_idx in zip(ref_indices, pred_indices):
        iou = iou_matrix[ref_idx][pred_idx]
        if iou >= iou_threshold:
            tp += 1
        else:
            fp += 1

    fn = num_ref_instances - tp

    pq = tp / (tp + 0.5 * fp + 0.5 * fn)
    sq = tp / num_ref_instances
    rq = tp / (0.5 * num_ref_instances + 0.5 * num_pred_instances)

    return pq, sq, rq, tp, fp, fn
