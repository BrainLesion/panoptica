import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy import ndimage
from typing import Tuple


def compute_instance_iou(
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


def label_instances(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Label connected components in a binary mask.

    Args:
        mask (np.ndarray): Binary mask (2D or 3D array).

    Returns:
        Tuple[np.ndarray, int]:
            - Labeled mask with instances
            - Number of instances found
    """
    labeled, num_instances = ndimage.label(mask)
    return labeled, num_instances


def panoptic_quality_for_binary_masks(
    ref_masks: np.ndarray,
    pred_masks: np.ndarray,
    iou_threshold: float = 0.5,
) -> Tuple[float, float, float, int, int, int]:
    """
    Compute Panoptic Quality (PQ), Segmentation Quality (SQ), and Recognition Quality (RQ) for binary masks.

    Args:
        ref_masks (np.ndarray): Reference binary masks (2D or 3D array).
        pred_masks (np.ndarray): Predicted binary masks (2D or 3D array).
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

    ref_labels, num_ref_instances = label_instances(ref_masks)
    pred_labels, num_pred_instances = label_instances(pred_masks)

    # Handle edge cases
    if num_ref_instances == 0 and num_pred_instances == 0:
        return 1.0, 1.0, 1.0, 0, 0, 0
    elif num_ref_instances == 0:
        return 0.0, 0.0, 0.0, 0, num_pred_instances, 0
    elif num_pred_instances == 0:
        return 0.0, 0.0, 0.0, 0, 0, num_ref_instances

    # Calculate IoU for all instance pairs
    iou_matrix = np.zeros((num_ref_instances, num_pred_instances))
    for ref_instance_idx in range(1, num_ref_instances + 1):
        for pred_instance_idx in range(1, num_pred_instances + 1):
            iou_matrix[ref_instance_idx - 1][
                pred_instance_idx - 1
            ] = compute_instance_iou(
                ref_labels, pred_labels, ref_instance_idx, pred_instance_idx
            )

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
