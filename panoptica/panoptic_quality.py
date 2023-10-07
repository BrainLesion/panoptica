import numpy as np
from scipy.optimize import linear_sum_assignment
from multiprocessing import Pool
import warnings

from typing import Tuple

import cc3d
from scipy import ndimage


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
    modus: str,
    iou_threshold: float = 0.5,
    cca_backend: str = "cc3d",
) -> Tuple[float, float, float, int, int, int]:
    """
    Compute Panoptic Quality (PQ), Segmentation Quality (SQ), and Recognition Quality (RQ) metrics for binary masks.

    Args:
        ref_mask (np.ndarray): Reference mask (2D or 3D binary array).
        pred_mask (np.ndarray): Predicted mask (2D or 3D binary array).
        modus (str): Processing mode:
            - "im" for direct comparison of instance masks
            - "cc" for connected component analysis on masks
        iou_threshold (float, optional): IoU threshold for considering a match. Defaults to 0.5.
        cca_backend (str, optional): The backend for connected component analysis. Options are "cc3d" or "scipy".
            Defaults to "cc3d".

    Returns:
        Tuple[float, float, float, int, int, int]:
            - Panoptic Quality (PQ): A metric that balances segmentation and recognition quality.
            - Segmentation Quality (SQ): The accuracy of instance segmentation.
            - Recognition Quality (RQ): The accuracy of instance recognition.
            - True Positives (tp): Number of correctly matched instances.
            - False Positives (fp): Number of extra predicted instances.
            - False Negatives (fn): Number of missing reference instances.

    Notes:
        - In "cc" mode, connected component analysis is performed on masks to identify instances.
        - In "im" mode, instance masks are directly compared without connected component analysis.

    Example:
        ref_mask = np.array([[0, 1, 1], [0, 2, 2], [0, 0, 3]])
        pred_mask = np.array([[0, 0, 1], [2, 2, 2], [0, 0, 0]])
        pq, sq, rq, tp, fp, fn = panoptic_quality(ref_mask, pred_mask, mode="im")
        print(f"PQ: {pq}, SQ: {sq}, RQ: {rq}, TP: {tp}, FP: {fp}, FN: {fn}")
    """
    if modus == "im":
        # Use instance masks directly without connected component analysis
        ref_labels = ref_mask
        num_ref_instances = count_unique_without_zeros(ref_mask)

        pred_labels = pred_mask
        num_pred_instances = count_unique_without_zeros(ref_mask)

    elif modus == "cc":
        # Perform connected component analysis on masks
        ref_labels, num_ref_instances = _label_instances(
            mask=ref_mask,
            cca_backend=cca_backend,
        )
        pred_labels, num_pred_instances = _label_instances(
            mask=pred_mask,
            cca_backend=cca_backend,
        )

    # Handle edge cses
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

    # Compute Segmentation Quality (SQ) as the average IoU of matched segments
    if tp == 0:
        sq = 0.0  # Set SQ to 0 when there are no true positives
    else:
        sq = np.sum(np.max(iou_matrix, axis=0)) / tp  # Calculate SQ as usual

    # Calculate Recognition Quality (RQ)
    rq = tp / (tp + 0.5 * fp + 0.5 * fn)

    # Compute Panoptic Quality (PQ) as the product of SQ and RQ
    pq = sq * rq

    return pq, sq, rq, tp, fp, fn
