import numpy as np
from scipy.optimize import linear_sum_assignment
from multiprocessing import Pool
from typing import Tuple


from .utils import _compute_instance_iou, _label_instances, _count_unique_without_zeros

from .timing import measure_time


@measure_time
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
        num_ref_instances = _count_unique_without_zeros(ref_mask)

        pred_labels = pred_mask
        num_pred_instances = _count_unique_without_zeros(ref_mask)

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

    # print(instance_pairs)

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
    iou_list = []  # iou list for tp instances

    for ref_idx, pred_idx in zip(ref_indices, pred_indices):
        iou = iou_matrix[ref_idx][pred_idx]
        if iou >= iou_threshold:
            tp += 1
            iou_list.append(iou)
        else:
            fp += 1

    # print(iou_list)
    fn = num_ref_instances - tp

    # Compute Segmentation Quality (SQ) as the average IoU of matched segments
    if tp == 0:
        sq = 0.0  # Set SQ to 0 when there are no true positives
    else:
        sq = np.sum(iou_list) / tp  # Average IoU for TPs

    # Calculate Recognition Quality (RQ)
    rq = tp / (tp + 0.5 * fp + 0.5 * fn)

    # Compute Panoptic Quality (PQ) as the product of SQ and RQ
    pq = sq * rq

    return pq, sq, rq, tp, fp, fn
