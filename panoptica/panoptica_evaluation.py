from __future__ import annotations
from multiprocessing import Pool

import numpy as np
from scipy.optimize import linear_sum_assignment

from .utils import (
    _label_instances,
    _compute_instance_iou,
    _compute_instance_volumetric_dice,
)
from .result import PanopticaResult

from .timing import measure_time


@measure_time
def panoptica_evaluation(
    ref_mask: np.ndarray,
    pred_mask: np.ndarray,
    modus: str,
    iou_threshold: float = 0.5,
    cca_backend: str = "cc3d",
) -> PanopticaResult:
    """
    Compute Panoptic Quality (PQ), Segmentation Quality (SQ), Recognition Quality (RQ),
    and instance Dice metric for binary masks.

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
        PanopticaResult: An object containing PQ, SQ, RQ, and other metrics.
    """

    # Handle different processing modes
    if modus == "im":
        print("please use the object oriented interface for the moment:")
        print("from panoptica.instance import InstanceSegmentationEvaluator")

    elif modus == "cc":
        ref_labels, num_ref_instances = _label_instances(
            mask=ref_mask, cca_backend=cca_backend
        )
        pred_labels, num_pred_instances = _label_instances(
            mask=pred_mask, cca_backend=cca_backend
        )

    # Handle cases where either the reference or the prediction is empty
    if num_ref_instances == 0 and num_pred_instances == 0:
        # Both references and predictions are empty, perfect match
        return PanopticaResult(
            num_ref_instances=0,
            num_pred_instances=0,
            tp=0,
            dice_list=[],
            iou_list=[],
        )
    elif num_ref_instances == 0:
        # All references are missing, only false positives
        return PanopticaResult(
            num_ref_instances=0,
            num_pred_instances=num_pred_instances,
            tp=0,
            dice_list=[],
            iou_list=[],
        )
    elif num_pred_instances == 0:
        # All predictions are missing, only false negatives
        return PanopticaResult(
            num_ref_instances=num_ref_instances,
            num_pred_instances=0,
            tp=0,
            dice_list=[],
            iou_list=[],
        )

    # Create a pool of worker processes to parallelize the computation
    with Pool() as pool:
        # Generate all possible pairs of instance indices for IoU computation
        instance_pairs = [
            (ref_labels, pred_labels, ref_idx, pred_idx)
            for ref_idx in range(1, num_ref_instances + 1)
            for pred_idx in range(1, num_pred_instances + 1)
        ]

        # Calculate IoU for all instance pairs in parallel using starmap
        iou_values = pool.starmap(_compute_instance_iou, instance_pairs)

    # Reshape the resulting IoU values into a matrix
    iou_matrix = np.array(iou_values).reshape((num_ref_instances, num_pred_instances))

    # Use linear_sum_assignment to find the best matches
    ref_indices, pred_indices = linear_sum_assignment(-iou_matrix)

    # Initialize variables for True Positives (tp) and False Positives (fp)
    tp, dice_list, iou_list = 0, [], []

    # Loop through matched instances to compute PQ components
    for ref_idx, pred_idx in zip(ref_indices, pred_indices):
        iou = iou_matrix[ref_idx][pred_idx]
        if iou >= iou_threshold:
            # Match found, increment true positive count and collect IoU and Dice values
            tp += 1
            iou_list.append(iou)

            # Compute Dice for matched instances
            dice = _compute_instance_volumetric_dice(
                ref_labels=ref_labels,
                pred_labels=pred_labels,
                ref_instance_idx=ref_idx + 1,
                pred_instance_idx=pred_idx + 1,
            )
            dice_list.append(dice)

    # Create and return the PanopticaResult object with computed metrics
    return PanopticaResult(
        num_ref_instances=num_ref_instances,
        num_pred_instances=num_pred_instances,
        tp=tp,
        dice_list=dice_list,
        iou_list=iou_list,
    )
