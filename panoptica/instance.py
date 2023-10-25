from __future__ import annotations
from typing import Tuple
from multiprocessing import Pool
import warnings

import numpy as np


from scipy.optimize import linear_sum_assignment

from .timing import measure_time


from .evaluator import Evaluator
from .result import PanopticaResult


class InstanceSegmentationEvaluator(Evaluator):
    def __init__(self):
        pass

    @measure_time
    def evaluate(
        self,
        reference_mask: np.ndarray,
        prediction_mask: np.ndarray,
        iou_threshold: float,
    ):
        ref_labels = reference_mask
        ref_nonzero_unique_labels = self._unique_without_zeros(arr=ref_labels)
        num_ref_instances = len(ref_nonzero_unique_labels)

        pred_labels = prediction_mask
        pred_nonzero_unique_labels = self._unique_without_zeros(arr=ref_labels)
        num_pred_instances = len(pred_nonzero_unique_labels)

        self._handle_edge_cases(
            num_ref_instances=num_ref_instances,
            num_pred_instances=num_pred_instances,
        )

        # Initialize variables for True Positives (tp) and False Positives (fp)
        tp, fp, dice_list, iou_list = 0, 0, [], []

        # begin iou computation
        for ref_idx in ref_nonzero_unique_labels:
            pass
        # Create a pool of worker processes to parallelize the computation

        with Pool() as pool:
            # Generate all possible pairs of instance indices for IoU computation
            instance_pairs = [
                (ref_labels, pred_labels, ref_idx, pred_idx)
                for ref_idx in range(1, num_ref_instances + 1)
                for pred_idx in range(1, num_pred_instances + 1)
            ]

            # Calculate IoU for all instance pairs in parallel using starmap
            iou_values = pool.starmap(self._compute_instance_iou, instance_pairs)

        # Reshape the resulting IoU values into a matrix
        iou_matrix = np.array(iou_values).reshape(
            (num_ref_instances, num_pred_instances)
        )

        # Use linear_sum_assignment to find the best matches
        ref_indices, pred_indices = linear_sum_assignment(-iou_matrix)

        # Initialize variables for True Positives (tp) and False Positives (fp)
        tp, fp, dice_list, iou_list = 0, 0, [], []

        # Loop through matched instances to compute PQ components
        for ref_idx, pred_idx in zip(ref_indices, pred_indices):
            iou = iou_matrix[ref_idx][pred_idx]
            if iou >= iou_threshold:
                # Match found, increment true positive count and collect IoU and Dice values
                tp += 1
                iou_list.append(iou)

                # Compute Dice for matched instances
                dice = self._compute_instance_volumetric_dice(
                    ref_labels=ref_labels,
                    pred_labels=pred_labels,
                    ref_instance_idx=ref_idx + 1,
                    pred_instance_idx=pred_idx + 1,
                )
                dice_list.append(dice)
            else:
                # No match found, increment false positive count
                fp += 1

        # Create and return the PanopticaResult object with computed metrics
        return PanopticaResult(
            num_ref_instances=num_ref_instances,
            num_pred_instances=num_pred_instances,
            tp=tp,
            fp=fp,
            dice_list=dice_list,
            iou_list=iou_list,
        )

    def _unique_without_zeros(self, arr: np.ndarray) -> np.ndarray:
        """
        Get unique non-zero values from a NumPy array.

        Parameters:
            arr (np.ndarray): Input NumPy array.

        Returns:
            np.ndarray: Unique non-zero values from the input array.

        Issues a warning if negative values are present.
        """
        if np.any(arr < 0):
            warnings.warn("Negative values are present in the input array.")

        return np.unique(arr[arr != 0])
