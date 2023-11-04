from __future__ import annotations

import concurrent.futures
import warnings
from typing import Tuple

import numpy as np

from panoptica.evaluator import Evaluator
from panoptica.result import PanopticaResult
from panoptica.timing import measure_time


class InstanceSegmentationEvaluator(Evaluator):
    """
    Evaluator for instance segmentation results.

    This class extends the Evaluator class and provides methods for evaluating instance segmentation masks
    using metrics such as Intersection over Union (IoU) and Dice coefficient.

    Methods:
        evaluate(reference_mask, prediction_mask, iou_threshold): Evaluate the instance segmentation masks.
        _unique_without_zeros(arr): Get unique non-zero values from a NumPy array.

    """

    def __init__(self):
        # TODO consider initializing evaluator with metrics it should compute
        pass

    @measure_time
    def evaluate(
        self,
        reference_mask: np.ndarray,
        prediction_mask: np.ndarray,
        iou_threshold: float,
    ) -> PanopticaResult:
        """
        Evaluate the intersection over union (IoU) and Dice coefficient for instance segmentation masks.

        Args:
            reference_mask (np.ndarray): The reference instance segmentation mask.
            prediction_mask (np.ndarray): The predicted instance segmentation mask.
            iou_threshold (float): The IoU threshold for considering a match.

        Returns:
            PanopticaResult: A named tuple containing evaluation results.
        """
        ref_labels = reference_mask
        ref_nonzero_unique_labels = self._unique_without_zeros(arr=ref_labels)
        num_ref_instances = len(ref_nonzero_unique_labels)

        pred_labels = prediction_mask
        pred_nonzero_unique_labels = self._unique_without_zeros(arr=pred_labels)
        num_pred_instances = len(pred_nonzero_unique_labels)

        self._handle_edge_cases(
            num_ref_instances=num_ref_instances,
            num_pred_instances=num_pred_instances,
        )

        # Initialize variables for True Positives (tp)
        tp, dice_list, iou_list = 0, [], []

        # Use concurrent.futures.ThreadPoolExecutor for parallelization
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    self._evaluate_instance,
                    ref_labels,
                    pred_labels,
                    ref_idx,
                    iou_threshold,
                )
                for ref_idx in ref_nonzero_unique_labels
            ]

            for future in concurrent.futures.as_completed(futures):
                tp_i, dice_i, iou_i = future.result()
                tp += tp_i
                if dice_i is not None:
                    dice_list.append(dice_i)
                if iou_i is not None:
                    iou_list.append(iou_i)

        # Create and return the PanopticaResult object with computed metrics
        return PanopticaResult(
            num_ref_instances=num_ref_instances,
            num_pred_instances=num_pred_instances,
            tp=tp,
            dice_list=dice_list,
            iou_list=iou_list,
        )

    def _evaluate_instance(
        self,
        ref_labels: np.ndarray,
        pred_labels: np.ndarray,
        ref_idx: int,
        iou_threshold: float,
    ) -> Tuple[int, float, float]:
        """
        Evaluate a single instance.

        Args:
            ref_labels (np.ndarray): Reference instance segmentation mask.
            pred_labels (np.ndarray): Predicted instance segmentation mask.
            ref_idx (int): The label of the current instance.
            iou_threshold (float): The IoU threshold for considering a match.

        Returns:
            Tuple[int, float, float]: Tuple containing True Positives (int), Dice coefficient (float), and IoU (float).
        """
        iou = self._compute_iou(
            reference=ref_labels == ref_idx,
            prediction=pred_labels == ref_idx,
        )
        if iou > iou_threshold:
            tp = 1
            dice = self._compute_dice_coefficient(
                reference=ref_labels == ref_idx,
                prediction=pred_labels == ref_idx,
            )
        else:
            tp = 0
            dice = None

        return tp, dice, iou

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
