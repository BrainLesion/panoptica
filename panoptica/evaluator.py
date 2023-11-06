from abc import ABC, abstractmethod

import numpy as np

from panoptica.result import PanopticaResult


class Evaluator(ABC):
    """
    Abstract base class for evaluating instance segmentation results.

    Subclasses of this Evaluator should implement the abstract methods 'evaluate' and 'count_number_of_instances'.
    """

    def __init__(self):
        pass

    @abstractmethod
    def evaluate(
        self,
        reference_mask: np.ndarray,
        prediction_mask: np.ndarray,
        iou_threshold: float,
    ) -> PanopticaResult:
        """
        Evaluate the instance segmentation results based on the reference and prediction masks.

        Args:
            reference_mask (np.ndarray): Binary mask representing reference instances.
            prediction_mask (np.ndarray): Binary mask representing prediction instances.
            iou_threshold (float): IoU threshold for considering a prediction as a true positive.

        Returns:
            PanopticaResult: Result object with evaluation metrics.
        """
        pass

    def _handle_edge_cases(
        self, num_ref_instances: int, num_pred_instances: int
    ) -> PanopticaResult:
        """
        Handle edge cases when comparing reference and prediction masks.

        Args:
            num_ref_instances (int): Number of instances in the reference mask.
            num_pred_instances (int): Number of instances in the prediction mask.

        Returns:
            PanopticaResult: Result object with evaluation metrics.
        """
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
        if num_ref_instances == 0:
            # All references are missing, only false positives
            return PanopticaResult(
                num_ref_instances=0,
                num_pred_instances=num_pred_instances,
                tp=0,
                dice_list=[],
                iou_list=[],
            )
        if num_pred_instances == 0:
            # All predictions are missing, only false negatives
            return PanopticaResult(
                num_ref_instances=num_ref_instances,
                num_pred_instances=0,
                tp=0,
                dice_list=[],
                iou_list=[],
            )

    def _compute_iou(self, reference: np.ndarray, prediction: np.ndarray) -> float:
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
        self,
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
