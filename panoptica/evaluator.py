from abc import ABC, abstractmethod
import numpy as np
from .result import PanopticaResult


class Evaluator(ABC):
    def __init__(
        self,
    ):
        pass

    @abstractmethod
    def evaluate(
        self,
        reference_mask: np.ndarray,
        prediction_mask: np.ndarray,
        iou_threshold: float,
    ):
        pass

    @abstractmethod
    def count_number_of_instances(
        self,
        mask: np.ndarray,
    ):
        pass

    def _handle_edge_cases(
        self,
        num_ref_instances: int,
        num_pred_instances: int,
    ):
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

    def _compute_iou(
        self,
        reference: np.ndarray,
        prediction: np.ndarray,
    ) -> float:
        """
        Compute Intersection over Union (IoU) between two masks.

        Args:
            reference (np.ndarray): Reference mask.
            prediction (np.ndarray): Prediction mask.

        Returns:
            float: IoU between the two masks.
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
