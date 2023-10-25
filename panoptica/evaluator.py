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
                fp=0,
                dice_list=[],
                iou_list=[],
            )
        if num_ref_instances == 0:
            # All references are missing, only false positives
            return PanopticaResult(
                num_ref_instances=0,
                num_pred_instances=num_pred_instances,
                tp=0,
                fp=num_pred_instances,
                dice_list=[],
                iou_list=[],
            )
        if num_pred_instances == 0:
            # All predictions are missing, only false negatives
            return PanopticaResult(
                num_ref_instances=num_ref_instances,
                num_pred_instances=0,
                tp=0,
                fp=0,
                dice_list=[],
                iou_list=[],
            )

    def _compute_instance_volumetric_dice(
        self,
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
