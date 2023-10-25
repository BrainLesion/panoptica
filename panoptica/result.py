from __future__ import annotations
from typing import List
import numpy as np


class PanopticaResult:
    """
    Represents the result of the Panoptic Quality (PQ) computation.

    Attributes:
        num_ref_instances (int): Number of reference instances.
        num_pred_instances (int): Number of predicted instances.
        tp (int): Number of correctly matched instances (True Positives).
        fp (int): Number of extra predicted instances (False Positives).
    """

    def __init__(
        self,
        num_ref_instances: int,
        num_pred_instances: int,
        tp: int,
        dice_list: List[float],
        iou_list: List[float],
    ):
        """
        Initialize a PanopticaResult object.

        Args:
            num_ref_instances (int): Number of reference instances.
            num_pred_instances (int): Number of predicted instances.
            tp (int): Number of correctly matched instances (True Positives).
            dice_list (List[float]): List of Dice coefficients for matched instances.
            iou_list (List[float]): List of IoU values for matched instances.
        """
        self._tp = tp
        self._dice_list = dice_list
        self._iou_list = iou_list
        self._num_ref_instances = num_ref_instances
        self._num_pred_instances = num_pred_instances

    def __str__(self):
        return (
            f"Panoptic Quality (PQ): {self.pq}\n"
            f"Segmentation Quality (SQ): {self.sq}\n"
            f"Recognition Quality (RQ): {self.rq}\n"
            f"True Positives (tp): {self.tp}\n"
            f"False Positives (fp): {self.fp}\n"
            f"False Negatives (fn): {self.fn}\n"
            f"instance_dice: {self.instance_dice}\n"
            f"Number of instances in prediction: {self.num_pred_instances}\n"
            f"Number of instances in reference: {self.num_ref_instances}"
        )

    @property
    def num_ref_instances(self) -> int:
        """
        Get the number of reference instances.

        Returns:
            int: Number of reference instances.
        """
        return self._num_ref_instances

    @property
    def num_pred_instances(self) -> int:
        """
        Get the number of predicted instances.

        Returns:
            int: Number of predicted instances.
        """
        return self._num_pred_instances

    @property
    def tp(self) -> int:
        """
        Calculate the number of True Positives (TP).

        Returns:
            int: Number of True Positives.
        """
        return self._tp

    @property
    def fp(self) -> int:
        """
        Calculate the number of False Positives (FP).

        Returns:
            int: Number of False Positives.
        """
        return self.num_pred_instances - self.tp

    @property
    def fn(self) -> int:
        """
        Calculate the number of False Negatives (FN).

        Returns:
            int: Number of False Negatives.
        """
        return self.num_ref_instances - self.tp

    @property
    def rq(self) -> float:
        """
        Calculate the Recognition Quality (RQ) based on TP, FP, and FN.

        Returns:
            float: Recognition Quality (RQ).
        """
        if self.tp == 0:
            return 0.0
        return self.tp / (self.tp + 0.5 * self.fp + 0.5 * self.fn)

    @property
    def sq(self) -> float:
        """
        Calculate the Segmentation Quality (SQ) based on IoU values.

        Returns:
            float: Segmentation Quality (SQ).
        """
        if self.tp == 0:
            return 0.0
        return np.sum(self._iou_list) / self.tp

    @property
    def pq(self) -> float:
        """
        Calculate the Panoptic Quality (PQ) based on SQ and RQ.

        Returns:
            float: Panoptic Quality (PQ).
        """
        return self.sq * self.rq

    @property
    def instance_dice(self) -> float:
        """
        Calculate the average Dice coefficient for matched instances.

        Returns:
            float: Average Dice coefficient.
        """
        return np.mean(self._dice_list)
