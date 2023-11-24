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
        assd_list: List[float],
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
        self._assd_list = assd_list

    def __str__(self):
        return (
            f"Number of instances in prediction: {self.num_pred_instances}\n"
            f"Number of instances in reference: {self.num_ref_instances}\n"
            f"True Positives (tp): {self.tp}\n"
            f"False Positives (fp): {self.fp}\n"
            f"False Negatives (fn): {self.fn}\n"
            f"Recognition Quality / F1 Score (RQ): {self.rq}\n"
            f"Segmentation Quality (SQ): {self.sq} ± {self.sq_sd}\n"
            f"Panoptic Quality (PQ): {self.pq}\n"
            f"DSC-based Segmentation Quality (DQ_DSC): {self.sq_dsc} ± {self.sq_dsc_sd}\n"
            f"DSC-based Panoptic Quality (PQ_DSC): {self.pq_dsc}\n"
            f"Average symmetric surface distance (ASSD): {self.sq_assd} ± {self.sq_assd_sd}\n"
            f"ASSD-based Panoptic Quality (PQ_ASSD): {self.pq_assd}"
        )

    def to_dict(self):
        return {
            "num_pred_instances": self.num_pred_instances,
            "num_ref_instances": self.num_ref_instances,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "rq": self.rq,
            "sq": self.sq,
            "sq_sd": self.sq_sd,
            "pq": self.pq,
            "sq_dsc": self.sq_dsc,
            "sq_dsc_sd": self.sq_dsc_sd,
            "pq_dsc": self.pq_dsc,
            "sq_assd": self.sq_assd,
            "sq_assd_sd": self.sq_assd_sd,
            "pq_assd": self.pq_assd,
        }

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
            return 0.0 if self.num_pred_instances + self.num_ref_instances > 0 else np.nan
        return self.tp / (self.tp + 0.5 * self.fp + 0.5 * self.fn)

    @property
    def sq(self) -> float:
        """
        Calculate the Segmentation Quality (SQ) based on IoU values.

        Returns:
            float: Segmentation Quality (SQ).
        """
        if self.tp == 0:
            return 0.0 if self.num_pred_instances + self.num_ref_instances > 0 else np.nan
        return np.sum(self._iou_list) / self.tp

    @property
    def sq_sd(self) -> float:
        """
        Calculate the standard deviation of Segmentation Quality (SQ) based on IoU values.

        Returns:
            float: Standard deviation of Segmentation Quality (SQ).
        """
        return np.std(self._iou_list) if len(self._iou_list) > 0 else np.nan

    @property
    def pq(self) -> float:
        """
        Calculate the Panoptic Quality (PQ) based on SQ and RQ.

        Returns:
            float: Panoptic Quality (PQ).
        """
        return self.sq * self.rq

    @property
    def sq_dsc(self) -> float:
        """
        Calculate the average Dice coefficient for matched instances. Analogue to segmentation quality but based on DSC.

        Returns:
            float: Average Dice coefficient.
        """
        if self.tp == 0:
            return 0.0 if self.num_pred_instances + self.num_ref_instances > 0 else np.nan
        return np.sum(self._dice_list) / self.tp

    @property
    def sq_dsc_sd(self) -> float:
        """
        Calculate the standard deviation of average Dice coefficient for matched instances. Analogue to segmentation quality but based on DSC.

        Returns:
            float: Standard deviation of Average Dice coefficient.
        """
        return np.std(self._dice_list) if len(self._dice_list) > 0 else np.nan

    @property
    def pq_dsc(self) -> float:
        """
        Calculate the Panoptic Quality (PQ) based on DSC-based SQ and RQ.

        Returns:
            float: Panoptic Quality (PQ).
        """
        return self.sq_dsc * self.rq

    @property
    def sq_assd(self) -> float:
        """
        Calculate the average average symmetric surface distance (ASSD) for matched instances. Analogue to segmentation quality but based on ASSD.

        Returns:
            float: average symmetric surface distance. (ASSD)
        """
        if self.tp == 0:
            return np.nan if self.num_pred_instances + self.num_ref_instances == 0 else np.inf
        return np.sum(self._assd_list) / self.tp

    @property
    def sq_assd_sd(self) -> float:
        """
        Calculate the standard deviation of average symmetric surface distance (ASSD) for matched instances. Analogue to segmentation quality but based on ASSD.
        Returns:
            float: Standard deviation of average symmetric surface distance (ASSD).
        """
        return np.std(self._assd_list) if len(self._assd_list) > 0 else np.nan

    @property
    def pq_assd(self) -> float:
        """
        Calculate the Panoptic Quality (PQ) based on ASSD-based SQ and RQ.

        Returns:
            float: Panoptic Quality (PQ).
        """
        return self.sq_assd * self.rq
