from dataclasses import dataclass
from enum import EnumMeta, Enum
from typing import Any, Callable

import numpy as np

from panoptica.metrics import (
    _average_symmetric_surface_distance,
    _compute_dice_coefficient,
    _compute_iou,
    _compute_centerline_dice_coefficient,
)
from panoptica.utils.constants import _Enum_Compare, auto


@dataclass
class _Metric:
    """A Metric class containing a name, whether higher or lower values is better, and a function to calculate that metric between two instances in an array"""

    name: str
    decreasing: bool
    _metric_function: Callable

    def __call__(
        self,
        reference_arr: np.ndarray,
        prediction_arr: np.ndarray,
        ref_instance_idx: int | None = None,
        pred_instance_idx: int | list[int] | None = None,
        *args,
        **kwargs,
    ) -> int | float:
        if ref_instance_idx is not None and pred_instance_idx is not None:
            reference_arr = reference_arr.copy() == ref_instance_idx
            if isinstance(pred_instance_idx, int):
                pred_instance_idx = [pred_instance_idx]
            prediction_arr = np.isin(
                prediction_arr.copy(), pred_instance_idx
            )  # type:ignore
        return self._metric_function(reference_arr, prediction_arr, *args, **kwargs)

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, _Metric):
            return self.name == __value.name
        elif isinstance(__value, str):
            return self.name == __value
        else:
            return False

    def __str__(self) -> str:
        return f"{type(self).__name__}.{self.name}"

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self) -> int:
        return abs(hash(self.name)) % (10**8)

    @property
    def increasing(self):
        return not self.decreasing

    def score_beats_threshold(
        self, matching_score: float, matching_threshold: float
    ) -> bool:
        return (self.increasing and matching_score >= matching_threshold) or (
            self.decreasing and matching_score <= matching_threshold
        )


class DirectValueMeta(EnumMeta):
    "Metaclass that allows for directly getting an enum attribute"

    def __getattribute__(cls, name) -> _Metric:
        value = super().__getattribute__(name)
        if isinstance(value, cls):
            value = value.value
        return value


class Metric(_Enum_Compare):
    """Enum containing important metrics that must be calculated in the evaluator, can be set for thresholding in matching and evaluation
    Never call the .value member here, use the properties directly

    Returns:
        _type_: _description_
    """

    DSC = _Metric("DSC", False, _compute_dice_coefficient)
    IOU = _Metric("IOU", False, _compute_iou)
    ASSD = _Metric("ASSD", True, _average_symmetric_surface_distance)
    clDSC = _Metric("clDSC", False, _compute_centerline_dice_coefficient)

    def __call__(
        self,
        reference_arr: np.ndarray,
        prediction_arr: np.ndarray,
        ref_instance_idx: int | None = None,
        pred_instance_idx: int | list[int] | None = None,
        *args,
        **kwargs,
    ) -> int | float:
        """Calculates the underlaying metric

        Args:
            reference_arr (np.ndarray): Reference array
            prediction_arr (np.ndarray): Prediction array
            ref_instance_idx (int | None, optional): The index label to be evaluated for the reference. Defaults to None.
            pred_instance_idx (int | list[int] | None, optional): The index label to be evaluated for the prediction. Defaults to None.

        Returns:
            int | float: The metric value
        """
        return self.value(
            reference_arr=reference_arr,
            prediction_arr=prediction_arr,
            ref_instance_idx=ref_instance_idx,
            pred_instance_idx=pred_instance_idx,
            *args,
            **kwargs,
        )

    def score_beats_threshold(
        self, matching_score: float, matching_threshold: float
    ) -> bool:
        """Calculates whether a score beats a specified threshold

        Args:
            matching_score (float): Metric score
            matching_threshold (float): Threshold to compare against

        Returns:
            bool: True if the matching_score beats the threshold, False otherwise.
        """
        return (self.increasing and matching_score >= matching_threshold) or (
            self.decreasing and matching_score <= matching_threshold
        )

    @property
    def name(self):
        return self.value.name

    @property
    def decreasing(self):
        return self.value.decreasing

    @property
    def increasing(self):
        return self.value.increasing

    def __hash__(self) -> int:
        return abs(hash(self.name)) % (10**8)


class MetricMode(_Enum_Compare):
    """Different modalities from Metrics

    Args:
        _Enum_Compare (_type_): _description_
    """

    ALL = auto()
    AVG = auto()
    SUM = auto()
    STD = auto()


if __name__ == "__main__":
    print(Metric.DSC)
    # print(MatchingMetric.DSC.name)

    print(Metric.DSC == Metric.DSC)
    print(Metric.DSC == "DSC")
    print(Metric.DSC.name == "DSC")
    #
    print(Metric.DSC == Metric.IOU)
    print(Metric.DSC == "IOU")
