from dataclasses import dataclass
from enum import EnumMeta
from typing import Callable

import numpy as np

from panoptica.metrics import (
    _average_symmetric_surface_distance,
    _compute_dice_coefficient,
    _compute_iou,
    _compute_centerline_dice_coefficient,
)
from panoptica.utils.constants import _Enum_Compare, auto


@dataclass
class _MatchingMetric:
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
    ):
        if ref_instance_idx is not None and pred_instance_idx is not None:
            reference_arr = reference_arr.copy() == ref_instance_idx
            if isinstance(pred_instance_idx, int):
                pred_instance_idx = [pred_instance_idx]
            prediction_arr = np.isin(prediction_arr.copy(), pred_instance_idx)
        return self._metric_function(reference_arr, prediction_arr, *args, **kwargs)

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, _MatchingMetric):
            return self.name == __value.name
        elif isinstance(__value, str):
            return self.name == __value
        else:
            return False

    def __str__(self) -> str:
        return f"{type(self).__name__}.{self.name}"

    def __repr__(self) -> str:
        return str(self)

    @property
    def increasing(self):
        return not self.decreasing

    def score_beats_threshold(self, matching_score: float, matching_threshold: float) -> bool:
        return (self.increasing and matching_score >= matching_threshold) or (self.decreasing and matching_score <= matching_threshold)


# Important metrics that must be calculated in the evaluator, can be set for thresholding in matching and evaluation
class MatchingMetrics:
    DSC: _MatchingMetric = _MatchingMetric("DSC", False, _compute_dice_coefficient)
    IOU: _MatchingMetric = _MatchingMetric("IOU", False, _compute_iou)
    ASSD: _MatchingMetric = _MatchingMetric("ASSD", True, _average_symmetric_surface_distance)
    clDSC: _MatchingMetric = _MatchingMetric("clDSC", False, _compute_centerline_dice_coefficient)

class ListMetricMode(_Enum_Compare):
    ALL = auto()
    AVG = auto()
    SUM = auto()
    STD = auto()


class ListMetric(_Enum_Compare):
    DSC = MatchingMetrics.DSC.name
    IOU = MatchingMetrics.IOU.name
    ASSD = MatchingMetrics.ASSD.name
    clDSC = MatchingMetrics.clDSC.name

    def __hash__(self) -> int:
        return abs(hash(self.value)) % (10**8)


if __name__ == "__main__":
    print(MatchingMetrics.DSC)
    # print(MatchingMetric.DSC.name)

    print(MatchingMetrics.DSC == MatchingMetrics.DSC)
    print(MatchingMetrics.DSC == "DSC")
    print(MatchingMetrics.DSC.name == "DSC")
    #
    print(MatchingMetrics.DSC == MatchingMetrics.IOU)
    print(MatchingMetrics.DSC == "IOU")
