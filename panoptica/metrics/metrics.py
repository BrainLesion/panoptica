from dataclasses import dataclass
from enum import EnumMeta
from typing import Callable

import numpy as np

from panoptica.metrics import (
    _average_symmetric_surface_distance,
    _compute_dice_coefficient,
    _compute_iou,
)
from panoptica.utils.constants import Enum, _Enum_Compare, auto


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

    def score_beats_threshold(
        self, matching_score: float, matching_threshold: float
    ) -> bool:
        return (self.increasing and matching_score >= matching_threshold) or (
            self.decreasing and matching_score <= matching_threshold
        )


# class _EnumMeta(EnumMeta):
#    def __getattribute__(cls, name) -> MatchingMetric:
#        value = super().__getattribute__(name)
#        if isinstance(value, cls):
#            value = value.value
#        return value


# Important metrics that must be calculated in the evaluator, can be set for thresholding in matching and evaluation
# TODO make abstract class for metric, make enum with references to these classes for referenciation and user exposure
class Metrics:
    # TODO make this with meta above, and then it can function without the double name, right?
    DSC = _MatchingMetric("DSC", False, _compute_dice_coefficient)
    IOU = _MatchingMetric("IOU", False, _compute_iou)
    ASSD = _MatchingMetric("ASSD", True, _average_symmetric_surface_distance)
    # These are all lists of values


class ListMetric(_Enum_Compare):
    DSC = Metrics.DSC.name
    IOU = Metrics.IOU.name
    ASSD = Metrics.ASSD.name

    def __hash__(self) -> int:
        return abs(hash(self.value)) % (10**8)


# Metrics that are derived from list metrics and can be calculated later
# TODO map result properties to this enum
class EvalMetric(_Enum_Compare):
    TP = auto()
    FP = auto()
    FN = auto()
    RQ = auto()
    DQ_DSC = auto()
    PQ_DSC = auto()
    ASSD = auto()
    PQ_ASSD = auto()


MetricDict = dict[ListMetric | EvalMetric | str, float | list[float]]


list_of_applicable_std_metrics: list[EvalMetric] = [
    EvalMetric.RQ,
    EvalMetric.DQ_DSC,
    EvalMetric.PQ_ASSD,
    EvalMetric.ASSD,
    EvalMetric.PQ_ASSD,
]


if __name__ == "__main__":
    print(Metrics.DSC)
    # print(MatchingMetric.DSC.name)

    print(Metrics.DSC == Metrics.DSC)
    print(Metrics.DSC == "DSC")
    print(Metrics.DSC.name == "DSC")
    #
    print(Metrics.DSC == Metrics.IOU)
    print(Metrics.DSC == "IOU")
