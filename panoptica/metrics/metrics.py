from dataclasses import dataclass
from enum import EnumMeta
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from panoptica.metrics import (
    _compute_instance_average_symmetric_surface_distance,
    _compute_centerline_dice,
    _compute_instance_volumetric_dice,
    _compute_instance_iou,
    _compute_instance_relative_volume_difference,
    # _compute_instance_segmentation_tendency,
)
from panoptica.utils.constants import _Enum_Compare, auto

if TYPE_CHECKING:
    from panoptica.panoptica_result import PanopticaResult


@dataclass
class _Metric:
    """A Metric class containing a name, whether higher or lower values is better, and a function to calculate that metric between two instances in an array"""

    name: str
    long_name: str
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

    DSC = _Metric("DSC", "Dice", False, _compute_instance_volumetric_dice)
    IOU = _Metric("IOU", "Intersection over Union", False, _compute_instance_iou)
    ASSD = _Metric(
        "ASSD",
        "Average Symmetric Surface Distance",
        True,
        _compute_instance_average_symmetric_surface_distance,
    )
    clDSC = _Metric("clDSC", "Centerline Dice", False, _compute_centerline_dice)
    RVD = _Metric(
        "RVD",
        "Relative Volume Difference",
        True,
        _compute_instance_relative_volume_difference,
    )
    # ST = _Metric("ST", False, _compute_instance_segmentation_tendency)

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
    MIN = auto()
    MAX = auto()


class MetricType(_Enum_Compare):
    """Different type of metrics

    Args:
        _Enum_Compare (_type_): _description_
    """

    NO_PRINT = auto()
    MATCHING = auto()
    GLOBAL = auto()
    INSTANCE = auto()


class MetricCouldNotBeComputedException(Exception):
    """Exception for when a Metric cannot be computed"""

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class Evaluation_Metric:
    def __init__(
        self,
        name_id: str,
        metric_type: MetricType,
        calc_func: Callable | None,
        long_name: str | None = None,
        was_calculated: bool = False,
        error: bool = False,
    ):
        """This represents a metric in the evaluation derived from other metrics or list metrics (no circular dependancies!)

        Args:
            name_id (str): code-name of this metric, must be same as the member variable of PanopticResult
            calc_func (Callable): the function to calculate this metric based on the PanopticResult object
            long_name (str | None, optional): A longer descriptive name for printing/logging purposes. Defaults to None.
            was_calculated (bool, optional): Whether this metric has been calculated or not. Defaults to False.
            error (bool, optional): If true, means the metric could not have been calculated (because dependancies do not exist or have this flag set to True). Defaults to False.
        """
        self.id = name_id
        self.metric_type = metric_type
        self._calc_func = calc_func
        self.long_name = long_name
        self._was_calculated = was_calculated
        self._value = None
        self._error = error
        self._error_obj: MetricCouldNotBeComputedException | None = None

    def __call__(self, result_obj: "PanopticaResult") -> Any:
        """If called, needs to return its way, raise error or calculate it

        Args:
            result_obj (PanopticaResult): _description_

        Raises:
            MetricCouldNotBeComputedException: _description_
            self._error_obj: _description_

        Returns:
            Any: _description_
        """
        # ERROR
        if self._error:
            if self._error_obj is None:
                self._error_obj = MetricCouldNotBeComputedException(
                    f"Metric {self.id} requested, but could not be computed"
                )
            raise self._error_obj
        # Already calculated?
        if self._was_calculated:
            return self._value

        # Calculate it
        try:
            assert (
                not self._was_calculated
            ), f"Metric {self.id} was called to compute, but is set to have been already calculated"
            assert (
                self._calc_func is not None
            ), f"Metric {self.id} was called to compute, but has no calculation function set"
            value = self._calc_func(result_obj)
        except MetricCouldNotBeComputedException as e:
            value = e
            self._error = True
            self._error_obj = e
        self._was_calculated = True

        self._value = value
        return self._value

    def __str__(self) -> str:
        if self.long_name is not None:
            return self.long_name + f" ({self.id})"
        else:
            return self.id


class Evaluation_List_Metric:
    def __init__(
        self,
        name_id: Metric,
        empty_list_std: float | None,
        value_list: list[float] | None,  # None stands for not calculated
        is_edge_case: bool = False,
        edge_case_result: float | None = None,
    ):
        """This represents the metrics resulting from a Metric calculated between paired instances (IoU, ASSD, Dice, ...)

        Args:
            name_id (Metric): code-name of this metric
            empty_list_std (float): Value for the standard deviation if the list of values is empty
            value_list (list[float] | None): List of values of that metric (only the TPs)
        """
        self.id = name_id
        self.error = value_list is None
        self.ALL: list[float] | None = value_list
        if is_edge_case:
            self.AVG: float | None = edge_case_result
            self.SUM: None | float = edge_case_result
            self.MIN: None | float = edge_case_result
            self.MAX: None | float = edge_case_result
        else:
            self.AVG = None if self.ALL is None else np.average(self.ALL)
            self.SUM = None if self.ALL is None else np.sum(self.ALL)
            self.MIN = (
                None if self.ALL is None or len(self.ALL) == 0 else np.min(self.ALL)
            )
            self.MAX = (
                None if self.ALL is None or len(self.ALL) == 0 else np.max(self.ALL)
            )

        self.STD = (
            None
            if self.ALL is None
            else empty_list_std if len(self.ALL) == 0 else np.std(self.ALL)
        )

    def __getitem__(self, mode: MetricMode | str):
        if self.error:
            raise MetricCouldNotBeComputedException(
                f"Metric {self.id} has not been calculated, add it to your eval_metrics"
            )
        if isinstance(mode, MetricMode):
            mode = mode.name
        if hasattr(self, mode):
            return getattr(self, mode)
        else:
            raise MetricCouldNotBeComputedException(
                f"List_Metric {self.id} does not contain {mode} member"
            )


if __name__ == "__main__":
    print(Metric.DSC)
    # print(MatchingMetric.DSC.name)

    print(Metric.DSC == Metric.DSC)
    print(Metric.DSC == "DSC")
    print(Metric.DSC.name == "DSC")
    #
    print(Metric.DSC == Metric.IOU)
    print(Metric.DSC == "IOU")
    print(Metric.DSC == "IOU")
