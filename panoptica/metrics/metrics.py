"""Metric registry: the Metric enum and its supporting value and edge-case types."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

# Import directly from the metric submodules (not the panoptica.metrics package) to
# avoid a circular import: panoptica.metrics.__init__ imports this module, so importing
# back from the package here is ordering-sensitive and breaks under import sorting.
from panoptica.metrics.assd import (
    _compute_instance_average_symmetric_surface_distance,
)
from panoptica.metrics.center_distance import _compute_instance_center_distance
from panoptica.metrics.cldice import _compute_centerline_dice
from panoptica.metrics.dice import _compute_instance_volumetric_dice
from panoptica.metrics.hausdorff_distance import (
    _compute_instance_hausdorff_distance,
    _compute_instance_hausdorff_distance95,
)
from panoptica.metrics.iou import _compute_instance_iou
from panoptica.metrics.normalized_surface_dice import (
    _compute_instance_normalized_surface_dice,
)
from panoptica.metrics.relative_absolute_volume_error import (
    _compute_instance_relative_volume_error,
)
from panoptica.metrics.relative_volume_difference import (
    _compute_instance_relative_volume_difference,
)
from panoptica.utils.constants import _Enum_Compare, auto

if TYPE_CHECKING:
    from panoptica.metrics.configured_metric import ConfiguredMetric
    from panoptica.core.result import PanopticaResult


@dataclass
class _Metric:
    """Represents a metric with a name, direction (increasing or decreasing), and a calculation function.

    This class provides a framework for defining and calculating metrics, which can be used
    to evaluate the similarity or performance between reference and prediction arrays.
    The metric direction indicates whether higher or lower values are better.

    Attributes:
        name (str): Short name of the metric.
        long_name (str): Full descriptive name of the metric.
        decreasing (bool): If True, lower metric values are better; otherwise, higher values are preferred.
        requires_spatial (bool): If True, the metric requires spatial structure for meaningful computation.
        _metric_function (Callable): A callable function that computes the metric
            between two input arrays.
        display_name (str): Short human label used when building per-instance result
            long-names (e.g. "IoU", "Dsc"). Falls back to ``name`` when empty.
        supports_pq (bool): If True, the metric gets a ``pq_{suffix}`` result column
            (only the overlap metrics IOU/DSC/clDSC do).
        sq_unit_interval (bool): If True, the per-instance sq/pq are bounded to
            ``[0, 1]`` (which also makes them eligible for AUTC integration).
        instance_order (int | None): Registration order of this metric's sq/pq
            result columns. ``None`` means the metric is not registered as an
            instance metric. This drives the file-header / result-dict column order
            and must stay fixed.
        modes (frozenset[str]): Evaluation modes the metric applies to
            ("instance" and/or "global").
        param_spec (tuple[str, ...]): Names of the fixed parameters this metric
            accepts beyond ``voxelspacing`` (e.g. NSD accepts ("threshold",)).
        supports_semantic (bool): Whether the metric is meaningful as a whole-image
            binary ("semantic"/global) metric. Distance and boundary metrics
            (ASSD/CEDI/HD/HD95) are only meaningful on a single object, so they set
            this False: they stay available per instance and per region (each region
            is one object) but are excluded from whole-image evaluation.

    Example:
        >>> my_metric = _Metric(name="accuracy", long_name="Accuracy", decreasing=False, requires_spatial=False, _metric_function=accuracy_function)
        >>> score = my_metric(reference_array, prediction_array)
        >>> print(score)

    """

    name: str
    long_name: str
    decreasing: bool
    requires_spatial: bool
    _metric_function: Callable
    suffix_override: str | None = None
    # Per-instance result-column metadata, folded in from the former
    # _INSTANCE_METRIC_SPECS table in panoptica_result.py (see #189 / #181) so the
    # metric definition is the single source of truth.
    display_name: str = ""
    supports_pq: bool = False
    sq_unit_interval: bool = False
    instance_order: int | None = None
    # Capability metadata (#181): applicable modes and accepted fixed parameters.
    modes: frozenset[str] = frozenset({"instance", "global"})
    param_spec: tuple[str, ...] = ()
    # Whether a whole-image ("semantic"/global) variant is meaningful. Single-object
    # distance/boundary metrics set this False (see attribute docstring above).
    supports_semantic: bool = True

    def __call__(
        self,
        reference_arr: np.ndarray,
        prediction_arr: np.ndarray,
        ref_instance_idx: int | None = None,
        pred_instance_idx: int | list[int] | None = None,
        *args,
        **kwargs,
    ) -> int | float:
        """Calculates the metric between reference and prediction arrays.

        Args:
            reference_arr (np.ndarray): The reference array.
            prediction_arr (np.ndarray): The prediction array.
            ref_instance_idx (int, optional): The instance index to filter in the reference array.
            pred_instance_idx (int | list[int], optional): Instance index or indices to filter in
                the prediction array.
            *args: Additional positional arguments for the metric function.
            **kwargs: Additional keyword arguments for the metric function.

        Returns:
            int | float: The computed metric value.
        """
        if ref_instance_idx is not None and pred_instance_idx is not None:
            # ``==`` and ``np.isin`` already allocate fresh arrays and never mutate
            # their inputs, so an explicit ``.copy()`` first would only double the
            # memory traffic on this hot path (runs per metric and per match pair).
            reference_arr = reference_arr == ref_instance_idx
            if isinstance(pred_instance_idx, int):
                pred_instance_idx = [pred_instance_idx]
            prediction_arr = np.isin(prediction_arr, pred_instance_idx)
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
        """Hash based on metric name, constrained to fit within 8 digits.

        Returns:
            int: The hash value of the metric.
        """
        return abs(hash(self.name)) % (10**8)

    @property
    def increasing(self):
        """Indicates if higher values of the metric are better.

        Returns:
            bool: True if increasing values are preferred, otherwise False.
        """
        return not self.decreasing

    @property
    def suffix(self) -> str:
        """Returns the override if set, otherwise defaults to '_name'"""
        return (
            f"_{self.name.lower()}"
            if self.suffix_override is None
            else self.suffix_override
        )

    def score_beats_threshold(
        self, matching_score: float, matching_threshold: float
    ) -> bool:
        """Determines if a matching score meets a specified threshold.

        Args:
            matching_score (float): The score to evaluate.
            matching_threshold (float): The threshold value to compare against.

        Returns:
            bool: True if the score meets the threshold, taking into account the
            metric's preferred direction.
        """
        return (self.increasing and matching_score >= matching_threshold) or (
            self.decreasing and matching_score <= matching_threshold
        )


class Metric(_Enum_Compare):
    """Enum containing important metrics that must be calculated in the evaluator, can be set for thresholding in matching and evaluation
    Never call the .value member here, use the properties directly.
    """

    DSC = _Metric(
        "DSC",
        "Dice",
        False,
        False,
        _compute_instance_volumetric_dice,
        display_name="Dsc",
        supports_pq=True,
        sq_unit_interval=True,
        instance_order=1,
    )
    IOU = _Metric(
        "IOU",
        "Intersection over Union",
        False,
        False,
        _compute_instance_iou,
        suffix_override="",
        display_name="IoU",
        supports_pq=True,
        sq_unit_interval=True,
        instance_order=0,
    )
    ASSD = _Metric(
        "ASSD",
        "Average Symmetric Surface Distance",
        True,
        True,
        _compute_instance_average_symmetric_surface_distance,
        display_name="ASSD",
        instance_order=3,
        supports_semantic=False,
    )
    clDSC = _Metric(
        "clDSC",
        "Centerline Dice",
        False,
        False,
        _compute_centerline_dice,
        display_name="Centerline Dsc",
        supports_pq=True,
        sq_unit_interval=True,
        instance_order=2,
    )
    RVD = _Metric(
        "RVD",
        "Relative Volume Difference",
        True,
        False,
        _compute_instance_relative_volume_difference,
        display_name="Relative Volume Difference",
        instance_order=4,
    )
    RVAE = _Metric(
        "RVAE",
        "Relative Volume Absolute Error",
        True,
        False,
        _compute_instance_relative_volume_error,
        display_name="Relative Volume Absolute Error",
        instance_order=5,
    )
    CEDI = _Metric(
        "CEDI",
        "Center Distance",
        True,
        True,
        _compute_instance_center_distance,
        display_name="Center Distance",
        instance_order=6,
        supports_semantic=False,
    )
    HD = _Metric(
        "HD",
        "Hausdorff Distance",
        True,
        True,
        _compute_instance_hausdorff_distance,
        display_name="Hausdorff Distance",
        instance_order=7,
        supports_semantic=False,
    )
    HD95 = _Metric(
        "HD95",
        "Hausdorff Distance 95",
        True,
        True,
        _compute_instance_hausdorff_distance95,
        display_name="Hausdorff Distance 95",
        instance_order=8,
        supports_semantic=False,
    )
    NSD = _Metric(
        "NSD",
        "Normalized Surface Dice",
        True,
        True,
        _compute_instance_normalized_surface_dice,
        display_name="Normalized Surface Dice",
        instance_order=9,
        param_spec=("threshold",),
    )

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
            reference_arr,
            prediction_arr,
            ref_instance_idx,
            pred_instance_idx,
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

    @property
    def requires_spatial(self):
        return self.value.requires_spatial

    @property
    def display_name(self) -> str:
        """Short human label for result long-names (falls back to ``name``)."""
        return self.value.display_name or self.value.name

    @property
    def supports_pq(self) -> bool:
        return self.value.supports_pq

    @property
    def sq_unit_interval(self) -> bool:
        return self.value.sq_unit_interval

    @property
    def instance_order(self) -> int | None:
        return self.value.instance_order

    @property
    def modes(self) -> frozenset[str]:
        return self.value.modes

    @property
    def param_spec(self) -> tuple[str, ...]:
        return self.value.param_spec

    @property
    def supports_semantic(self) -> bool:
        """Whether a whole-image binary ("global") variant of this metric is meaningful."""
        return self.value.supports_semantic

    def instance(self, **params) -> "ConfiguredMetric":
        """Configure this metric for instance-wise evaluation (e.g. ``Metric.NSD.instance(threshold=4)``)."""
        from panoptica.metrics.configured_metric import ConfiguredMetric

        return ConfiguredMetric(self, "instance", params)

    def as_global(self, **params) -> "ConfiguredMetric":
        """Configure this metric for global (whole-image binary) evaluation.

        Named ``as_global`` because ``global`` is a Python keyword.
        """
        from panoptica.metrics.configured_metric import ConfiguredMetric

        return ConfiguredMetric(self, "global", params)

    def get_result_key(self, prefix: str, is_std: bool = False) -> str:
        """
        Generates standard keys for PanopticaResult (e.g., 'sq', 'pq_dsc', 'sq_hd_std').
        """
        key = f"{prefix}{self.value.suffix}"

        if is_std:
            key += "_std"

        return key

    def __hash__(self) -> int:
        return abs(hash(self.name)) % (10**8)


class MetricMode(_Enum_Compare):
    """Different modalities from Metrics"""

    ALL = auto()
    AVG = auto()
    SUM = auto()
    STD = auto()
    MIN = auto()
    MAX = auto()


class MetricType(_Enum_Compare):
    """Different type of metrics"""

    NO_PRINT = auto()
    MATCHING = auto()
    GLOBAL = auto()
    INSTANCE = auto()


class MetricCouldNotBeComputedException(Exception):
    """Exception for when a Metric cannot be computed"""

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class Evaluation_Metric:
    """This represents a metric in the evaluation derived from other metrics or list metrics (no circular dependancies!)

    Args:
        name_id (str): code-name of this metric, must be same as the member variable of PanopticResult
        calc_func (Callable): the function to calculate this metric based on the PanopticResult object
        long_name (str | None, optional): A longer descriptive name for printing/logging purposes. Defaults to None.
        was_calculated (bool, optional): Whether this metric has been calculated or not. Defaults to False.
        error (bool, optional): If true, means the metric could not have been calculated (because dependancies do not exist or have this flag set to True). Defaults to False.
    """

    def __init__(
        self,
        name_id: str,
        metric_type: MetricType,
        calc_func: Callable | None,
        long_name: str | None = None,
        was_calculated: bool = False,
        error: bool = False,
        lower_bound: float | None = None,
        upper_bound: float | None = None,
    ):
        self.id = name_id
        self.metric_type = metric_type
        self._calc_func = calc_func
        self.long_name = long_name
        self._was_calculated = was_calculated
        self._value = None
        self._error = error
        self._error_obj: MetricCouldNotBeComputedException | None = None
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __call__(self, result_obj: "PanopticaResult") -> Any:
        """If called, needs to return its way, raise error or calculate it

        Args:
            result_obj (PanopticaResult): The result object the metric is computed from.

        Raises:
            MetricCouldNotBeComputedException: If the metric or one of its dependencies could not be computed.

        Returns:
            Any: The computed metric value.
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
            if self._was_calculated:
                raise RuntimeError(
                    f"Metric {self.id} was called to compute, but is set to have been already calculated"
                )
            if self._calc_func is None:
                raise RuntimeError(
                    f"Metric {self.id} was called to compute, but has no calculation function set"
                )
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
            self.AVG = (
                None
                if self.ALL is None or len(self.ALL) == 0
                else float(np.average(self.ALL))
            )
            self.SUM = (
                None if self.ALL is None or len(self.ALL) == 0 else np.sum(self.ALL)
            )
            self.MIN = (
                None if self.ALL is None or len(self.ALL) == 0 else np.min(self.ALL)
            )
            self.MAX = (
                None if self.ALL is None or len(self.ALL) == 0 else np.max(self.ALL)
            )

        if self.ALL is None:
            self.STD = None
        elif len(self.ALL) == 0:
            self.STD = empty_list_std
        else:
            self.STD = float(np.std(self.ALL))

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
