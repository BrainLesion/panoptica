from __future__ import annotations

from typing import Any, Callable
import numpy as np
from panoptica.metrics import MetricMode, Metric
from panoptica.metrics import (
    _compute_dice_coefficient,
    _compute_centerline_dice_coefficient,
)
from panoptica.utils import EdgeCaseHandler
from panoptica.utils.processing_pair import MatchedInstancePair


class MetricCouldNotBeComputedException(Exception):
    """Exception for when a Metric cannot be computed"""

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class Evaluation_Metric:
    def __init__(
        self,
        name_id: str,
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
        self.calc_func = calc_func
        self.long_name = long_name
        self.was_calculated = was_calculated
        self.error = error
        self.error_obj: MetricCouldNotBeComputedException | None = None

    def __call__(self, result_obj: PanopticaResult) -> Any:
        if self.error:
            if self.error_obj is None:
                raise MetricCouldNotBeComputedException(
                    f"Metric {self.id} requested, but could not be computed"
                )
            else:
                raise self.error_obj
        assert (
            not self.was_calculated
        ), f"Metric {self.id} was called to compute, but is set to have been already calculated"
        assert (
            self.calc_func is not None
        ), f"Metric {self.id} was called to compute, but has no calculation function set"
        try:
            value = self.calc_func(result_obj)
        except MetricCouldNotBeComputedException as e:
            value = e
            self.error = True
            self.error_obj = e
        return value

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
        else:
            self.AVG = None if self.ALL is None else np.average(self.ALL)
            self.SUM = None if self.ALL is None else np.sum(self.ALL)
        self.STD = (
            None
            if self.ALL is None
            else empty_list_std
            if len(self.ALL) == 0
            else np.std(self.ALL)
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


class PanopticaResult(object):
    def __init__(
        self,
        reference_arr: np.ndarray,
        prediction_arr: np.ndarray,
        # TODO some metadata object containing dtype, voxel spacing, ...
        num_pred_instances: int,
        num_ref_instances: int,
        tp: int,
        list_metrics: dict[Metric, list[float]],
        edge_case_handler: EdgeCaseHandler,
    ):
        """Result object for Panoptica, contains all calculatable metrics

        Args:
            reference_arr (np.ndarray): matched reference arr
            prediction_arr (np.ndarray): matched prediction arr
            num_pred_instances (int): number of prediction instances
            num_ref_instances (int): number of reference instances
            tp (int): number of true positives (matched instances)
            list_metrics (dict[Metric, list[float]]): dictionary containing the metrics for each TP
            edge_case_handler (EdgeCaseHandler): EdgeCaseHandler object that handles various forms of edge cases
        """
        self._edge_case_handler = edge_case_handler
        empty_list_std = self._edge_case_handler.handle_empty_list_std()
        self._prediction_arr = prediction_arr
        self._reference_arr = reference_arr
        ######################
        # Evaluation Metrics #
        ######################
        self._evaluation_metrics: dict[str, Evaluation_Metric] = {}
        #
        # region Already Calculated
        self.num_ref_instances: int
        self._add_metric(
            "num_ref_instances",
            None,
            long_name="Number of instances in reference",
            default_value=num_ref_instances,
            was_calculated=True,
        )
        self.num_pred_instances: int
        self._add_metric(
            "num_pred_instances",
            None,
            long_name="Number of instances in prediction",
            default_value=num_pred_instances,
            was_calculated=True,
        )
        self.tp: int
        self._add_metric(
            "tp",
            None,
            long_name="True Positives",
            default_value=tp,
            was_calculated=True,
        )
        # endregion
        #
        # region Basic
        self.fp: int
        self._add_metric(
            "fp",
            fp,
            long_name="False Positives",
        )
        self.fn: int
        self._add_metric(
            "fn",
            fn,
            long_name="False Negatives",
        )
        self.rq: float
        self._add_metric(
            "rq",
            rq,
            long_name="Recognition Quality",
        )
        # endregion
        #
        # region Global
        self.global_bin_dsc: int
        self._add_metric(
            "global_bin_dsc",
            global_bin_dsc,
            long_name="Global Binary Dice",
        )
        #
        self.global_bin_cldsc: int
        self._add_metric(
            "global_bin_cldsc",
            global_bin_cldsc,
            long_name="Global Binary Centerline Dice",
        )
        # endregion
        #
        # region IOU
        self.sq: float
        self._add_metric(
            "sq",
            sq,
            long_name="Segmentation Quality IoU",
        )
        self.sq_std: float
        self._add_metric(
            "sq_std",
            sq_std,
            long_name="Segmentation Quality IoU Standard Deviation",
        )
        self.pq: float
        self._add_metric(
            "pq",
            pq,
            long_name="Panoptic Quality IoU",
        )
        # endregion
        #
        # region DICE
        self.sq_dsc: float
        self._add_metric(
            "sq_dsc",
            sq_dsc,
            long_name="Segmentation Quality Dsc",
        )
        self.sq_dsc_std: float
        self._add_metric(
            "sq_dsc_std",
            sq_dsc_std,
            long_name="Segmentation Quality Dsc Standard Deviation",
        )
        self.pq_dsc: float
        self._add_metric(
            "pq_dsc",
            pq_dsc,
            long_name="Panoptic Quality Dsc",
        )
        # endregion
        #
        # region clDICE
        self.sq_cldsc: float
        self._add_metric(
            "sq_cldsc",
            sq_cldsc,
            long_name="Segmentation Quality Centerline Dsc",
        )
        self.sq_cldsc_std: float
        self._add_metric(
            "sq_cldsc_std",
            sq_cldsc_std,
            long_name="Segmentation Quality Centerline Dsc Standard Deviation",
        )
        self.pq_cldsc: float
        self._add_metric(
            "pq_cldsc",
            pq_cldsc,
            long_name="Panoptic Quality Centerline Dsc",
        )
        # endregion
        #
        # region ASSD
        self.sq_assd: float
        self._add_metric(
            "sq_assd",
            sq_assd,
            long_name="Segmentation Quality Assd",
        )
        self.sq_assd_std: float
        self._add_metric(
            "sq_assd_std",
            sq_assd_std,
            long_name="Segmentation Quality Assd Standard Deviation",
        )
        # endregion

        ##################
        # List Metrics   #
        ##################
        self._list_metrics: dict[Metric, Evaluation_List_Metric] = {}
        for k, v in list_metrics.items():
            is_edge_case, edge_case_result = self._edge_case_handler.handle_zero_tp(
                metric=k,
                tp=self.tp,
                num_pred_instances=self.num_pred_instances,
                num_ref_instances=self.num_ref_instances,
            )
            self._list_metrics[k] = Evaluation_List_Metric(
                k, empty_list_std, v, is_edge_case, edge_case_result
            )

    def _add_metric(
        self,
        name_id: str,
        calc_func: Callable | None,
        long_name: str | None = None,
        default_value=None,
        was_calculated: bool = False,
    ):
        setattr(self, name_id, default_value)
        # assert hasattr(self, name_id), f"added metric {name_id} but it is not a member variable of this class"
        if calc_func is None:
            assert (
                was_calculated
            ), "Tried to add a metric without a calc_function but that hasn't been calculated yet, how did you think this could works?"
        eval_metric = Evaluation_Metric(name_id, calc_func, long_name, was_calculated)
        self._evaluation_metrics[name_id] = eval_metric
        return default_value

    def calculate_all(self, print_errors: bool = False):
        """Calculates all possible metrics that can be derived

        Args:
            print_errors (bool, optional): If true, will print every metric that could not be computed and its reason. Defaults to False.
        """
        metric_errors: dict[str, Exception] = {}
        for k, v in self._evaluation_metrics.items():
            try:
                v = getattr(self, k)
            except Exception as e:
                metric_errors[k] = e

        if print_errors:
            for k, v in metric_errors.items():
                print(f"Metric {k}: {v}")

    def __str__(self) -> str:
        text = ""
        for k, v in self._evaluation_metrics.items():
            if k.endswith("_std"):
                continue
            if v.was_calculated and not v.error:
                # is there standard deviation for this?
                text += f"{v}: {self.__getattribute__(k)}"
                k_std = k + "_std"
                if (
                    k_std in self._evaluation_metrics
                    and self._evaluation_metrics[k_std].was_calculated
                    and not self._evaluation_metrics[k_std].error
                ):
                    text += f" +- {self.__getattribute__(k_std)}"
                text += "\n"
        return text

    def to_dict(self) -> dict:
        return self._evaluation_metrics

    def get_list_metric(self, metric: Metric, mode: MetricMode):
        if metric in self._list_metrics:
            return self._list_metrics[metric][mode]
        else:
            raise MetricCouldNotBeComputedException(
                f"{metric} could not be found, have you set it in eval_metrics during evaluation?"
            )

    def _calc_metric(self, metric_name: str, supress_error: bool = False):
        if metric_name in self._evaluation_metrics:
            try:
                value = self._evaluation_metrics[metric_name](self)
            except MetricCouldNotBeComputedException as e:
                value = e
            if isinstance(value, MetricCouldNotBeComputedException):
                self._evaluation_metrics[metric_name].error = True
                self._evaluation_metrics[metric_name].was_calculated = True
                if not supress_error:
                    raise value
            self._evaluation_metrics[metric_name].was_calculated = True
            return value
        else:
            raise MetricCouldNotBeComputedException(
                f"could not find metric with name {metric_name}"
            )

    def __getattribute__(self, __name: str) -> Any:
        attr = None
        try:
            attr = object.__getattribute__(self, __name)
        except AttributeError as e:
            if __name in self._evaluation_metrics.keys():
                pass
            else:
                raise e
        if attr is None:
            if self._evaluation_metrics[__name].error:
                raise MetricCouldNotBeComputedException(
                    f"Requested metric {__name} that could not be computed"
                )
            elif not self._evaluation_metrics[__name].was_calculated:
                value = self._calc_metric(__name)
                setattr(self, __name, value)
                if isinstance(value, MetricCouldNotBeComputedException):
                    raise value
                return value
        else:
            return attr


#########################
# Calculation functions #
#########################


# region Basic
def fp(res: PanopticaResult):
    return res.num_pred_instances - res.tp


def fn(res: PanopticaResult):
    return res.num_ref_instances - res.tp


def rq(res: PanopticaResult):
    """
    Calculate the Recognition Quality (RQ) based on TP, FP, and FN.

    Returns:
        float: Recognition Quality (RQ).
    """
    if res.tp == 0:
        return 0.0 if res.num_pred_instances + res.num_ref_instances > 0 else np.nan
    return res.tp / (res.tp + 0.5 * res.fp + 0.5 * res.fn)


# endregion


# region IOU
def sq(res: PanopticaResult):
    return res.get_list_metric(Metric.IOU, mode=MetricMode.AVG)


def sq_std(res: PanopticaResult):
    return res.get_list_metric(Metric.IOU, mode=MetricMode.STD)


def pq(res: PanopticaResult):
    return res.sq * res.rq


# endregion


# region DSC
def sq_dsc(res: PanopticaResult):
    return res.get_list_metric(Metric.DSC, mode=MetricMode.AVG)


def sq_dsc_std(res: PanopticaResult):
    return res.get_list_metric(Metric.DSC, mode=MetricMode.STD)


def pq_dsc(res: PanopticaResult):
    return res.sq_dsc * res.rq


# endregion


# region clDSC
def sq_cldsc(res: PanopticaResult):
    return res.get_list_metric(Metric.clDSC, mode=MetricMode.AVG)


def sq_cldsc_std(res: PanopticaResult):
    return res.get_list_metric(Metric.clDSC, mode=MetricMode.STD)


def pq_cldsc(res: PanopticaResult):
    return res.sq_cldsc * res.rq


# endregion


# region ASSD
def sq_assd(res: PanopticaResult):
    return res.get_list_metric(Metric.ASSD, mode=MetricMode.AVG)


def sq_assd_std(res: PanopticaResult):
    return res.get_list_metric(Metric.ASSD, mode=MetricMode.STD)


# endregion


# region Global
def global_bin_dsc(res: PanopticaResult):
    if res.tp == 0:
        return 0.0
    pred_binary = res._prediction_arr.copy()
    ref_binary = res._reference_arr.copy()
    pred_binary[pred_binary != 0] = 1
    ref_binary[ref_binary != 0] = 1
    return _compute_dice_coefficient(ref_binary, pred_binary)


def global_bin_cldsc(res: PanopticaResult):
    if res.tp == 0:
        return 0.0
    pred_binary = res._prediction_arr.copy()
    ref_binary = res._reference_arr.copy()
    pred_binary[pred_binary != 0] = 1
    ref_binary[ref_binary != 0] = 1
    return _compute_centerline_dice_coefficient(ref_binary, pred_binary)


# endregion


if __name__ == "__main__":
    c = PanopticaResult(
        reference_arr=np.zeros([5, 5, 5]),
        prediction_arr=np.zeros([5, 5, 5]),
        num_ref_instances=2,
        num_pred_instances=5,
        tp=0,
        list_metrics={Metric.IOU: []},
        edge_case_handler=EdgeCaseHandler(),
    )

    print(c)

    c.calculate_all(print_errors=True)
    print(c)

    # print(c.sq)
