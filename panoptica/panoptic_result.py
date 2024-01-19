from __future__ import annotations

from typing import Any, List, Callable

import numpy as np

from panoptica.metrics import _MatchingMetric, ListMetricMode, ListMetric
from panoptica.utils import EdgeCaseHandler


# TODO instead of result member variables, make an ENUM and on init it constructs based on the ENUM the Evaluation_Metric objects?
# should have autocompletion, and easier to handle?


class MetricCouldNotBeComputedException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

    pass


# TODO save value in here, make a getter function, if value is error, raise it instead of returning?
class Evaluation_Metric:
    def __init__(
        self,
        name_id: str,
        calc_func: Callable,
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

    def __call__(self, result_obj: Panoptic_Result) -> Any:
        if self.error:
            raise MetricCouldNotBeComputedException(f"Metric {self.id} requested, but could not be computed")
        assert not self.was_calculated, f"Metric {self.id} was called to compute, but is set to have been already calculated"
        return self.calc_func(result_obj)

    def __str__(self) -> str:
        if self.long_name is not None:
            return self.long_name + f" ({self.id})"
        else:
            return self.id


class Evaluation_List_Metric:
    def __init__(
        self,
        name_id: ListMetric,
        empty_list_std: float | None,
        value_list: list[float] | None,  # None stands for not calculated
        is_edge_case: bool = False,
        edge_case_result: float | None = None,
    ):
        """This represents the metrics resulting from a ListMetric (IoU, ASSD, Dice)

        Args:
            name_id (ListMetric): code-name of this metric
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
        self.STD = None if self.ALL is None else empty_list_std if len(self.ALL) == 0 else np.std(self.ALL)

    def __getitem__(self, mode: ListMetricMode | str):
        if self.error:
            raise MetricCouldNotBeComputedException(f"Metric {self.id} has not been calculated, add it to your eval_metrics")
        if isinstance(mode, ListMetricMode):
            mode = mode.name
        if hasattr(self, mode):
            return getattr(self, mode)
        else:
            raise MetricCouldNotBeComputedException(f"List_Metric {self.id} does not contain {mode} member")


class PanopticaResult(object):
    def __init__(
        self,
        num_ref_instances: int,
        num_pred_instances: int,
        tp: int,
        list_metrics: dict[ListMetric, list[float]],
        edge_case_handler: EdgeCaseHandler,
    ):
        self.edge_case_handler = edge_case_handler
        empty_list_std = self.edge_case_handler.handle_empty_list_std()
        ######################
        # Evaluation Metrics #
        ######################
        self.evaluation_metrics: dict[str, Evaluation_Metric] = {}
        #
        # Already Calculated
        self.num_ref_instances: int
        self.add_metric(
            "num_ref_instances",
            None,
            long_name="Number of instances in reference",
            default_value=num_ref_instances,
            was_calculated=True,
        )
        self.num_pred_instances: int
        self.add_metric(
            "num_pred_instances",
            None,
            long_name="Number of instances in prediction",
            default_value=num_pred_instances,
            was_calculated=True,
        )
        self.tp: int
        self.add_metric(
            "tp",
            None,
            long_name="True Positives",
            default_value=tp,
            was_calculated=True,
        )
        # Basic
        self.fp: int
        self.add_metric(
            "fp",
            fp,
            long_name="False Positives",
        )
        self.fn: int
        self.add_metric(
            "fn",
            fn,
            long_name="False Negatives",
        )
        self.rq: float
        self.add_metric(
            "rq",
            rq,
            long_name="Recognition Quality",
        )
        # IOU
        self.sq: float
        self.add_metric(
            "sq",
            sq,
            long_name="Segmentation Quality IoU",
        )
        self.sq_std: float
        self.add_metric(
            "sq_std",
            sq_std,
            long_name="Segmentation Quality IoU Standard Deviation",
        )
        self.pq: float
        self.add_metric(
            "pq",
            pq,
            long_name="Panoptic Quality IoU",
        )
        # DICE
        self.sq_dsc: float
        self.add_metric(
            "sq_dsc",
            sq_dsc,
            long_name="Segmentation Quality Dsc",
        )
        self.sq_dsc_std: float
        self.add_metric(
            "sq_dsc_std",
            sq_dsc_std,
            long_name="Segmentation Quality Dsc Standard Deviation",
        )
        self.pq: float
        self.add_metric(
            "pq_dsc",
            pq_dsc,
            long_name="Panoptic Quality Dsc",
        )
        # ASSD
        self.sq_assd: float
        self.add_metric(
            "sq_assd",
            sq_assd,
            long_name="Segmentation Quality Assd",
        )
        self.sq_assd_std: float
        self.add_metric(
            "sq_assd_std",
            sq_assd_std,
            long_name="Segmentation Quality Assd Standard Deviation",
        )

        ##################
        # List Metrics   #
        ##################
        self.list_metrics: dict[ListMetric, Evaluation_List_Metric] = {}
        for k, v in list_metrics.items():
            is_edge_case, edge_case_result = self.edge_case_handler.handle_zero_tp(
                metric=k,
                tp=self.tp,
                num_pred_instances=self.num_pred_instances,
                num_ref_instances=self.num_ref_instances,
            )
            self.list_metrics[k] = Evaluation_List_Metric(k, empty_list_std, v, is_edge_case, edge_case_result)

    def add_metric(
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
        self.evaluation_metrics[name_id] = eval_metric
        return default_value

    def calculate_all(self, print_errors: bool = False):
        metric_errors: dict[str, Exception] = {}
        for k, v in self.evaluation_metrics.items():
            try:
                v = getattr(self, k)
            except Exception as e:
                metric_errors[k] = e

        if print_errors:
            for k, v in metric_errors.items():
                print(f"Metric {k}: {v}")

    def __str__(self) -> str:
        text = ""
        for k, v in self.evaluation_metrics.items():
            if k.endswith("_std"):
                continue
            if v.was_calculated and not v.error:
                # is there standard deviation for this?
                text += f"{v}: {self.__getattribute__(k)}"
                k_std = k + "_std"
                if (
                    k_std in self.evaluation_metrics
                    and self.evaluation_metrics[k_std].was_calculated
                    and not self.evaluation_metrics[k_std].error
                ):
                    text += f" +- {self.__getattribute__(k_std)}"
                text += "\n"
        return text

    def to_dict(self) -> dict:
        return self.evaluation_metrics

    def get_list_metric(self, metric: ListMetric, mode: ListMetricMode):
        if metric in self.list_metrics:
            return self.list_metrics[metric][mode]
        else:
            raise MetricCouldNotBeComputedException(f"{metric} could not be found, have you set it in eval_metrics during evaluation?")

    def _calc_metric(self, metric_name: str, supress_error: bool = False):
        if metric_name in self.evaluation_metrics:
            try:
                value = self.evaluation_metrics[metric_name](self)
            except MetricCouldNotBeComputedException as e:
                value = e
            if isinstance(value, MetricCouldNotBeComputedException):
                self.evaluation_metrics[metric_name].error = True
                self.evaluation_metrics[metric_name].was_calculated = True
                if not supress_error:
                    raise value
            self.evaluation_metrics[metric_name].was_calculated = True
            return value
        else:
            raise MetricCouldNotBeComputedException(f"could not find metric with name {metric_name}")

    def __getattribute__(self, __name: str) -> Any:
        attr = None
        try:
            attr = object.__getattribute__(self, __name)
        except AttributeError as e:
            if __name in self.evaluation_metrics.keys():
                pass
            else:
                raise e
        if attr is None:
            if self.evaluation_metrics[__name].error:
                raise MetricCouldNotBeComputedException(f"Requested metric {__name} that could not be computed")
            elif not self.evaluation_metrics[__name].was_calculated:
                value = self._calc_metric(__name)
                setattr(self, __name, value)
                if isinstance(value, MetricCouldNotBeComputedException):
                    raise value
                return value
        else:
            return attr


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


# IOU
def sq(res: PanopticaResult):
    return res.get_list_metric(ListMetric.IOU, mode=ListMetricMode.AVG)


def sq_std(res: PanopticaResult):
    return res.get_list_metric(ListMetric.IOU, mode=ListMetricMode.STD)


def pq(res: PanopticaResult):
    return res.sq * res.rq


# DSC
def sq_dsc(res: PanopticaResult):
    return res.get_list_metric(ListMetric.DSC, mode=ListMetricMode.AVG)


def sq_dsc_std(res: PanopticaResult):
    return res.get_list_metric(ListMetric.DSC, mode=ListMetricMode.STD)


def pq_dsc(res: PanopticaResult):
    return res.sq_dsc * res.rq


# ASSD
def sq_assd(res: PanopticaResult):
    return res.get_list_metric(ListMetric.ASSD, mode=ListMetricMode.AVG)


def sq_assd_std(res: PanopticaResult):
    return res.get_list_metric(ListMetric.ASSD, mode=ListMetricMode.STD)


if __name__ == "__main__":
    c = PanopticaResult(
        num_ref_instances=2,
        num_pred_instances=5,
        tp=2,
        list_metrics={ListMetric.IOU: [1.0, 0.8]},
        edge_case_handler=EdgeCaseHandler(),
    )

    print(c)

    c.calculate_all(print_errors=True)
    print(c)

    # print(c.sq)
