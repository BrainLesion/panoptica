from __future__ import annotations

from typing import Any, Callable

import numpy as np

from panoptica.metrics import (
    Evaluation_List_Metric,
    Evaluation_Metric,
    Metric,
    MetricCouldNotBeComputedException,
    MetricMode,
    MetricType,
)
from panoptica.utils import EdgeCaseHandler


class PanopticaResult(object):

    def __init__(
        self,
        reference_arr: np.ndarray,
        prediction_arr: np.ndarray,
        num_pred_instances: int,
        num_ref_instances: int,
        tp: int,
        list_metrics: dict[Metric, list[float]],
        edge_case_handler: EdgeCaseHandler,
        global_metrics: list[Metric] = [],
        computation_time: float | None = None,
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
        self._evaluation_metrics: dict[str, Evaluation_Metric] = {}
        self._edge_case_handler = edge_case_handler
        empty_list_std = self._edge_case_handler.handle_empty_list_std().value
        self._global_metrics: list[Metric] = global_metrics
        self.computation_time = computation_time
        ######################
        # Evaluation Metrics #
        ######################
        #
        # region Already Calculated
        self.num_ref_instances: int
        self._add_metric(
            "num_ref_instances",
            MetricType.MATCHING,
            None,
            long_name="Number of instances in reference",
            default_value=num_ref_instances,
            was_calculated=True,
        )
        self.num_pred_instances: int
        self._add_metric(
            "num_pred_instances",
            MetricType.MATCHING,
            None,
            long_name="Number of instances in prediction",
            default_value=num_pred_instances,
            was_calculated=True,
        )
        self.tp: int
        self._add_metric(
            "tp",
            MetricType.MATCHING,
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
            MetricType.MATCHING,
            fp,
            long_name="False Positives",
        )
        self.fn: int
        self._add_metric(
            "fn",
            MetricType.MATCHING,
            fn,
            long_name="False Negatives",
        )
        self.prec: int
        self._add_metric(
            "prec",
            MetricType.NO_PRINT,
            prec,
            long_name="Precision (positive predictive value)",
        )
        self.rec: int
        self._add_metric(
            "rec",
            MetricType.NO_PRINT,
            rec,
            long_name="Recall (sensitivity)",
        )
        self.rq: float
        self._add_metric(
            "rq",
            MetricType.MATCHING,
            rq,
            long_name="Recognition Quality / F1-Score",
        )
        # endregion
        #
        #
        # region IOU
        self.sq: float
        self._add_metric(
            "sq",
            MetricType.INSTANCE,
            sq,
            long_name="Segmentation Quality IoU",
        )
        self.sq_std: float
        self._add_metric(
            "sq_std",
            MetricType.INSTANCE,
            sq_std,
            long_name="Segmentation Quality IoU Standard Deviation",
        )
        self.pq: float
        self._add_metric(
            "pq",
            MetricType.INSTANCE,
            pq,
            long_name="Panoptic Quality IoU",
        )
        # endregion
        #
        # region DICE
        self.sq_dsc: float
        self._add_metric(
            "sq_dsc",
            MetricType.INSTANCE,
            sq_dsc,
            long_name="Segmentation Quality Dsc",
        )
        self.sq_dsc_std: float
        self._add_metric(
            "sq_dsc_std",
            MetricType.INSTANCE,
            sq_dsc_std,
            long_name="Segmentation Quality Dsc Standard Deviation",
        )
        self.pq_dsc: float
        self._add_metric(
            "pq_dsc",
            MetricType.INSTANCE,
            pq_dsc,
            long_name="Panoptic Quality Dsc",
        )
        # endregion
        #
        # region clDICE
        self.sq_cldsc: float
        self._add_metric(
            "sq_cldsc",
            MetricType.INSTANCE,
            sq_cldsc,
            long_name="Segmentation Quality Centerline Dsc",
        )
        self.sq_cldsc_std: float
        self._add_metric(
            "sq_cldsc_std",
            MetricType.INSTANCE,
            sq_cldsc_std,
            long_name="Segmentation Quality Centerline Dsc Standard Deviation",
        )
        self.pq_cldsc: float
        self._add_metric(
            "pq_cldsc",
            MetricType.INSTANCE,
            pq_cldsc,
            long_name="Panoptic Quality Centerline Dsc",
        )
        # endregion
        #
        # region ASSD
        self.sq_assd: float
        self._add_metric(
            "sq_assd",
            MetricType.INSTANCE,
            sq_assd,
            long_name="Segmentation Quality ASSD",
        )
        self.sq_assd_std: float
        self._add_metric(
            "sq_assd_std",
            MetricType.INSTANCE,
            sq_assd_std,
            long_name="Segmentation Quality ASSD Standard Deviation",
        )
        # endregion
        #
        # region RVD
        self.sq_rvd: float
        self._add_metric(
            "sq_rvd",
            MetricType.INSTANCE,
            sq_rvd,
            long_name="Segmentation Quality Relative Volume Difference",
        )
        self.sq_rvd_std: float
        self._add_metric(
            "sq_rvd_std",
            MetricType.INSTANCE,
            sq_rvd_std,
            long_name="Segmentation Quality Relative Volume Difference Standard Deviation",
        )
        # endregion

        # region Global
        # Just for autocomplete
        self.global_bin_dsc: int
        self.global_bin_iou: int
        self.global_bin_cldsc: int
        self.global_bin_assd: int
        self.global_bin_rvd: int
        # endregion

        ##################
        # List Metrics   #
        ##################
        self._list_metrics: dict[Metric, Evaluation_List_Metric] = {}
        # Loop over all available metric, add it to evaluation_list_metric if available, but also add the global references

        arrays_present = False
        if prediction_arr is not None and reference_arr is not None:
            pred_binary = prediction_arr.copy()
            ref_binary = reference_arr.copy()
            pred_binary[pred_binary != 0] = 1
            ref_binary[ref_binary != 0] = 1
            arrays_present = True

        for m in Metric:
            if m in list_metrics:
                is_edge_case, edge_case_result = self._edge_case_handler.handle_zero_tp(
                    metric=m,
                    tp=self.tp,
                    num_pred_instances=self.num_pred_instances,
                    num_ref_instances=self.num_ref_instances,
                )
                self._list_metrics[m] = Evaluation_List_Metric(
                    m, empty_list_std, list_metrics[m], is_edge_case, edge_case_result
                )
            # even if not available, set the global vars
            default_value = None
            was_calculated = False

            if m in self._global_metrics and arrays_present:
                default_value = self._calc_global_bin_metric(
                    m, pred_binary, ref_binary, do_binarize=False
                )
                was_calculated = True

            self._add_metric(
                f"global_bin_{m.name.lower()}",
                MetricType.GLOBAL,
                lambda x: MetricCouldNotBeComputedException(
                    f"Global Metric {m} not set"
                ),
                long_name="Global Binary " + m.value.long_name,
                default_value=default_value,
                was_calculated=was_calculated,
            )

    def _calc_global_bin_metric(
        self,
        metric: Metric,
        prediction_arr,
        reference_arr,
        do_binarize: bool = True,
    ):
        if metric not in self._global_metrics:
            raise MetricCouldNotBeComputedException(f"Global Metric {metric} not set")

        if do_binarize:
            pred_binary = prediction_arr.copy()
            ref_binary = reference_arr.copy()
            pred_binary[pred_binary != 0] = 1
            ref_binary[ref_binary != 0] = 1
        else:
            pred_binary = prediction_arr
            ref_binary = reference_arr

        prediction_empty = pred_binary.sum() == 0
        reference_empty = ref_binary.sum() == 0
        if prediction_empty or reference_empty:
            is_edgecase, result = self._edge_case_handler.handle_zero_tp(
                metric, 0, int(prediction_empty), int(reference_empty)
            )
            if is_edgecase:
                return result

        return metric(
            reference_arr=ref_binary,
            prediction_arr=pred_binary,
        )

    def _add_metric(
        self,
        name_id: str,
        metric_type: MetricType,
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
        eval_metric = Evaluation_Metric(
            name_id,
            metric_type=metric_type,
            calc_func=calc_func,
            long_name=long_name,
            was_calculated=was_calculated,
        )
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

    def _calc(self, k, v):
        try:
            v = getattr(self, k)
            return False, v
        except Exception as e:
            return True, e

    def __str__(self) -> str:
        text = ""
        for metric_type in MetricType:
            if metric_type == MetricType.NO_PRINT:
                continue
            if metric_type == MetricType.GLOBAL and len(self._global_metrics) == 0:
                continue
            text += f"\n+++ {metric_type.name} +++\n"
            for k, v in self._evaluation_metrics.items():
                if v.metric_type != metric_type:
                    continue
                if k.endswith("_std"):
                    continue
                if v._was_calculated and not v._error:
                    # is there standard deviation for this?
                    text += f"{v}: {self.__getattribute__(k)}"
                    k_std = k + "_std"
                    if (
                        k_std in self._evaluation_metrics
                        and self._evaluation_metrics[k_std]._was_calculated
                        and not self._evaluation_metrics[k_std]._error
                    ):
                        text += f" +- {self.__getattribute__(k_std)}"
                    text += "\n"
        return text

    def to_dict(self) -> dict:
        return {
            k: getattr(self, v.id)
            for k, v in self._evaluation_metrics.items()
            if (v._error == False and v._was_calculated)
        }

    @property
    def evaluation_metrics(self):
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
                self._evaluation_metrics[metric_name]._error = True
                self._evaluation_metrics[metric_name]._was_calculated = True
                if not supress_error:
                    raise value
            self._evaluation_metrics[metric_name]._was_calculated = True
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
            if __name == "_evaluation_metrics":
                raise e
            if __name in self._evaluation_metrics.keys():
                pass
            else:
                raise e
        if __name == "_evaluation_metrics":
            return attr
        if (
            object.__getattribute__(self, "_evaluation_metrics") is not None
            and __name in self._evaluation_metrics.keys()
        ):
            if attr is None:
                if self._evaluation_metrics[__name]._error:
                    raise MetricCouldNotBeComputedException(
                        f"Requested metric {__name} that could not be computed"
                    )
                elif not self._evaluation_metrics[__name]._was_calculated:
                    value = self._calc_metric(__name)
                    setattr(self, __name, value)
                    if isinstance(value, MetricCouldNotBeComputedException):
                        raise value
                    return value
        return attr


#########################
# Calculation functions #
#########################


# region Basic
def fp(res: PanopticaResult):
    return res.num_pred_instances - res.tp


def fn(res: PanopticaResult):
    return res.num_ref_instances - res.tp


def prec(res: PanopticaResult):
    return res.tp / (res.tp + res.fp)


def rec(res: PanopticaResult):
    return res.tp / (res.tp + res.fn)


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


# region RVD
def sq_rvd(res: PanopticaResult):
    return res.get_list_metric(Metric.RVD, mode=MetricMode.AVG)


def sq_rvd_std(res: PanopticaResult):
    return res.get_list_metric(Metric.RVD, mode=MetricMode.STD)


# endregion


def _build_global_bin_metric_function(metric: Metric):

    def function_template(res: PanopticaResult):
        if metric not in res._global_metrics:
            raise MetricCouldNotBeComputedException(f"Global Metric {metric} not set")
        if res.tp == 0:
            is_edgecase, result = res._edge_case_handler.handle_zero_tp(
                metric, res.tp, res.num_pred_instances, res.num_ref_instances
            )
            if is_edgecase:
                return result
        pred_binary = res._prediction_arr.copy()
        ref_binary = res._reference_arr.copy()
        pred_binary[pred_binary != 0] = 1
        ref_binary[ref_binary != 0] = 1
        return metric(
            reference_arr=res._reference_arr,
            prediction_arr=res._prediction_arr,
        )

    return lambda x: function_template(x)


# endregion
