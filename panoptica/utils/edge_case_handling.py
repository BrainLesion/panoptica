from typing import Any

import numpy as np

from panoptica.metrics import ListMetric, Metrics
from panoptica.utils.constants import _Enum_Compare, auto


class EdgeCaseResult(_Enum_Compare):
    INF = np.inf
    NAN = np.nan
    ZERO = 0.0
    ONE = 1.0
    NONE = None


class EdgeCaseZeroTP(_Enum_Compare):
    NO_INSTANCES = auto()
    EMPTY_PRED = auto()
    EMPTY_REF = auto()
    NORMAL = auto()

    def __hash__(self) -> int:
        return self.value


class MetricZeroTPEdgeCaseHandling(object):
    def __init__(
        self,
        default_result: EdgeCaseResult,
        no_instances_result: EdgeCaseResult | None = None,
        empty_prediction_result: EdgeCaseResult | None = None,
        empty_reference_result: EdgeCaseResult | None = None,
        normal: EdgeCaseResult | None = None,
    ) -> None:
        self.edgecase_dict: dict[EdgeCaseZeroTP, EdgeCaseResult] = {}
        self.edgecase_dict[EdgeCaseZeroTP.EMPTY_PRED] = (
            empty_prediction_result
            if empty_prediction_result is not None
            else default_result
        )
        self.edgecase_dict[EdgeCaseZeroTP.EMPTY_REF] = (
            empty_reference_result
            if empty_reference_result is not None
            else default_result
        )
        self.edgecase_dict[EdgeCaseZeroTP.NO_INSTANCES] = (
            no_instances_result if no_instances_result is not None else default_result
        )
        self.edgecase_dict[EdgeCaseZeroTP.NORMAL] = (
            normal if normal is not None else default_result
        )

    def __call__(
        self, tp: int, num_pred_instances, num_ref_instances
    ) -> tuple[bool, float | None]:
        if tp != 0:
            return False, EdgeCaseResult.NONE.value
        #
        elif num_pred_instances + num_ref_instances == 0:
            return True, self.edgecase_dict[EdgeCaseZeroTP.NO_INSTANCES].value
        elif num_ref_instances == 0:
            return True, self.edgecase_dict[EdgeCaseZeroTP.EMPTY_REF].value
        elif num_pred_instances == 0:
            return True, self.edgecase_dict[EdgeCaseZeroTP.EMPTY_PRED].value
        elif num_pred_instances > 0 and num_ref_instances > 0:
            return True, self.edgecase_dict[EdgeCaseZeroTP.NORMAL].value

        raise NotImplementedError(
            f"MetricZeroTPEdgeCaseHandling: couldn't handle case, got tp {tp}, n_pred_instances {num_pred_instances}, n_ref_instances {num_ref_instances}"
        )

    def __str__(self) -> str:
        txt = ""
        for k, v in self.edgecase_dict.items():
            if v is not None:
                txt += str(k) + ": " + str(v) + "\n"
        return txt


class EdgeCaseHandler:
    def __init__(
        self,
        listmetric_zeroTP_handling: dict[ListMetric, MetricZeroTPEdgeCaseHandling] = {
            ListMetric.DSC: MetricZeroTPEdgeCaseHandling(
                no_instances_result=EdgeCaseResult.NAN,
                default_result=EdgeCaseResult.ZERO,
            ),
            ListMetric.IOU: MetricZeroTPEdgeCaseHandling(
                no_instances_result=EdgeCaseResult.NAN,
                empty_prediction_result=EdgeCaseResult.ZERO,
                default_result=EdgeCaseResult.ZERO,
            ),
            ListMetric.ASSD: MetricZeroTPEdgeCaseHandling(
                no_instances_result=EdgeCaseResult.NAN,
                default_result=EdgeCaseResult.INF,
            ),
        },
        empty_list_std: EdgeCaseResult = EdgeCaseResult.NAN,
    ) -> None:
        self.__listmetric_zeroTP_handling: dict[
            ListMetric, MetricZeroTPEdgeCaseHandling
        ] = listmetric_zeroTP_handling
        self.__empty_list_std = empty_list_std

    def handle_zero_tp(
        self,
        metric: ListMetric,
        tp: int,
        num_pred_instances: int,
        num_ref_instances: int,
    ) -> tuple[bool, float | None]:
        if metric not in self.__listmetric_zeroTP_handling:
            raise NotImplementedError(
                f"Metric {metric} encountered zero TP, but no edge handling available"
            )

        return self.__listmetric_zeroTP_handling[metric](
            tp=tp,
            num_pred_instances=num_pred_instances,
            num_ref_instances=num_ref_instances,
        )

    def get_metric_zero_tp_handle(self, metric: ListMetric):
        return self.__listmetric_zeroTP_handling[metric]

    def handle_empty_list_std(self):
        return self.__empty_list_std.value

    def __str__(self) -> str:
        txt = f"EdgeCaseHandler:\n - Standard Deviation of Empty = {self.__empty_list_std}"
        for k, v in self.__listmetric_zeroTP_handling.items():
            txt += f"\n- {k}: {str(v)}"
        return str(txt)


if __name__ == "__main__":
    handler = EdgeCaseHandler()

    print()
    # print(handler.get_metric_zero_tp_handle(ListMetric.IOU))
    r = handler.handle_zero_tp(
        ListMetric.IOU, tp=0, num_pred_instances=1, num_ref_instances=1
    )
    print(r)

    iou_test = MetricZeroTPEdgeCaseHandling(
        no_instances_result=EdgeCaseResult.NAN,
        default_result=EdgeCaseResult.ZERO,
    )
    # print(iou_test)
    t = iou_test(tp=0, num_pred_instances=1, num_ref_instances=1)
    print(t)

    # iou_test = default_iou
    # print(iou_test)
    # t = iou_test(tp=0, num_pred_instances=1, num_ref_instances=1)
    # print(t)
