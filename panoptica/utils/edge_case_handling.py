import numpy as np
from panoptica.metrics import Metric
from panoptica.utils.constants import _Enum_Compare, auto
from panoptica.utils.config import SupportsConfig


class EdgeCaseResult(_Enum_Compare):
    """Enumeration of edge case values used for handling specific metric situations.

    This enum defines several common edge case values for handling zero-true-positive (zero-TP)
    situations in various metrics. The values include infinity, NaN, zero, one, and None.

    Attributes:
        INF: Represents infinity (`np.inf`).
        NAN: Represents not-a-number (`np.nan`).
        ZERO: Represents zero (0.0).
        ONE: Represents one (1.0).
        NONE: Represents a None value.

    Methods:
        value: Returns the value associated with the edge case.
        __call__(): Returns the numeric or None representation of the enum member.
    """

    INF = auto()  # np.inf
    NAN = auto()  # np.nan
    ZERO = auto()  # 0.0
    ONE = auto()  # 1.0
    NONE = auto()  # None

    @property
    def value(self):
        return self()

    def __call__(self):
        transfer_dict = {
            EdgeCaseResult.INF.name: np.inf,
            EdgeCaseResult.NAN.name: np.nan,
            EdgeCaseResult.ZERO.name: 0.0,
            EdgeCaseResult.ONE.name: 1.0,
            EdgeCaseResult.NONE.name: None,
        }
        if self.name in transfer_dict:
            return transfer_dict[self.name]
        raise KeyError(f"No defined value for EdgeCaseResult {str(self)}")


class EdgeCaseZeroTP(_Enum_Compare):
    """Enum defining scenarios that could produce zero true positives (zero-TP) in metrics.

    Attributes:
        NO_INSTANCES: No instances in both the prediction and reference.
        EMPTY_PRED: The prediction is empty.
        EMPTY_REF: The reference is empty.
        NORMAL: A typical scenario with non-zero instances.
    """

    NO_INSTANCES = auto()
    EMPTY_PRED = auto()
    EMPTY_REF = auto()
    NORMAL = auto()

    def __hash__(self) -> int:
        return self.value


class MetricZeroTPEdgeCaseHandling(SupportsConfig):
    """Handles zero-TP edge cases for metrics, mapping different zero-TP scenarios to specific results.

    Attributes:
        default_result (EdgeCaseResult | None): Default result if specific edge cases are not provided.
        no_instances_result (EdgeCaseResult | None): Result when no instances are present.
        empty_prediction_result (EdgeCaseResult | None): Result when prediction is empty.
        empty_reference_result (EdgeCaseResult | None): Result when reference is empty.
        normal (EdgeCaseResult | None): Result when a normal zero-TP scenario occurs.

    Methods:
        __call__(tp, num_pred_instances, num_ref_instances): Determines if an edge case is detected and returns its result.
        __eq__(value): Compares this handling object to another.
        __str__(): String representation of edge cases.
        _yaml_repr(cls, node): YAML representation for the edge case.
    """

    def __init__(
        self,
        default_result: EdgeCaseResult | None = None,
        no_instances_result: EdgeCaseResult | None = None,
        empty_prediction_result: EdgeCaseResult | None = None,
        empty_reference_result: EdgeCaseResult | None = None,
        normal: EdgeCaseResult | None = None,
    ) -> None:
        assert default_result is not None or (
            no_instances_result is not None
            and empty_prediction_result is not None
            and empty_reference_result is not None
            and normal is not None
        ), "default_result is None and the rest is not fully specified"

        self._default_result = default_result
        self._edgecase_dict: dict[EdgeCaseZeroTP, EdgeCaseResult] = {}
        self._edgecase_dict[EdgeCaseZeroTP.EMPTY_PRED] = (
            empty_prediction_result
            if empty_prediction_result is not None
            else default_result
        )
        self._edgecase_dict[EdgeCaseZeroTP.EMPTY_REF] = (
            empty_reference_result
            if empty_reference_result is not None
            else default_result
        )
        self._edgecase_dict[EdgeCaseZeroTP.NO_INSTANCES] = (
            no_instances_result if no_instances_result is not None else default_result
        )
        self._edgecase_dict[EdgeCaseZeroTP.NORMAL] = (
            normal if normal is not None else default_result
        )

    def __call__(
        self, tp: int, num_pred_instances, num_ref_instances
    ) -> tuple[bool, float | None]:
        if tp != 0:
            return False, EdgeCaseResult.NONE.value
        #
        elif num_pred_instances + num_ref_instances == 0:
            return True, self._edgecase_dict[EdgeCaseZeroTP.NO_INSTANCES].value
        elif num_ref_instances == 0:
            return True, self._edgecase_dict[EdgeCaseZeroTP.EMPTY_REF].value
        elif num_pred_instances == 0:
            return True, self._edgecase_dict[EdgeCaseZeroTP.EMPTY_PRED].value
        elif num_pred_instances > 0 and num_ref_instances > 0:
            return True, self._edgecase_dict[EdgeCaseZeroTP.NORMAL].value

        raise NotImplementedError(
            f"MetricZeroTPEdgeCaseHandling: couldn't handle case, got tp {tp}, n_pred_instances {num_pred_instances}, n_ref_instances {num_ref_instances}"
        )

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, MetricZeroTPEdgeCaseHandling):
            for s, k in self._edgecase_dict.items():
                if s not in __value._edgecase_dict or k != __value._edgecase_dict[s]:
                    return False
            return True
        return False

    def __str__(self) -> str:
        txt = ""
        for k, v in self._edgecase_dict.items():
            if v is not None:
                txt += str(k) + ": " + str(v) + "\n"
        return txt

    @classmethod
    def _yaml_repr(cls, node) -> dict:
        return {
            "no_instances_result": node._edgecase_dict[EdgeCaseZeroTP.NO_INSTANCES],
            "empty_prediction_result": node._edgecase_dict[EdgeCaseZeroTP.EMPTY_PRED],
            "empty_reference_result": node._edgecase_dict[EdgeCaseZeroTP.EMPTY_REF],
            "normal": node._edgecase_dict[EdgeCaseZeroTP.NORMAL],
        }


class EdgeCaseHandler(SupportsConfig):
    """Manages edge cases across multiple metrics, including standard deviation handling for empty lists.

    Attributes:
        listmetric_zeroTP_handling (dict): Dictionary mapping metrics to their zero-TP edge case handling.
        empty_list_std (EdgeCaseResult): Default edge case for handling standard deviation of empty lists.

    Methods:
        handle_zero_tp(metric, tp, num_pred_instances, num_ref_instances): Checks if an edge case exists and returns its result.
        listmetric_zeroTP_handling: Returns the edge case handling dictionary.
        get_metric_zero_tp_handle(metric): Returns the zero-TP handler for a specific metric.
        handle_empty_list_std(): Handles standard deviation of empty lists.
        _yaml_repr(cls, node): YAML representation of the handler.
    """

    def __init__(
        self,
        listmetric_zeroTP_handling: dict[Metric, MetricZeroTPEdgeCaseHandling] = {
            Metric.DSC: MetricZeroTPEdgeCaseHandling(
                no_instances_result=EdgeCaseResult.NAN,
                default_result=EdgeCaseResult.ZERO,
            ),
            Metric.clDSC: MetricZeroTPEdgeCaseHandling(
                no_instances_result=EdgeCaseResult.NAN,
                default_result=EdgeCaseResult.ZERO,
            ),
            Metric.IOU: MetricZeroTPEdgeCaseHandling(
                no_instances_result=EdgeCaseResult.NAN,
                empty_prediction_result=EdgeCaseResult.ZERO,
                default_result=EdgeCaseResult.ZERO,
            ),
            Metric.ASSD: MetricZeroTPEdgeCaseHandling(
                no_instances_result=EdgeCaseResult.NAN,
                default_result=EdgeCaseResult.INF,
            ),
            Metric.RVD: MetricZeroTPEdgeCaseHandling(
                default_result=EdgeCaseResult.NAN,
            ),
            Metric.RVAE: MetricZeroTPEdgeCaseHandling(
                default_result=EdgeCaseResult.NAN,
            ),
        },
        empty_list_std: EdgeCaseResult = EdgeCaseResult.NAN,
    ) -> None:
        self.__listmetric_zeroTP_handling: dict[
            Metric, MetricZeroTPEdgeCaseHandling
        ] = listmetric_zeroTP_handling
        self.__empty_list_std: EdgeCaseResult = empty_list_std

    def handle_zero_tp(
        self,
        metric: Metric,
        tp: int,
        num_pred_instances: int,
        num_ref_instances: int,
    ) -> tuple[bool, float | None]:
        """_summary_

        Args:
            metric (Metric): _description_
            tp (int): _description_
            num_pred_instances (int): _description_
            num_ref_instances (int): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            tuple[bool, float | None]: if edge case, and its edge case value
        """
        if tp != 0:
            return False, EdgeCaseResult.NONE.value
        if metric not in self.__listmetric_zeroTP_handling:
            raise NotImplementedError(
                f"Metric {metric} encountered zero TP, but no edge handling available"
            )

        return self.__listmetric_zeroTP_handling[metric](
            tp=tp,
            num_pred_instances=num_pred_instances,
            num_ref_instances=num_ref_instances,
        )

    @property
    def listmetric_zeroTP_handling(self):
        return self.__listmetric_zeroTP_handling

    def get_metric_zero_tp_handle(self, metric: Metric):
        return self.__listmetric_zeroTP_handling[metric]

    def handle_empty_list_std(self) -> EdgeCaseResult | None:
        return self.__empty_list_std

    def __str__(self) -> str:
        txt = f"EdgeCaseHandler:\n - Standard Deviation of Empty = {self.__empty_list_std}"
        for k, v in self.__listmetric_zeroTP_handling.items():
            txt += f"\n- {k}: {str(v)}"
        return str(txt)

    @classmethod
    def _yaml_repr(cls, node) -> dict:
        return {
            "listmetric_zeroTP_handling": node.__listmetric_zeroTP_handling,
            "empty_list_std": node.__empty_list_std,
        }
