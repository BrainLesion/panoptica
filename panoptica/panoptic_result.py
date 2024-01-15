from __future__ import annotations

from typing import Any, List

import numpy as np

from panoptica.metrics import EvalMetric, ListMetric, MetricDict, _MatchingMetric
from panoptica.utils import EdgeCaseHandler


class PanopticaResult:
    """
    Represents the result of the Panoptic Quality (PQ) computation.

    Attributes:
        num_ref_instances (int): Number of reference instances.
        num_pred_instances (int): Number of predicted instances.
        tp (int): Number of correctly matched instances (True Positives).
        fp (int): Number of extra predicted instances (False Positives).
    """

    def __init__(
        self,
        num_ref_instances: int,
        num_pred_instances: int,
        tp: int,
        list_metrics: dict[_MatchingMetric | str, list[float]],
        edge_case_handler: EdgeCaseHandler,
    ):
        """
        Initialize a PanopticaResult object.

        Args:
            num_ref_instances (int): Number of reference instances.
            num_pred_instances (int): Number of predicted instances.
            tp (int): Number of correctly matched instances (True Positives).
            list_metrics: dict[MatchingMetric | str, list[float]]: TBD
            edge_case_handler: EdgeCaseHandler: TBD
        """
        self._tp = tp
        self.edge_case_handler = edge_case_handler
        self.metric_dict: MetricDict = {}
        for k, v in list_metrics.items():
            if isinstance(k, _MatchingMetric):
                k = k.name
            self.metric_dict[k] = v

        # for k in ListMetric:
        #    if k.name not in self.metric_dict:
        #        self.metric_dict[k.name] = []
        self._num_ref_instances = num_ref_instances
        self._num_pred_instances = num_pred_instances

        # TODO instead of all the properties, make a generic function inputting metric and std or not,
        # and returns it if contained in dictionary,
        # otherwise calls function to calculates, saves it and return

    def __str__(self):
        text = (
            f"Number of instances in prediction: {self.num_pred_instances}\n"
            f"Number of instances in reference: {self.num_ref_instances}\n"
            f"True Positives (tp): {self.tp}\n"
            f"False Positives (fp): {self.fp}\n"
            f"False Negatives (fn): {self.fn}\n"
            f"Recognition Quality / F1 Score (RQ): {self.rq}\n"
        )

        if ListMetric.IOU.name in self.metric_dict:
            text += f"Segmentation Quality (SQ): {self.sq} ± {self.sq_sd}\n"
            text += f"Panoptic Quality (PQ): {self.pq}\n"

        if ListMetric.DSC.name in self.metric_dict:
            text += f"DSC-based Segmentation Quality (DQ_DSC): {self.sq_dsc} ± {self.sq_dsc_sd}\n"
            text += f"DSC-based Panoptic Quality (PQ_DSC): {self.pq_dsc}\n"

        if ListMetric.ASSD.name in self.metric_dict:
            text += f"Average symmetric surface distance (ASSD): {self.sq_assd} ± {self.sq_assd_sd}\n"
            text += f"ASSD-based Panoptic Quality (PQ_ASSD): {self.pq_assd}"
        return text

    def to_dict(self):
        eval_dict = {
            "num_pred_instances": self.num_pred_instances,
            "num_ref_instances": self.num_ref_instances,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "rq": self.rq,
        }

        if ListMetric.IOU.name in self.metric_dict:
            eval_dict["sq"] = self.sq
            eval_dict["sq_sd"] = self.sq_sd
            eval_dict["pq"] = self.pq

        if ListMetric.DSC.name in self.metric_dict:
            eval_dict["sq_dsc"] = self.sq_dsc
            eval_dict["sq_dsc_sd"] = self.sq_dsc_sd
            eval_dict["pq_dsc"] = self.pq_dsc

        if ListMetric.ASSD.name in self.metric_dict:
            eval_dict["sq_assd"] = self.sq_assd
            eval_dict["sq_assd_sd"] = self.sq_assd_sd
            eval_dict["pq_assd"] = self.pq_assd
        return eval_dict

    @property
    def num_ref_instances(self) -> int:
        """
        Get the number of reference instances.

        Returns:
            int: Number of reference instances.
        """
        return self._num_ref_instances

    @property
    def num_pred_instances(self) -> int:
        """
        Get the number of predicted instances.

        Returns:
            int: Number of predicted instances.
        """
        return self._num_pred_instances

    @property
    def tp(self) -> int:
        """
        Calculate the number of True Positives (TP).

        Returns:
            int: Number of True Positives.
        """
        return self._tp

    @property
    def fp(self) -> int:
        """
        Calculate the number of False Positives (FP).

        Returns:
            int: Number of False Positives.
        """
        return self.num_pred_instances - self.tp

    @property
    def fn(self) -> int:
        """
        Calculate the number of False Negatives (FN).

        Returns:
            int: Number of False Negatives.
        """
        return self.num_ref_instances - self.tp

    @property
    def rq(self) -> float:
        """
        Calculate the Recognition Quality (RQ) based on TP, FP, and FN.

        Returns:
            float: Recognition Quality (RQ).
        """
        if self.tp == 0:
            return (
                0.0 if self.num_pred_instances + self.num_ref_instances > 0 else np.nan
            )
        return self.tp / (self.tp + 0.5 * self.fp + 0.5 * self.fn)

    @property
    def sq(self) -> float:
        """
        Calculate the Segmentation Quality (SQ) based on IoU values.

        Returns:
            float: Segmentation Quality (SQ).
        """
        is_edge_case, result = self.edge_case_handler.handle_zero_tp(
            metric=ListMetric.IOU,
            tp=self.tp,
            num_pred_instances=self.num_pred_instances,
            num_ref_instances=self.num_ref_instances,
        )
        if is_edge_case:
            return result
        if ListMetric.IOU.name not in self.metric_dict:
            print("Requested SQ but no IOU metric evaluated")
            return None
        return np.sum(self.metric_dict[ListMetric.IOU.name]) / self.tp

    @property
    def sq_sd(self) -> float:
        """
        Calculate the standard deviation of Segmentation Quality (SQ) based on IoU values.

        Returns:
            float: Standard deviation of Segmentation Quality (SQ).
        """
        if ListMetric.IOU.name not in self.metric_dict:
            print("Requested SQ_SD but no IOU metric evaluated")
            return None
        return (
            np.std(self.metric_dict[ListMetric.IOU.name])
            if len(self.metric_dict[ListMetric.IOU.name]) > 0
            else self.edge_case_handler.handle_empty_list_std()
        )

    @property
    def pq(self) -> float:
        """
        Calculate the Panoptic Quality (PQ) based on SQ and RQ.

        Returns:
            float: Panoptic Quality (PQ).
        """
        sq = self.sq
        rq = self.rq
        if sq is None or rq is None:
            return None
        else:
            return sq * rq

    @property
    def sq_dsc(self) -> float:
        """
        Calculate the average Dice coefficient for matched instances. Analogue to segmentation quality but based on DSC.

        Returns:
            float: Average Dice coefficient.
        """
        is_edge_case, result = self.edge_case_handler.handle_zero_tp(
            metric=ListMetric.DSC,
            tp=self.tp,
            num_pred_instances=self.num_pred_instances,
            num_ref_instances=self.num_ref_instances,
        )
        if is_edge_case:
            return result
        if ListMetric.DSC.name not in self.metric_dict:
            print("Requested DSC but no DSC metric evaluated")
            return None
        return np.sum(self.metric_dict[ListMetric.DSC.name]) / self.tp

    @property
    def sq_dsc_sd(self) -> float:
        """
        Calculate the standard deviation of average Dice coefficient for matched instances. Analogue to segmentation quality but based on DSC.

        Returns:
            float: Standard deviation of Average Dice coefficient.
        """
        if ListMetric.DSC.name not in self.metric_dict:
            print("Requested DSC_SD but no DSC metric evaluated")
            return None
        return (
            np.std(self.metric_dict[ListMetric.DSC.name])
            if len(self.metric_dict[ListMetric.DSC.name]) > 0
            else self.edge_case_handler.handle_empty_list_std()
        )

    @property
    def pq_dsc(self) -> float:
        """
        Calculate the Panoptic Quality (PQ) based on DSC-based SQ and RQ.

        Returns:
            float: Panoptic Quality (PQ).
        """
        sq = self.sq_dsc
        rq = self.rq
        if sq is None or rq is None:
            return None
        else:
            return sq * rq

    @property
    def sq_assd(self) -> float:
        """
        Calculate the average average symmetric surface distance (ASSD) for matched instances. Analogue to segmentation quality but based on ASSD.

        Returns:
            float: average symmetric surface distance. (ASSD)
        """
        is_edge_case, result = self.edge_case_handler.handle_zero_tp(
            metric=ListMetric.ASSD,
            tp=self.tp,
            num_pred_instances=self.num_pred_instances,
            num_ref_instances=self.num_ref_instances,
        )
        if is_edge_case:
            return result
        if ListMetric.ASSD.name not in self.metric_dict:
            print("Requested ASSD but no ASSD metric evaluated")
            return None
        return np.sum(self.metric_dict[ListMetric.ASSD.name]) / self.tp

    @property
    def sq_assd_sd(self) -> float:
        """
        Calculate the standard deviation of average symmetric surface distance (ASSD) for matched instances. Analogue to segmentation quality but based on ASSD.
        Returns:
            float: Standard deviation of average symmetric surface distance (ASSD).
        """
        if ListMetric.ASSD.name not in self.metric_dict:
            print("Requested ASSD_SD but no ASSD metric evaluated")
            return None
        return (
            np.std(self.metric_dict[ListMetric.ASSD.name])
            if len(self.metric_dict[ListMetric.ASSD.name]) > 0
            else self.edge_case_handler.handle_empty_list_std()
        )

    @property
    def pq_assd(self) -> float:
        """
        Calculate the Panoptic Quality (PQ) based on ASSD-based SQ and RQ.

        Returns:
            float: Panoptic Quality (PQ).
        """
        return self.sq_assd * self.rq


# TODO make general getter that takes metric enum and std or not
# splits up into lists or not
# use below structure
def getter(value: int):
    return value


class Test(object):
    def __init__(self) -> None:
        self.x: int
        self.y: int

    # x = property(fget=getter(value=45))

    def __getattribute__(self, __name: str) -> Any:
        attr = None
        try:
            attr = object.__getattribute__(self, __name)
        except AttributeError as e:
            pass
        if attr is None:
            value = getter(5)
            setattr(self, __name, value)
            return value
        else:
            return attr

    # def __getattribute__(self, name):
    #    if some_predicate(name):
    #        # ...
    #    else:
    #        # Default behaviour
    #        return object.__getattribute__(self, name)


if __name__ == "__main__":
    c = Test()

    print(c.x)

    c.x = 4

    print(c.x)
