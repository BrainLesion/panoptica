from time import perf_counter
from typing import Type

from panoptica.instance_approximator import InstanceApproximator
from panoptica.instance_evaluator import evaluate_matched_instance
from panoptica.instance_matcher import InstanceMatchingAlgorithm
from panoptica.metrics import Metric, _Metric
from panoptica.panoptica_result import PanopticaResult
from panoptica.utils.timing import measure_time
from panoptica.utils import EdgeCaseHandler
from panoptica.utils.citation_reminder import citation_reminder
from panoptica.utils.processing_pair import (
    MatchedInstancePair,
    SemanticPair,
    UnmatchedInstancePair,
    _ProcessingPair,
    InputType,
)
import numpy as np
from panoptica.utils.config import SupportsConfig
from panoptica.utils.segmentation_class import SegmentationClassGroups, LabelGroup


class Panoptica_Evaluator(SupportsConfig):

    def __init__(
        self,
        expected_input: InputType = InputType.MATCHED_INSTANCE,
        instance_approximator: InstanceApproximator | None = None,
        instance_matcher: InstanceMatchingAlgorithm | None = None,
        edge_case_handler: EdgeCaseHandler | None = None,
        segmentation_class_groups: SegmentationClassGroups | None = None,
        eval_metrics: list[Metric] = [Metric.DSC, Metric.IOU, Metric.ASSD, Metric.RVD],
        decision_metric: Metric | None = None,
        decision_threshold: float | None = None,
        log_times: bool = False,
        verbose: bool = False,
    ) -> None:
        """Creates a Panoptica_Evaluator, that saves some parameters to be used for all subsequent evaluations

        Args:
            expected_input (type, optional): Expected DataPair Input Type. Defaults to InputType.MATCHED_INSTANCE (which is type(MatchedInstancePair)).
            instance_approximator (InstanceApproximator | None, optional): Determines which instance approximator is used if necessary. Defaults to None.
            instance_matcher (InstanceMatchingAlgorithm | None, optional): Determines which instance matching algorithm is used if necessary. Defaults to None.
            iou_threshold (float, optional): Iou Threshold for evaluation. Defaults to 0.5.
        """
        self.__expected_input = expected_input
        #
        self.__instance_approximator = instance_approximator
        self.__instance_matcher = instance_matcher
        self.__eval_metrics = eval_metrics
        self.__decision_metric = decision_metric
        self.__decision_threshold = decision_threshold

        self.__segmentation_class_groups = segmentation_class_groups

        self.__edge_case_handler = (
            edge_case_handler if edge_case_handler is not None else EdgeCaseHandler()
        )
        if self.__decision_metric is not None:
            assert (
                self.__decision_threshold is not None
            ), "decision metric set but no decision threshold for it"
        #
        self.__log_times = log_times
        self.__verbose = verbose

    @classmethod
    def _yaml_repr(cls, node) -> dict:
        return {
            "expected_input": node.__expected_input,
            "instance_approximator": node.__instance_approximator,
            "instance_matcher": node.__instance_matcher,
            "edge_case_handler": node.__edge_case_handler,
            "segmentation_class_groups": node.__segmentation_class_groups,
            "eval_metrics": node.__eval_metrics,
            "decision_metric": node.__decision_metric,
            "decision_threshold": node.__decision_threshold,
            "log_times": node.__log_times,
            "verbose": node.__verbose,
        }

    @citation_reminder
    @measure_time
    def evaluate(
        self,
        prediction_arr: np.ndarray,
        reference_arr: np.ndarray,
        result_all: bool = True,
        verbose: bool | None = None,
    ) -> dict[str, tuple[PanopticaResult, dict[str, _ProcessingPair]]]:
        processing_pair = self.__expected_input(prediction_arr, reference_arr)
        assert isinstance(
            processing_pair, self.__expected_input.value
        ), f"input not of expected type {self.__expected_input}"

        if self.__segmentation_class_groups is None:
            return {
                "ungrouped": panoptic_evaluate(
                    processing_pair=processing_pair,
                    edge_case_handler=self.__edge_case_handler,
                    instance_approximator=self.__instance_approximator,
                    instance_matcher=self.__instance_matcher,
                    eval_metrics=self.__eval_metrics,
                    decision_metric=self.__decision_metric,
                    decision_threshold=self.__decision_threshold,
                    result_all=result_all,
                    log_times=self.__log_times,
                    verbose=True if verbose is None else verbose,
                    verbose_calc=self.__verbose if verbose is None else verbose,
                )
            }

        self.__segmentation_class_groups.has_defined_labels_for(
            processing_pair.prediction_arr, raise_error=True
        )
        self.__segmentation_class_groups.has_defined_labels_for(
            processing_pair.reference_arr, raise_error=True
        )

        result_grouped = {}
        for group_name, label_group in self.__segmentation_class_groups.items():
            assert isinstance(label_group, LabelGroup)

            prediction_arr_grouped = label_group(processing_pair.prediction_arr)
            reference_arr_grouped = label_group(processing_pair.reference_arr)

            single_instance_mode = label_group.single_instance
            processing_pair_grouped = processing_pair.__class__(prediction_arr=prediction_arr_grouped, reference_arr=reference_arr_grouped)  # type: ignore
            decision_threshold = self.__decision_threshold
            if single_instance_mode and not isinstance(
                processing_pair, MatchedInstancePair
            ):
                processing_pair_grouped = MatchedInstancePair(
                    prediction_arr=processing_pair_grouped.prediction_arr,
                    reference_arr=processing_pair_grouped.reference_arr,
                )
                decision_threshold = 0.0

            result_grouped[group_name] = panoptic_evaluate(
                processing_pair=processing_pair_grouped,
                edge_case_handler=self.__edge_case_handler,
                instance_approximator=self.__instance_approximator,
                instance_matcher=self.__instance_matcher,
                eval_metrics=self.__eval_metrics,
                decision_metric=self.__decision_metric,
                decision_threshold=decision_threshold,
                result_all=result_all,
                log_times=self.__log_times,
                verbose=True if verbose is None else verbose,
                verbose_calc=self.__verbose if verbose is None else verbose,
            )
        return result_grouped


def panoptic_evaluate(
    processing_pair: (
        SemanticPair | UnmatchedInstancePair | MatchedInstancePair | PanopticaResult
    ),
    instance_approximator: InstanceApproximator | None = None,
    instance_matcher: InstanceMatchingAlgorithm | None = None,
    eval_metrics: list[Metric] = [Metric.DSC, Metric.IOU, Metric.ASSD],
    decision_metric: Metric | None = None,
    decision_threshold: float | None = None,
    edge_case_handler: EdgeCaseHandler | None = None,
    log_times: bool = False,
    result_all: bool = True,
    verbose=False,
    verbose_calc=False,
    **kwargs,
) -> tuple[PanopticaResult, dict[str, _ProcessingPair]]:
    """
    Perform panoptic evaluation on the given processing pair.

    Args:
        processing_pair (SemanticPair | UnmatchedInstancePair | MatchedInstancePair | PanopticaResult):
            The processing pair to be evaluated.
        instance_approximator (InstanceApproximator | None, optional):
            The instance approximator used for approximating instances in the SemanticPair.
        instance_matcher (InstanceMatchingAlgorithm | None, optional):
            The instance matcher used for matching instances in the UnmatchedInstancePair.
        iou_threshold (float, optional):
            The IoU threshold for evaluating matched instances. Defaults to 0.5.
        **kwargs:
            Additional keyword arguments.

    Returns:
        tuple[PanopticaResult, dict[str, _ProcessingPair]]:
            A tuple containing the panoptic result and a dictionary of debug data.

    Raises:
        AssertionError: If the input processing pair does not match the expected types.
        RuntimeError: If the end of the panoptic pipeline is reached without producing results.

    Example:
    >>> panoptic_evaluate(SemanticPair(...), instance_approximator=InstanceApproximator(), iou_threshold=0.6)
    (PanopticaResult(...), {'UnmatchedInstanceMap': _ProcessingPair(...), 'MatchedInstanceMap': _ProcessingPair(...)})
    """
    if verbose:
        print("Panoptic: Start Evaluation")
    if edge_case_handler is None:
        # use default edgecase handler
        edge_case_handler = EdgeCaseHandler()
    debug_data: dict[str, _ProcessingPair] = {}
    # First Phase: Instance Approximation
    if isinstance(processing_pair, PanopticaResult):
        if verbose:
            print("-- Input was Panoptic Result, will just return")
        return processing_pair, debug_data

    # Crops away unecessary space of zeroes
    processing_pair.crop_data()

    if isinstance(processing_pair, SemanticPair):
        assert (
            instance_approximator is not None
        ), "Got SemanticPair but not InstanceApproximator"
        if verbose:
            print("-- Got SemanticPair, will approximate instances")
        start = perf_counter()
        processing_pair = instance_approximator.approximate_instances(processing_pair)
        if log_times:
            print(f"-- Approximation took {perf_counter() - start} seconds")
        debug_data["UnmatchedInstanceMap"] = processing_pair.copy()

    # Second Phase: Instance Matching
    if isinstance(processing_pair, UnmatchedInstancePair):
        processing_pair = _handle_zero_instances_cases(
            processing_pair,
            eval_metrics=eval_metrics,
            edge_case_handler=edge_case_handler,
        )

    if isinstance(processing_pair, UnmatchedInstancePair):
        if verbose:
            print("-- Got UnmatchedInstancePair, will match instances")
        assert (
            instance_matcher is not None
        ), "Got UnmatchedInstancePair but not InstanceMatchingAlgorithm"
        start = perf_counter()
        processing_pair = instance_matcher.match_instances(
            processing_pair,
        )
        if log_times:
            print(f"-- Matching took {perf_counter() - start} seconds")

        debug_data["MatchedInstanceMap"] = processing_pair.copy()

    # Third Phase: Instance Evaluation
    if isinstance(processing_pair, MatchedInstancePair):
        processing_pair = _handle_zero_instances_cases(
            processing_pair,
            eval_metrics=eval_metrics,
            edge_case_handler=edge_case_handler,
        )

    if isinstance(processing_pair, MatchedInstancePair):
        if verbose:
            print("-- Got MatchedInstancePair, will evaluate instances")
        start = perf_counter()
        processing_pair = evaluate_matched_instance(
            processing_pair,
            eval_metrics=eval_metrics,
            decision_metric=decision_metric,
            decision_threshold=decision_threshold,
            edge_case_handler=edge_case_handler,
        )
        if log_times:
            print(f"-- Instance Evaluation took {perf_counter() - start} seconds")

    if isinstance(processing_pair, PanopticaResult):
        if result_all:
            processing_pair.calculate_all(print_errors=verbose_calc)
        return processing_pair, debug_data

    raise RuntimeError("End of panoptic pipeline reached without results")


def _handle_zero_instances_cases(
    processing_pair: UnmatchedInstancePair | MatchedInstancePair,
    edge_case_handler: EdgeCaseHandler,
    eval_metrics: list[_Metric] = [Metric.DSC, Metric.IOU, Metric.ASSD],
) -> UnmatchedInstancePair | MatchedInstancePair | PanopticaResult:
    """
    Handle edge cases when comparing reference and prediction masks.

    Args:
        num_ref_instances (int): Number of instances in the reference mask.
        num_pred_instances (int): Number of instances in the prediction mask.

    Returns:
        PanopticaResult: Result object with evaluation metrics.
    """
    n_reference_instance = processing_pair.n_reference_instance
    n_prediction_instance = processing_pair.n_prediction_instance

    panoptica_result_args = {
        "list_metrics": {Metric[k.name]: [] for k in eval_metrics},
        "tp": 0,
        "edge_case_handler": edge_case_handler,
        "reference_arr": processing_pair.reference_arr,
        "prediction_arr": processing_pair.prediction_arr,
    }

    is_edge_case = False

    # Handle cases where either the reference or the prediction is empty
    if n_prediction_instance == 0 and n_reference_instance == 0:
        # Both references and predictions are empty, perfect match
        n_reference_instance = 0
        n_prediction_instance = 0
        is_edge_case = True
    elif n_reference_instance == 0:
        # All references are missing, only false positives
        n_reference_instance = 0
        n_prediction_instance = n_prediction_instance
        is_edge_case = True
    elif n_prediction_instance == 0:
        # All predictions are missing, only false negatives
        n_reference_instance = n_reference_instance
        n_prediction_instance = 0
        is_edge_case = True

    if is_edge_case:
        panoptica_result_args["num_ref_instances"] = n_reference_instance
        panoptica_result_args["num_pred_instances"] = n_prediction_instance
        return PanopticaResult(**panoptica_result_args)

    return processing_pair
