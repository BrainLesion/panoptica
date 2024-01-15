from abc import ABC, abstractmethod
from time import perf_counter
from typing import Type

from panoptica.instance_approximator import InstanceApproximator
from panoptica.instance_evaluator import evaluate_matched_instance
from panoptica.instance_matcher import InstanceMatchingAlgorithm
from panoptica.metrics import Metrics, _MatchingMetric
from panoptica.panoptic_result import PanopticaResult
from panoptica.timing import measure_time
from panoptica.utils import EdgeCaseHandler
from panoptica.utils.processing_pair import (
    MatchedInstancePair,
    SemanticPair,
    UnmatchedInstancePair,
    _ProcessingPair,
)
from panoptica.utils.citation_reminder import citation_reminder


class Panoptic_Evaluator:
    def __init__(
        self,
        expected_input: Type[SemanticPair]
        | Type[UnmatchedInstancePair]
        | Type[MatchedInstancePair] = MatchedInstancePair,
        instance_approximator: InstanceApproximator | None = None,
        instance_matcher: InstanceMatchingAlgorithm | None = None,
        edge_case_handler: EdgeCaseHandler | None = None,
        eval_metrics: list[_MatchingMetric] = [Metrics.DSC, Metrics.IOU, Metrics.ASSD],
        decision_metric: _MatchingMetric | None = None,
        decision_threshold: float | None = None,
        log_times: bool = False,
        verbose: bool = False,
    ) -> None:
        """Creates a Panoptic_Evaluator, that saves some parameters to be used for all subsequent evaluations

        Args:
            expected_input (type, optional): Expected DataPair Input. Defaults to type(MatchedInstancePair).
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

    @citation_reminder
    @measure_time
    def evaluate(
        self,
        processing_pair: SemanticPair
        | UnmatchedInstancePair
        | MatchedInstancePair
        | PanopticaResult,
    ) -> tuple[PanopticaResult, dict[str, _ProcessingPair]]:
        assert (
            type(processing_pair) == self.__expected_input
        ), f"input not of expected type {self.__expected_input}"
        return panoptic_evaluate(
            processing_pair=processing_pair,
            edge_case_handler=self.__edge_case_handler,
            instance_approximator=self.__instance_approximator,
            instance_matcher=self.__instance_matcher,
            eval_metrics=self.__eval_metrics,
            decision_metric=self.__decision_metric,
            decision_threshold=self.__decision_threshold,
            log_times=self.__log_times,
            verbose=self.__verbose,
        )


def panoptic_evaluate(
    processing_pair: SemanticPair
    | UnmatchedInstancePair
    | MatchedInstancePair
    | PanopticaResult,
    instance_approximator: InstanceApproximator | None = None,
    instance_matcher: InstanceMatchingAlgorithm | None = None,
    eval_metrics: list[_MatchingMetric] = [Metrics.DSC, Metrics.IOU, Metrics.ASSD],
    decision_metric: _MatchingMetric | None = None,
    decision_threshold: float | None = None,
    edge_case_handler: EdgeCaseHandler | None = None,
    log_times: bool = False,
    verbose: bool = False,
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
    print("Panoptic: Start Evaluation")
    if edge_case_handler is None:
        # use default edgecase handler
        edge_case_handler = EdgeCaseHandler()
    debug_data: dict[str, _ProcessingPair] = {}
    # First Phase: Instance Approximation
    if isinstance(processing_pair, PanopticaResult):
        print("-- Input was Panoptic Result, will just return")
        return processing_pair, debug_data

    # Crops away unecessary space of zeroes
    processing_pair.crop_data()

    if isinstance(processing_pair, SemanticPair):
        assert (
            instance_approximator is not None
        ), "Got SemanticPair but not InstanceApproximator"
        print("-- Got SemanticPair, will approximate instances")
        processing_pair = instance_approximator.approximate_instances(processing_pair)
        start = perf_counter()
        processing_pair = instance_approximator.approximate_instances(processing_pair)
        if log_times:
            print(f"-- Approximation took {perf_counter() - start} seconds")
        debug_data["UnmatchedInstanceMap"] = processing_pair.copy()

    # Second Phase: Instance Matching
    if isinstance(processing_pair, UnmatchedInstancePair):
        processing_pair = _handle_zero_instances_cases(
            processing_pair, edge_case_handler=edge_case_handler
        )

    if isinstance(processing_pair, UnmatchedInstancePair):
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
            processing_pair, edge_case_handler=edge_case_handler
        )

    if isinstance(processing_pair, MatchedInstancePair):
        print("-- Got MatchedInstancePair, will evaluate instances")
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
        return processing_pair, debug_data

    raise RuntimeError("End of panoptic pipeline reached without results")


def _handle_zero_instances_cases(
    processing_pair: UnmatchedInstancePair | MatchedInstancePair,
    edge_case_handler: EdgeCaseHandler,
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

    # Handle cases where either the reference or the prediction is empty
    if n_prediction_instance == 0 and n_reference_instance == 0:
        # Both references and predictions are empty, perfect match
        return PanopticaResult(
            num_ref_instances=0,
            num_pred_instances=0,
            tp=0,
            list_metrics={},
            edge_case_handler=edge_case_handler,
        )
    if n_reference_instance == 0:
        # All references are missing, only false positives
        return PanopticaResult(
            num_ref_instances=0,
            num_pred_instances=n_prediction_instance,
            tp=0,
            list_metrics={},
            edge_case_handler=edge_case_handler,
        )
    if n_prediction_instance == 0:
        # All predictions are missing, only false negatives
        return PanopticaResult(
            num_ref_instances=n_reference_instance,
            num_pred_instances=0,
            tp=0,
            list_metrics={},
            edge_case_handler=edge_case_handler,
        )
    return processing_pair
