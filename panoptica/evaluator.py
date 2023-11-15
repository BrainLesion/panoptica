from abc import ABC, abstractmethod
from typing import Type

import numpy as np

from panoptica.instance_approximator import InstanceApproximator
from panoptica.instance_evaluator import evaluate_matched_instance
from panoptica.instance_matcher import InstanceMatchingAlgorithm
from panoptica.result import PanopticaResult
from panoptica.timing import measure_time
from panoptica.utils.datatypes import (
    MatchedInstancePair,
    SemanticPair,
    UnmatchedInstancePair,
    _ProcessingPair,
)


class Panoptic_Evaluator:
    def __init__(
        self,
        expected_input: Type[SemanticPair] | Type[UnmatchedInstancePair] | Type[MatchedInstancePair] = MatchedInstancePair,
        instance_approximator: InstanceApproximator | None = None,
        instance_matcher: InstanceMatchingAlgorithm | None = None,
        log_times: bool = False,
        verbose: bool = False,
        iou_threshold: float = 0.5,
    ) -> None:
        """Creates a Panoptic_Evaluator, that saves some parameters to be used for all subsequent evaluations

        Args:
            expected_input (type, optional): Expected DataPair Input. Defaults to type(MatchedInstancePair).
            instance_approximator (InstanceApproximator | None, optional): Determines which instance approximator is used if necessary. Defaults to None.
            instance_matcher (InstanceMatchingAlgorithm | None, optional): Determines which instance matching algorithm is used if necessary. Defaults to None.
            iou_threshold (float, optional): Iou Threshold for evaluation. Defaults to 0.5.
        """
        self.__expected_input = expected_input
        self.__instance_approximator = instance_approximator
        self.__instance_matcher = instance_matcher
        self.__iou_threshold = iou_threshold
        self.__log_times = log_times
        self.__verbose = verbose

    @measure_time
    def evaluate(
        self, processing_pair: SemanticPair | UnmatchedInstancePair | MatchedInstancePair | PanopticaResult
    ) -> tuple[PanopticaResult, dict[str, _ProcessingPair]]:
        assert type(processing_pair) == self.__expected_input, f"input not of expected type {self.__expected_input}"
        return panoptic_evaluate(
            processing_pair=processing_pair,
            instance_approximator=self.__instance_approximator,
            instance_matcher=self.__instance_matcher,
            iou_threshold=self.__iou_threshold,
            log_times=self.__log_times,
            verbose=self.__verbose,
        )


def panoptic_evaluate(
    processing_pair: SemanticPair | UnmatchedInstancePair | MatchedInstancePair | PanopticaResult,
    instance_approximator: InstanceApproximator | None = None,
    instance_matcher: InstanceMatchingAlgorithm | None = None,
    log_times: bool = False,
    verbose: bool = False,
    iou_threshold: float = 0.5,
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
    debug_data: dict[str, _ProcessingPair] = {}
    # First Phase: Instance Approximation
    if isinstance(processing_pair, PanopticaResult):
        print("-- Input was Panoptic Result, will just return")
        return processing_pair, debug_data

    # Crops away unecessary space of zeroes
    processing_pair.crop_data()

    if isinstance(processing_pair, SemanticPair):
        assert instance_approximator is not None, "Got SemanticPair but not InstanceApproximator"
        print("-- Got SemanticPair, will approximate instances")
        processing_pair = instance_approximator.approximate_instances(processing_pair)
        debug_data["UnmatchedInstanceMap"] = processing_pair.copy()

    # Second Phase: Instance Matching
    if isinstance(processing_pair, UnmatchedInstancePair):
        processing_pair = _handle_zero_instances_cases(processing_pair)

    if isinstance(processing_pair, UnmatchedInstancePair):
        print("-- Got UnmatchedInstancePair, will match instances")
        assert instance_matcher is not None, "Got UnmatchedInstancePair but not InstanceMatchingAlgorithm"
        processing_pair = instance_matcher.match_instances(processing_pair)

        debug_data["MatchedInstanceMap"] = processing_pair.copy()

    # Third Phase: Instance Evaluation
    if isinstance(processing_pair, MatchedInstancePair):
        processing_pair = _handle_zero_instances_cases(processing_pair)

    if isinstance(processing_pair, MatchedInstancePair):
        print("-- Got MatchedInstancePair, will evaluate instances")
        processing_pair = evaluate_matched_instance(processing_pair, iou_threshold=iou_threshold)

    if isinstance(processing_pair, PanopticaResult):
        return processing_pair, debug_data

    raise RuntimeError("End of panoptic pipeline reached without results")


def _handle_zero_instances_cases(
    processing_pair: UnmatchedInstancePair | MatchedInstancePair,
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
            dice_list=[],
            iou_list=[],
            assd_list=[],
        )
    if n_reference_instance == 0:
        # All references are missing, only false positives
        return PanopticaResult(
            num_ref_instances=0,
            num_pred_instances=n_prediction_instance,
            tp=0,
            dice_list=[],
            iou_list=[],
            assd_list=[],
        )
    if n_prediction_instance == 0:
        # All predictions are missing, only false negatives
        return PanopticaResult(
            num_ref_instances=n_reference_instance,
            num_pred_instances=0,
            tp=0,
            dice_list=[],
            iou_list=[],
            assd_list=[],
        )
    return processing_pair
