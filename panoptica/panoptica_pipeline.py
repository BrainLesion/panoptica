from typing import Optional
from time import perf_counter
from typing import TYPE_CHECKING

from panoptica.instance_approximator import InstanceApproximator
from panoptica.instance_evaluator import evaluate_matched_instance
from panoptica.instance_matcher import InstanceMatchingAlgorithm
from panoptica.metrics import Metric
from panoptica.panoptica_result import PanopticaResult
from panoptica.utils import EdgeCaseHandler
from panoptica.utils.processing_pair import (
    MatchedInstancePair,
    SemanticPair,
    UnmatchedInstancePair,
    InputType,
    EvaluateInstancePair,
    IntermediateStepsData,
)
from panoptica._functionals import _get_voronoi_regions
import numpy as np

if TYPE_CHECKING:
    import torch
    import SimpleITK as sitk
    import nibabel as nib


def _panoptic_evaluate(
    input_pair: SemanticPair | UnmatchedInstancePair | MatchedInstancePair,
    instance_approximator: InstanceApproximator | None = None,
    instance_matcher: InstanceMatchingAlgorithm | None = None,
    instance_metrics: list[Metric] = [Metric.DSC, Metric.IOU, Metric.ASSD],
    global_metrics: list[Metric] = [Metric.DSC],
    decision_metric: Metric | None = None,
    decision_threshold: float | None = None,
    matching_threshold: float | None = None,
    edge_case_handler: EdgeCaseHandler | None = None,
    log_times: bool = False,
    result_all: bool = True,
    verbose=False,
    verbose_calc=False,
    label_group=None,
    **kwargs,
) -> PanopticaResult:
    """
    Perform panoptic evaluation on the given processing pair.

    Args:
        input_pair: The processing pair to be evaluated.
        instance_approximator: The instance approximator used for approximating instances.
        instance_matcher: The instance matcher used for matching instances.
        instance_metrics: List of metrics to calculate for each instance.
        global_metrics: List of metrics to calculate globally.
        decision_metric: Metric used for determining true positives.
        decision_threshold: Threshold for the decision metric.
        edge_case_handler: Handler for edge cases.
        log_times: Whether to log computation times.
        result_all: Whether to calculate all metrics.
        verbose: Whether to print verbose information.
        verbose_calc: Whether to print calculation details.
        label_group: Group of labels to consider.
        **kwargs: Additional keyword arguments.

    Returns:
        PanopticaResult: Result of the panoptic evaluation.

    Raises:
        AssertionError: If the input processing pair does not match the expected types.
        RuntimeError: If the end of the panoptic pipeline is reached without producing results.
    """
    if verbose:
        print("Panoptic: Start Evaluation")
    if edge_case_handler is None:
        edge_case_handler = EdgeCaseHandler()

    if "voxelspacing" not in kwargs:
        kwargs["voxelspacing"] = (1.0,) * input_pair.reference_arr.ndim

    # Setup IntermediateStepsData
    intermediate_steps_data: IntermediateStepsData = IntermediateStepsData(input_pair)
    # Crops away unnecessary space of zeroes
    input_pair.crop_data()

    # Create initial metadata for parts handling
    # Get metadata directly from the processing pair as a dictionary
    instance_metadata = input_pair.get_metadata()

    processing_pair = input_pair.copy()

    # First Phase: Instance Approximation
    processing_pair = _phase_instance_approximation(
        processing_pair,
        intermediate_steps_data,
        instance_approximator=instance_approximator,
        instance_metadata=instance_metadata,
        label_group=label_group,
        log_times=log_times,
        verbose=verbose,
    )

    # Second Phase: Instance Matching
    processing_pair = _phase_instance_matching(
        processing_pair,
        intermediate_steps_data,
        instance_metrics=instance_metrics,
        instance_metadata=instance_metadata,
        global_metrics=global_metrics,
        edge_case_handler=edge_case_handler,
        instance_matcher=instance_matcher,
        matching_threshold=matching_threshold,
        label_group=label_group,
        log_times=log_times,
        verbose=verbose,
        **kwargs,
    )

    # Third Phase: Instance Evaluation
    processing_pair = _phase_instance_evaluation(
        processing_pair,
        intermediate_steps_data,
        instance_metrics=instance_metrics,
        instance_metadata=instance_metadata,
        global_metrics=global_metrics,
        edge_case_handler=edge_case_handler,
        decision_metric=decision_metric,
        decision_threshold=decision_threshold,
        log_times=log_times,
        verbose=verbose,
        **kwargs,
    )

    if isinstance(processing_pair, EvaluateInstancePair):
        # Update instance counts from the processed pair if available
        if instance_metadata["original_n_preds"] == 0:
            instance_metadata["original_n_preds"] = processing_pair.n_pred_instances
        if instance_metadata["original_n_refs"] == 0:
            instance_metadata["original_n_refs"] = processing_pair.n_ref_instances

        # Detect if many-to-one mappings were used (like in MaximizeMergeMatching)
        # This happens when the effective number of prediction instances is less than original
        has_many_to_one_mappings = (
            processing_pair.n_pred_instances < instance_metadata["original_n_preds"]
        )

        # Use effective counts if many-to-one mappings were detected, otherwise use original counts
        final_n_pred_instances = (
            processing_pair.n_pred_instances
            if has_many_to_one_mappings
            else instance_metadata["original_n_preds"]
        )
        final_n_ref_instances = (
            processing_pair.n_ref_instances
            if has_many_to_one_mappings
            else instance_metadata["original_n_refs"]
        )

        processing_pair = PanopticaResult(
            reference_arr=processing_pair.reference_arr,
            prediction_arr=processing_pair.prediction_arr,
            processing_pair_orig_shape=instance_metadata["original_shape"],
            n_pred_instances=final_n_pred_instances,
            n_ref_instances=final_n_ref_instances,
            n_ref_labels=instance_metadata["n_ref_labels"],
            label_group=label_group,
            tp=processing_pair.tp,
            list_metrics=processing_pair.list_metrics,
            global_metrics=global_metrics,
            edge_case_handler=edge_case_handler,
            intermediate_steps_data=intermediate_steps_data,
            **kwargs,
        )

    if isinstance(processing_pair, PanopticaResult):
        processing_pair._global_metrics = global_metrics
        if result_all:
            processing_pair.calculate_all(print_errors=verbose_calc)
        return processing_pair

    raise RuntimeError("End of panoptic pipeline reached without results")


def _panoptic_evaluate_region_wise(
    input_pair: SemanticPair | UnmatchedInstancePair,
    instance_approximator: InstanceApproximator | None = None,
    instance_matcher: InstanceMatchingAlgorithm | None = None,
    instance_metrics: list[Metric] = [Metric.DSC, Metric.IOU, Metric.ASSD],
    global_metrics: list[Metric] = [Metric.DSC],
    edge_case_handler: EdgeCaseHandler | None = None,
    log_times: bool = False,
    result_all: bool = True,
    verbose=False,
    verbose_calc=False,
    label_group=None,
    **kwargs,
) -> PanopticaResult:
    """
    Perform panoptic evaluation on the given processing pair.

    Args:
        input_pair: The processing pair to be evaluated.
        instance_approximator: The instance approximator used for approximating instances.
        instance_matcher: The instance matcher used for matching instances.
        instance_metrics: List of metrics to calculate for each instance.
        global_metrics: List of metrics to calculate globally.
        edge_case_handler: Handler for edge cases.
        log_times: Whether to log computation times.
        result_all: Whether to calculate all metrics.
        verbose: Whether to print verbose information.
        verbose_calc: Whether to print calculation details.
        label_group: Group of labels to consider.
        **kwargs: Additional keyword arguments.

    Returns:
        PanopticaResult: Result of the panoptic evaluation.

    Raises:
        AssertionError: If the input processing pair does not match the expected types.
        RuntimeError: If the end of the panoptic pipeline is reached without producing results.
    """
    if verbose:
        print("Panoptic: Start Evaluation")
    if edge_case_handler is None:
        edge_case_handler = EdgeCaseHandler()

    if "voxelspacing" not in kwargs:
        kwargs["voxelspacing"] = (1.0,) * input_pair.reference_arr.ndim

    # Setup IntermediateStepsData
    intermediate_steps_data: IntermediateStepsData = IntermediateStepsData(input_pair)
    # Crops away unnecessary space of zeroes
    input_pair.crop_data()

    # Create initial metadata for parts handling
    # Get metadata directly from the processing pair as a dictionary
    instance_metadata = input_pair.get_metadata()

    processing_pair = input_pair.copy()

    # First Phase: Instance Approximation
    processing_pair = _phase_instance_approximation(
        processing_pair,
        intermediate_steps_data,
        instance_approximator=instance_approximator,
        instance_metadata=instance_metadata,
        label_group=label_group,
        log_times=log_times,
        verbose=verbose,
    )

    assert isinstance(
        processing_pair, UnmatchedInstancePair
    ), f"Expected UnmatchedInstancePair, got {type(processing_pair)}"
    processing_pair = _handle_zero_instances_cases(
        processing_pair,
        eval_metrics=instance_metrics,
        global_metrics=global_metrics,
        edge_case_handler=edge_case_handler,
    )

    # proceed if pipeline only if no edge case handling necessary
    if not isinstance(processing_pair, PanopticaResult):

        # create regions and label to regions
        region_map, num_features = _get_voronoi_regions(
            processing_pair.reference_arr, processing_pair.n_ref_instances
        )
        assert (
            num_features > 0
        ), "Expected at least one region in the reference mask for region-wise evaluation"

        region2result_map: dict[int, PanopticaResult] = {}

        for i in range(1, num_features + 1):
            region_mask = region_map == i

            intermediate_steps_data_r: IntermediateStepsData = IntermediateStepsData(
                input_pair
            )

            # multiply region mask with both prediction and reference arr
            processing_pair_r = UnmatchedInstancePair(
                processing_pair.prediction_arr * region_mask,
                processing_pair.reference_arr * region_mask,
            )

            # Second Phase: Instance Matching
            processing_pair_r = _phase_instance_matching(
                processing_pair_r,
                intermediate_steps_data_r,
                instance_metrics=instance_metrics,
                instance_metadata=instance_metadata,
                global_metrics=global_metrics,
                edge_case_handler=edge_case_handler,
                instance_matcher=instance_matcher,
                label_group=label_group,
                log_times=log_times,
                verbose=verbose,
                **kwargs,
            )

            # Third Phase: Instance Evaluation
            processing_pair_r = _phase_instance_evaluation(
                processing_pair_r,
                intermediate_steps_data_r,
                instance_metrics=instance_metrics,
                instance_metadata=instance_metadata,
                global_metrics=global_metrics,
                edge_case_handler=edge_case_handler,
                decision_metric=None,
                decision_threshold=None,
                log_times=log_times,
                verbose=verbose,
                **kwargs,
            )

            if isinstance(processing_pair_r, EvaluateInstancePair):
                # Update instance counts from the processed pair if available
                if instance_metadata["original_n_preds"] == 0:
                    instance_metadata["original_n_preds"] = (
                        processing_pair_r.n_pred_instances
                    )
                if instance_metadata["original_n_refs"] == 0:
                    instance_metadata["original_n_refs"] = (
                        processing_pair_r.n_ref_instances
                    )

                # Detect if many-to-one mappings were used (like in MaximizeMergeMatching)
                # This happens when the effective number of prediction instances is less than original
                has_many_to_one_mappings = (
                    processing_pair_r.n_pred_instances
                    < instance_metadata["original_n_preds"]
                )

                # Use effective counts if many-to-one mappings were detected, otherwise use original counts
                final_n_pred_instances = (
                    processing_pair_r.n_pred_instances
                    if has_many_to_one_mappings
                    else instance_metadata["original_n_preds"]
                )
                final_n_ref_instances = (
                    processing_pair_r.n_ref_instances
                    if has_many_to_one_mappings
                    else instance_metadata["original_n_refs"]
                )

                processing_pair_r = PanopticaResult(
                    reference_arr=processing_pair_r.reference_arr,
                    prediction_arr=processing_pair_r.prediction_arr,
                    processing_pair_orig_shape=instance_metadata["original_shape"],
                    n_pred_instances=final_n_pred_instances,
                    n_ref_instances=final_n_ref_instances,
                    n_ref_labels=instance_metadata["n_ref_labels"],
                    label_group=label_group,
                    tp=processing_pair_r.tp,
                    list_metrics=processing_pair_r.list_metrics,
                    global_metrics=global_metrics,
                    edge_case_handler=edge_case_handler,
                    intermediate_steps_data=intermediate_steps_data_r,
                    **kwargs,
                )

            if isinstance(processing_pair_r, PanopticaResult):
                processing_pair_r._global_metrics = global_metrics
                if result_all:
                    processing_pair_r.calculate_all(print_errors=False)

                region2result_map[i] = processing_pair_r

        if len(region2result_map) == num_features:
            # Combine results from all regions into a single PanopticaResult
            combined_result = PanopticaResult(
                reference_arr=input_pair.reference_arr,
                prediction_arr=input_pair.prediction_arr,
                processing_pair_orig_shape=instance_metadata["original_shape"],
                n_pred_instances=np.nan,
                n_ref_instances=np.nan,  # We set n_ref_instances to the number of regions, as each region corresponds to one reference instance
                n_ref_labels=instance_metadata["n_ref_labels"],
                label_group=label_group,
                tp=np.nan,
                list_metrics={},
                global_metrics=global_metrics,
                edge_case_handler=edge_case_handler,
                intermediate_steps_data=intermediate_steps_data,
                **kwargs,
            )
    else:
        # In case edge case handling already produced a result, we skip the region-wise processing and return the edge case result directly
        combined_result = processing_pair
        combined_result.tp = (
            np.nan
        )  # Set tp to nan to indicate that no true positive calculation was done
        combined_result.n_pred_instances = np.nan
        combined_result.n_ref_instances = np.nan
        num_features = 0

    # combined global metrics post-hoc
    for gm in global_metrics:
        gm_attr_name = f"global_bin_{gm.name.lower()}"
        rm_attr_name = f"region_avg_{gm.name.lower()}"

        if num_features > 0:
            setattr(
                combined_result,
                rm_attr_name,
                np.mean(
                    [
                        getattr(region2result_map[i], gm_attr_name)
                        for i in range(1, num_features + 1)
                    ]
                ),
            )
        else:
            setattr(combined_result, rm_attr_name, 0.0)

    return combined_result


def _phase_instance_approximation(
    processing_pair: SemanticPair,
    intermediate_steps_data: Optional[IntermediateStepsData],
    instance_approximator: InstanceApproximator,
    instance_metadata: dict,
    label_group=None,
    log_times=False,
    verbose=False,
):
    # First Phase: Instance Approximation
    if isinstance(processing_pair, SemanticPair):
        if intermediate_steps_data:
            intermediate_steps_data.add_intermediate_arr_data(
                processing_pair.copy(), InputType.SEMANTIC
            )
        assert (
            instance_approximator is not None
        ), "Got SemanticPair but not InstanceApproximator"
        if verbose:
            print("-- Got SemanticPair, will approximate instances")
        start = perf_counter()

        processing_pair = instance_approximator.approximate_instances(
            processing_pair,
            label_group=label_group,
        )

        if log_times:
            print(f"-- Approximation took {perf_counter() - start} seconds")

        # Update instance metadata after approximation
        if isinstance(processing_pair, (UnmatchedInstancePair, MatchedInstancePair)):
            instance_metadata["original_n_preds"] = processing_pair.n_pred_instances
            instance_metadata["original_n_refs"] = processing_pair.n_ref_instances

    return processing_pair


def _phase_instance_matching(
    processing_pair: UnmatchedInstancePair,
    intermediate_steps_data: IntermediateStepsData,
    instance_metrics: list[Metric],
    instance_metadata: dict,
    global_metrics: list[Metric],
    edge_case_handler: EdgeCaseHandler,
    instance_matcher: InstanceMatchingAlgorithm,
    matching_threshold: float | None = None,
    label_group=None,
    log_times=False,
    verbose=False,
    **kwargs,
):
    # Second Phase: Instance Matching
    if isinstance(processing_pair, UnmatchedInstancePair):
        intermediate_steps_data.add_intermediate_arr_data(
            processing_pair.copy(), InputType.UNMATCHED_INSTANCE
        )
        processing_pair = _handle_zero_instances_cases(
            processing_pair,
            eval_metrics=instance_metrics,
            global_metrics=global_metrics,
            edge_case_handler=edge_case_handler,
        )

    if isinstance(processing_pair, UnmatchedInstancePair):
        if verbose:
            print("-- Got UnmatchedInstancePair, will match instances")
        assert (
            instance_matcher is not None
        ), "Got UnmatchedInstancePair but not InstanceMatchingAlgorithm"
        start = perf_counter()

        match_kwargs = {
            "label_group": label_group,
            "n_ref_labels": instance_metadata["n_ref_labels"],
            "processing_pair_orig_shape": instance_metadata["original_shape"],
            **kwargs,
        }
        if matching_threshold is not None:
            match_kwargs["matching_threshold"] = matching_threshold

        processing_pair = instance_matcher.match_instances(
            processing_pair,
            **match_kwargs,
        )
        if log_times:
            print(f"-- Matching took {perf_counter() - start} seconds")
    return processing_pair


def _phase_instance_evaluation(
    processing_pair: UnmatchedInstancePair | MatchedInstancePair,
    intermediate_steps_data: IntermediateStepsData,
    instance_metrics: list[Metric],
    instance_metadata: dict,
    global_metrics: list[Metric],
    edge_case_handler: EdgeCaseHandler,
    decision_metric: Metric | None,
    decision_threshold: float | None,
    log_times=False,
    verbose=False,
    **kwargs,
):
    # Third Phase: Instance Evaluation
    if isinstance(processing_pair, MatchedInstancePair):
        intermediate_steps_data.add_intermediate_arr_data(
            processing_pair.copy(), InputType.MATCHED_INSTANCE
        )
        processing_pair = _handle_zero_instances_cases(
            processing_pair,
            eval_metrics=instance_metrics,
            global_metrics=global_metrics,
            edge_case_handler=edge_case_handler,
        )

    if isinstance(processing_pair, MatchedInstancePair):
        if verbose:
            print("-- Got MatchedInstancePair, will evaluate instances")
        start = perf_counter()
        processing_pair = evaluate_matched_instance(
            processing_pair,
            eval_metrics=instance_metrics,
            decision_metric=decision_metric,
            decision_threshold=decision_threshold,
            processing_pair_orig_shape=instance_metadata["original_shape"],
            n_ref_labels=instance_metadata["n_ref_labels"],
            **kwargs,
        )
        if log_times:
            print(f"-- Instance Evaluation took {perf_counter() - start} seconds")
    return processing_pair


def _handle_zero_instances_cases(
    processing_pair: UnmatchedInstancePair | MatchedInstancePair,
    edge_case_handler: EdgeCaseHandler,
    global_metrics: list[Metric],
    eval_metrics: list[Metric] = [Metric.DSC, Metric.IOU, Metric.ASSD],
) -> UnmatchedInstancePair | MatchedInstancePair | PanopticaResult:
    """
    Handle edge cases when comparing reference and prediction masks.

    Args:
        processing_pair: The processing pair containing reference and prediction data.
        edge_case_handler: Handler for edge cases.
        global_metrics: List of global metrics to calculate.
        eval_metrics: List of evaluation metrics to calculate for instances.

    Returns:
        UnmatchedInstancePair | MatchedInstancePair | PanopticaResult: The processed processing pair or evaluation result.
    """
    n_reference_instance = processing_pair.n_ref_instances
    n_prediction_instance = processing_pair.n_pred_instances

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
        panoptica_result_args["global_metrics"] = global_metrics
        panoptica_result_args["n_ref_instances"] = n_reference_instance
        panoptica_result_args["n_pred_instances"] = n_prediction_instance
        return PanopticaResult(**panoptica_result_args)

    return processing_pair
