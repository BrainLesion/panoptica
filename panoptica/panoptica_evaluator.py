from panoptica.panoptica_pipeline import _phase_instance_approximation
from time import perf_counter
from typing import Literal, Union
from typing import TYPE_CHECKING

from panoptica.instance_approximator import InstanceApproximator
from panoptica.instance_evaluator import evaluate_matched_instance
from panoptica.instance_matcher import InstanceMatchingAlgorithm, ThresholdBasedMatching
from panoptica.metrics import Metric
from panoptica.panoptica_result import PanopticaResult, PanopticaAUTCResult
from panoptica.utils.timing import measure_time
from panoptica.utils import EdgeCaseHandler
from panoptica.utils.citation_reminder import citation_reminder
from panoptica.utils.processing_pair import (
    MatchedInstancePair,
    SemanticPair,
    UnmatchedInstancePair,
    InputType,
)
from panoptica.utils.input_check_and_conversion.sanity_checker import (
    sanity_check_and_convert_to_array,
)
import numpy as np
from panoptica.utils.config import SupportsConfig
from panoptica.utils.segmentation_class import (
    SegmentationClassGroups,
    LabelGroup,
    _NoSegmentationClassGroups,
)
from panoptica.panoptica_pipeline import (
    _panoptic_evaluate,
    _panoptic_evaluate_region_wise,
)
from pathlib import Path

if TYPE_CHECKING:
    import torch
    import SimpleITK as sitk
    import nibabel as nib


class Panoptica_Evaluator(SupportsConfig):
    def __init__(
        self,
        expected_input: InputType = InputType.MATCHED_INSTANCE,
        instance_approximator: InstanceApproximator | None = None,
        instance_matcher: InstanceMatchingAlgorithm | None = None,
        edge_case_handler: EdgeCaseHandler | None = None,
        segmentation_class_groups: SegmentationClassGroups | None = None,
        instance_metrics: list[Metric] = [
            Metric.DSC,
            Metric.IOU,
            Metric.ASSD,
            Metric.RVD,
        ],
        global_metrics: list[Metric] = [Metric.DSC],
        decision_metric: Metric | None = None,
        decision_threshold: float | None = None,
        per_region_evaluation: bool = False,
        save_group_times: bool = False,
        log_times: bool = False,
        verbose: bool = False,
    ) -> None:
        """Creates a Panoptica_Evaluator, that saves some parameters to be used for all subsequent evaluations

        Args:
            expected_input (type, optional): Expected DataPair Input Type. Defaults to InputType.MATCHED_INSTANCE (which is type(MatchedInstancePair)).
            instance_approximator (InstanceApproximator | None, optional): Determines which instance approximator is used if necessary. Defaults to None.
            instance_matcher (InstanceMatchingAlgorithm | None, optional): Determines which instance matching algorithm is used if necessary. Defaults to None.

            edge_case_handler (edge_case_handler, optional): EdgeCaseHandler to be used. If none, will create the default one
            segmentation_class_groups (SegmentationClassGroups, optional): If not none, will evaluate per class group defined, instead of over all at the same time. A class group is a collection of labels that are considered of the same class / structure.

            instance_metrics (list[Metric]): List of all metrics that should be calculated between all instances
            global_metrics (list[Metric]): List of all metrics that should be calculated on the global binary masks

            decision_metric: (Metric | None, optional): This metric is the final decision point between True Positive and False Positive. Can be left away if the matching algorithm is used (it will match by a metric and threshold already)
            decision_threshold: (float | None, optional): Threshold for the decision_metric

            save_group_times(bool): If true, will save the computation time of each sample and put that into the result object.
            log_times (bool): If true, will print the times for the different phases of the pipeline.
            verbose (bool): If true, will spit out more details than you want.
        """
        self.__expected_input = expected_input
        #
        self.__instance_approximator = instance_approximator
        self.__instance_matcher = instance_matcher
        self.__eval_metrics = instance_metrics
        self.__global_metrics = global_metrics
        self.__decision_metric = decision_metric
        self.__decision_threshold = decision_threshold
        self.__resulting_metric_keys = None
        self.__save_group_times = save_group_times
        self.__per_region_evaluation = per_region_evaluation
        if self.__per_region_evaluation:
            assert (
                self.__decision_metric is None
            ), "Decision metric not supported for region-wise evaluation, as there are no matched instances. Please set decision_metric to None."
            assert (
                self.__decision_threshold is None
            ), "Decision threshold not supported for region-wise evaluation, as there are no matched instances. Please set decision_threshold to None."

        if segmentation_class_groups is None:
            segmentation_class_groups = _NoSegmentationClassGroups()
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
            "instance_metrics": node.__eval_metrics,
            "global_metrics": node.__global_metrics,
            "decision_metric": node.__decision_metric,
            "decision_threshold": node.__decision_threshold,
            "per_region_evaluation": node.__per_region_evaluation,
            "save_group_times": node.__save_group_times,
            "log_times": node.__log_times,
            "verbose": node.__verbose,
        }

    @citation_reminder
    @measure_time
    def evaluate(
        self,
        prediction_arr: Union[
            str,
            Path,
            np.ndarray,
            "torch.Tensor",
            "nib.nifti1.Nifti1Image",
            "sitk.Image",
        ],
        reference_arr: Union[
            str,
            Path,
            np.ndarray,
            "torch.Tensor",
            "nib.nifti1.Nifti1Image",
            "sitk.Image",
        ],
        result_all: bool = True,
        voxelspacing: tuple[float, ...] | None = None,
        save_group_times: bool | None = None,
        log_times: bool | None = None,
        verbose: bool | None = None,
    ) -> dict[str, PanopticaResult]:
        """Runs the panoptica evaluation pipeline on the given prediction and reference arrays.

        Args:
            prediction_arr (Union[ str, Path, np.ndarray, &quot;torch.Tensor&quot;, &quot;nib.nifti1.Nifti1Image&quot;, &quot;sitk.Image&quot;, ]): Prediction array or file path.
            reference_arr (Union[ str, Path, np.ndarray, &quot;torch.Tensor&quot;, &quot;nib.nifti1.Nifti1Image&quot;, &quot;sitk.Image&quot;, ]): Reference array or file path.
            result_all (bool, optional): If True, will calculate all metrics and return a PanopticaResult object. If False, will only return the metrics that were requested. Defaults to True.
            voxelspacing (tuple[float, ...] | None, optional): Voxel spacing for the evaluation. If None, will use default spacing of (1.0, 1.0, 1.0). Defaults to None.
            save_group_times (bool | None, optional): If None, will use the value set in the constructor. If True, will save the computation time of each sample and put that into the result object. Defaults to None.
            log_times (bool | None, optional): If None, will use the value set in the constructor. If True, will print the times for the different phases of the pipeline. Defaults to None.
            verbose (bool | None, optional): If None, will use the value set in the constructor. If True, will spit out more details than you want. Defaults to None.

        Returns:
            dict[str, PanopticaResult]: A dictionary with group names as keys and PanopticaResult objects as values, containing the evaluation results for each group.
        """
        processing_pair, metadata = self._preprocess_input(
            prediction_arr, reference_arr, voxelspacing
        )

        result_grouped: dict[str, PanopticaResult] = {}
        for group_name, label_group in self.__segmentation_class_groups.items():
            result_grouped[group_name] = self._evaluate_group(
                group_name=group_name,
                label_group=label_group,
                processing_pair=processing_pair,
                decision_threshold=self.__decision_threshold,
                result_all=result_all,
                save_group_times=(
                    self.__save_group_times
                    if save_group_times is None
                    else save_group_times
                ),
                log_times=log_times,
                verbose=verbose,
                **metadata,
            )
        return result_grouped

    @citation_reminder
    @measure_time
    def evaluate_autc(
        self,
        prediction_arr: Union[
            str,
            Path,
            np.ndarray,
            "torch.Tensor",
            "nib.nifti1.Nifti1Image",
            "sitk.Image",
        ],
        reference_arr: Union[
            str,
            Path,
            np.ndarray,
            "torch.Tensor",
            "nib.nifti1.Nifti1Image",
            "sitk.Image",
        ],
        threshold_step_size: float = 0.1,
        decision_threshold_mode: Literal["fixed", "step"] = "step",
        result_all: bool = True,
        voxelspacing: tuple[float, ...] | None = None,
        save_group_times: bool | None = None,
        log_times: bool | None = None,
        verbose: bool | None = None,
    ) -> dict[str, PanopticaAUTCResult]:
        """Runs the panoptica evaluation pipeline on the given prediction and reference arrays and computes the Area Under the Threshold Curve (AUTC) for a range of thresholds.

        Args:
            prediction_arr (Union[ str, Path, np.ndarray, &quot;torch.Tensor&quot;, &quot;nib.nifti1.Nifti1Image&quot;, &quot;sitk.Image&quot;, ]): Prediction array or file path.
            reference_arr (Union[ str, Path, np.ndarray, &quot;torch.Tensor&quot;, &quot;nib.nifti1.Nifti1Image&quot;, &quot;sitk.Image&quot;, ]): Reference array or file path.
            threshold_step_size (float, optional): The step size for the threshold range. Defaults to 0.1.
            decision_threshold_mode (Literal["fixed", "step"], optional): The mode for determining the decision threshold. Defaults to "step" which means the threshold will be the same as for the matching. fixed means it will use the Panoptica_Evaluator's decision_threshold for all thresholds, which can be set in the constructor.
            result_all (bool, optional): If True, will calculate all metrics and return a PanopticaResult object. If False, will only return the metrics that were requested. Defaults to True.
            voxelspacing (tuple[float, ...] | None, optional): Voxel spacing for the evaluation. If None, will use default spacing of (1.0, 1.0, 1.0). Defaults to None.
            save_group_times (bool | None, optional): If None, will use the value set in the constructor. If True, will save the computation time of each sample and put that into the result object. Defaults to None.
            log_times (bool | None, optional): If None, will use the value set in the constructor. If True, will print the times for the different phases of the pipeline. Defaults to None.
            verbose (bool | None, optional): If None, will use the value set in the constructor. If True, will spit out more details than you want. Defaults to None.

        Raises:
            TypeError: if instance matcher is not of type ThresholdBasedMatching
            ValueError: if InputType is MATCHED_INSTANCE or if mode is fixed and no __decision_threshold is set

        Returns:
            dict[str, PanopticaAUTCResult]: A dictionary with group names as keys and PanopticaAUTCResult objects as values, containing the evaluation results for each group.
        """

        if not isinstance(self.__instance_matcher, ThresholdBasedMatching):
            raise TypeError(
                f"evaluate_autc can only be used with ThresholdBasedMatching instance matchers, but got {type(self.__instance_matcher)}"
            )
        if self.__expected_input == InputType.MATCHED_INSTANCE:
            raise ValueError(
                "evaluate_autc cannot be used with already matched instance pairs as input"
            )
        if decision_threshold_mode == "fixed" and self.__decision_threshold is None:
            raise ValueError(
                "decision_threshold must be set in the constructor when using fixed decision_threshold mode for evaluate_autc"
            )

        processing_pair, metadata = self._preprocess_input(
            prediction_arr, reference_arr, voxelspacing
        )

        thresholds = self.generate_thresholds(threshold_step_size)
        result_grouped: dict[str, PanopticaAUTCResult] = {}
        save_group_times = self.__save_group_times if save_group_times is None else save_group_times
        for group_name, label_group in self.__segmentation_class_groups.items():
            if save_group_times:
                start_time = perf_counter()

            prediction_arr_grouped = label_group(processing_pair.prediction_arr)
            reference_arr_grouped = label_group(processing_pair.reference_arr)

            processing_pair_grouped = processing_pair.__class__(prediction_arr=prediction_arr_grouped, reference_arr=reference_arr_grouped)  # type: ignore
            instance_metadata = processing_pair_grouped.get_metadata()
            if label_group.single_instance and not isinstance(
                processing_pair, MatchedInstancePair
            ):
                processing_pair_grouped = MatchedInstancePair(
                    prediction_arr=processing_pair_grouped.prediction_arr,
                    reference_arr=processing_pair_grouped.reference_arr,
                )

            if isinstance(processing_pair_grouped, SemanticPair):
                processing_pair_grouped = _phase_instance_approximation(
                    processing_pair_grouped,
                    None,
                    self.__instance_approximator,
                    instance_metadata,
                    label_group,
                    log_times=self.__log_times if log_times is None else log_times,
                    verbose=self.__verbose if verbose is None else verbose,
                )

            threshold_results: dict[float, PanopticaResult] = {}
            for threshold in thresholds:
                decision_threshold = threshold
                if label_group.single_instance:
                    decision_threshold = 0.0
                elif decision_threshold_mode == "fixed":
                    decision_threshold = self.__decision_threshold

                threshold_results[threshold] = _panoptic_evaluate(
                    input_pair=processing_pair_grouped,
                    edge_case_handler=self.__edge_case_handler,
                    instance_approximator=self.__instance_approximator,
                    instance_matcher=self.__instance_matcher,
                    instance_metrics=self.__eval_metrics,
                    global_metrics=self.__global_metrics,
                    decision_metric=self.__decision_metric,
                    decision_threshold=decision_threshold,
                    matching_threshold=threshold,
                    result_all=result_all,
                    log_times=self.__log_times if log_times is None else log_times,
                    verbose=self.__verbose if verbose is None else verbose,
                    verbose_calc=self.__verbose if verbose is None else verbose,
                    label_group=label_group,
                    **metadata,
                )

            result_grouped[group_name] = PanopticaAUTCResult(
                threshold_results=threshold_results
            )
            if save_group_times:
                duration = perf_counter() - start_time
                result_grouped[group_name].computation_time = duration
        return result_grouped

    def _preprocess_input(
        self,
        prediction_arr: Union[
            str,
            Path,
            np.ndarray,
            "torch.Tensor",
            "nib.nifti1.Nifti1Image",
            "sitk.Image",
        ],
        reference_arr: Union[
            str,
            Path,
            np.ndarray,
            "torch.Tensor",
            "nib.nifti1.Nifti1Image",
            "sitk.Image",
        ],
        voxelspacing: tuple[float, ...] | None = None,
    ) -> tuple[Union[MatchedInstancePair, UnmatchedInstancePair, SemanticPair], dict]:
        """Handles data ingestion, sanity checking, and initial validation."""

        # Sanity check input and convert to numpy arrays
        ((prediction_arr, reference_arr), metadata), _ = (
            sanity_check_and_convert_to_array(prediction_arr, reference_arr)
        )

        if voxelspacing is not None:
            metadata["voxelspacing"] = voxelspacing

        # Take the numpy arrays and convert them to the panoptica internal data structure
        processing_pair = self.__expected_input(prediction_arr, reference_arr)
        assert isinstance(
            processing_pair, self.__expected_input.value
        ), f"input not of expected type {self.__expected_input}"

        # Validate labels
        self.__segmentation_class_groups.has_defined_labels_for(
            processing_pair.prediction_arr, raise_error=True
        )
        self.__segmentation_class_groups.has_defined_labels_for(
            processing_pair.reference_arr, raise_error=True
        )

        return processing_pair, metadata

    @staticmethod
    def generate_thresholds(step_size: float) -> np.ndarray:
        """Return AUTC threshold steps within the inclusive range [step_size, 1]."""
        if not 0 < step_size < 1:
            raise ValueError(
                "step_size must satisfy 0 < step_size < 1 to generate at least two AUTC thresholds"
            )
        thresholds = np.arange(step_size, 1.0 + step_size, step_size)
        thresholds = np.minimum(thresholds, 1.0)
        return np.unique(np.round(thresholds, 5))

    @property
    def segmentation_class_groups_names(self) -> list[str]:
        return self.__segmentation_class_groups.keys()

    @property
    def resulting_metric_keys(self) -> list[str]:
        if self.__resulting_metric_keys is None:
            res = self._get_dummy_result()
            self.__resulting_metric_keys = list(res.to_dict().keys())
        return self.__resulting_metric_keys

    def set_log_group_times(self, should_save: bool):
        self.__save_group_times = should_save

    def _set_instance_approximator(self, instance_approximator: InstanceApproximator):
        self.__instance_approximator = instance_approximator

    def _set_instance_matcher(self, matcher: InstanceMatchingAlgorithm):
        self.__instance_matcher = matcher

    def _get_dummy_result(self) -> PanopticaResult:
        """Helper method to generate a blank evaluation for extracting dynamic metric keys."""
        dummy_input = MatchedInstancePair(
            np.ones((1, 1, 1), dtype=np.uint8), np.ones((1, 1, 1), dtype=np.uint8)
        )
        return self._evaluate_group(
            group_name="",
            label_group=LabelGroup(1, single_instance=False),
            processing_pair=dummy_input,
            decision_threshold=self.__decision_threshold,
            result_all=True,
            voxelspacing=(1.0, 1.0, 1.0),
            save_group_times=False,
            log_times=False,
            verbose=False,
        )

    def get_autc_metric_keys(self, threshold_step_size: float) -> list[str]:
        """Must produce keys in exactly the same order as PanopticaAUTCResult.to_dict()."""
        res = self._get_dummy_result()
        keys = [f"autc_{m}" for m in sorted(res.autc_metrics)]
        
        base_keys = self.resulting_metric_keys
        for t in self.generate_thresholds(threshold_step_size):
            t_str = f"{t:g}"
            for m in base_keys:
                keys.append(f"t{t_str}_{m}")
        return keys

    def _evaluate_group(
        self,
        group_name: str,
        label_group: LabelGroup,
        processing_pair,
        decision_threshold: float | None = None,
        matching_threshold: float | None = None,
        result_all: bool = True,
        verbose: bool | None = None,
        log_times: bool | None = None,
        save_group_times: bool = False,
        **kwargs,
    ) -> PanopticaResult:
        assert isinstance(label_group, LabelGroup)
        if self.__save_group_times or save_group_times:
            start_time = perf_counter()

        prediction_arr_grouped = label_group(processing_pair.prediction_arr)
        reference_arr_grouped = label_group(processing_pair.reference_arr)

        single_instance_mode = label_group.single_instance
        processing_pair_grouped = processing_pair.__class__(prediction_arr=prediction_arr_grouped, reference_arr=reference_arr_grouped)  # type: ignore
        if single_instance_mode and not isinstance(
            processing_pair, MatchedInstancePair
        ):
            processing_pair_grouped = MatchedInstancePair(
                prediction_arr=processing_pair_grouped.prediction_arr,
                reference_arr=processing_pair_grouped.reference_arr,
            )
            decision_threshold = 0.0

        if self.__per_region_evaluation:
            result = _panoptic_evaluate_region_wise(
                input_pair=processing_pair_grouped,
                edge_case_handler=self.__edge_case_handler,
                instance_approximator=self.__instance_approximator,
                instance_matcher=self.__instance_matcher,
                instance_metrics=self.__eval_metrics,
                global_metrics=self.__global_metrics,
                result_all=result_all,
                log_times=self.__log_times if log_times is None else log_times,
                verbose=self.__verbose if verbose is None else verbose,
                verbose_calc=self.__verbose if verbose is None else verbose,
                label_group=label_group,
                **kwargs,
            )
        else:
            result = _panoptic_evaluate(
                input_pair=processing_pair_grouped,
                edge_case_handler=self.__edge_case_handler,
                instance_approximator=self.__instance_approximator,
                instance_matcher=self.__instance_matcher,
                instance_metrics=self.__eval_metrics,
                global_metrics=self.__global_metrics,
                decision_metric=self.__decision_metric,
                decision_threshold=decision_threshold,
                matching_threshold=matching_threshold,
                result_all=result_all,
                log_times=self.__log_times if log_times is None else log_times,
                verbose=self.__verbose if verbose is None else verbose,
                verbose_calc=self.__verbose if verbose is None else verbose,
                label_group=label_group,
                **kwargs,
            )
        if self.__save_group_times or save_group_times:
            duration = perf_counter() - start_time
            result.computation_time = duration
        return result
