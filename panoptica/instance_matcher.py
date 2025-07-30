from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np

from panoptica._functionals import (
    _calc_matching_metric_of_overlapping_labels,
    _calc_matching_metric_of_overlapping_partlabels,
    _map_labels,
)
from panoptica.metrics import Metric
from panoptica.utils.processing_pair import (
    MatchedInstancePair,
    UnmatchedInstancePair,
)
from panoptica.utils.instancelabelmap import InstanceLabelMap
from panoptica.utils.config import SupportsConfig
from panoptica.utils.label_group import LabelGroup, LabelPartGroup


@dataclass
class MatchingContext:
    """Encapsulates context information needed for matching operations."""

    label_group: Optional[LabelGroup] = None
    num_ref_labels: Optional[int] = None
    processing_pair_orig_shape: Optional[Tuple] = None

    @property
    def is_part_group(self) -> bool:
        """Check if this context represents a part group."""
        return isinstance(self.label_group, LabelPartGroup)


class InstanceMatchingAlgorithm(SupportsConfig, metaclass=ABCMeta):
    """
    Abstract base class for instance matching algorithms in panoptic segmentation evaluation.

    Methods:
        _match_instances(self, unmatched_instance_pair: UnmatchedInstancePair, context: MatchingContext = None, **kwargs) -> InstanceLabelMap:
            Abstract method to be implemented by subclasses for instance matching.

        match_instances(self, unmatched_instance_pair: UnmatchedInstancePair, **kwargs) -> MatchedInstancePair:
            Perform instance matching on the given UnmatchedInstancePair.

    Example:
    >>> class CustomInstanceMatcher(InstanceMatchingAlgorithm):
    ...     def _match_instances(self, unmatched_instance_pair: UnmatchedInstancePair, context: MatchingContext = None, **kwargs) -> InstanceLabelMap:
    ...         # Implementation of instance matching algorithm
    ...         pass
    ...
    >>> matcher = CustomInstanceMatcher()
    >>> unmatched_instance_pair = UnmatchedInstancePair(...)
    >>> result = matcher.match_instances(unmatched_instance_pair)
    """

    @abstractmethod
    def _match_instances(
        self,
        unmatched_instance_pair: UnmatchedInstancePair,
        context: Optional[MatchingContext] = None,
        **kwargs,
    ) -> InstanceLabelMap:
        """
        Abstract method to be implemented by subclasses for instance matching.

        Args:
            unmatched_instance_pair (UnmatchedInstancePair): The unmatched instance pair to be matched.
            context (Optional[MatchingContext]): Context information for matching. If None, a default context will be created.
            **kwargs: Additional keyword arguments.

        Returns:
            InstanceLabelMap: The result of the instance matching.
        """
        pass

    def match_instances(
        self,
        unmatched_instance_pair: UnmatchedInstancePair,
        label_group=None,
        num_ref_labels=None,
        processing_pair_orig_shape=None,
        **kwargs,
    ) -> MatchedInstancePair:
        """
        Perform instance matching on the given UnmatchedInstancePair.

        Args:
            unmatched_instance_pair (UnmatchedInstancePair): The unmatched instance pair to be matched.
            label_group: The label group object for this group.
            num_ref_labels: Number of reference labels.
            processing_pair_orig_shape: Original shape of the processing pair.
            **kwargs: Additional keyword arguments.

        Returns:
            MatchedInstancePair: The result of the instance matching.
        """
        # Create context only if any context information is provided
        context = None
        if (
            label_group is not None
            or num_ref_labels is not None
            or processing_pair_orig_shape is not None
        ):
            context = MatchingContext(
                label_group=label_group,
                num_ref_labels=num_ref_labels,
                processing_pair_orig_shape=processing_pair_orig_shape,
            )

        instance_labelmap = self._match_instances(
            unmatched_instance_pair,
            context,
            **kwargs,
        )

        return map_instance_labels(unmatched_instance_pair.copy(), instance_labelmap)

    def _calculate_matching_metric_pairs(
        self,
        unmatched_instance_pair: UnmatchedInstancePair,
        context: Optional[MatchingContext],
        matching_metric: Metric,
    ) -> List[Tuple[float, Tuple[int, int]]]:
        """
        Calculate matching metric pairs based on context.

        Args:
            unmatched_instance_pair: The unmatched instance pair.
            context: The matching context. If None, defaults to non-part group behavior.
            matching_metric: The metric to use for matching.

        Returns:
            List of (matching_score, (ref_label, pred_label)) tuples.
        """
        pred_arr, ref_arr = (
            unmatched_instance_pair.prediction_arr,
            unmatched_instance_pair.reference_arr,
        )
        ref_labels = unmatched_instance_pair.ref_labels

        if context is not None and context.is_part_group:
            return _calc_matching_metric_of_overlapping_partlabels(
                pred_arr,
                ref_arr,
                context.processing_pair_orig_shape,
                context.num_ref_labels,
                matching_metric=matching_metric,
            )
        else:
            return _calc_matching_metric_of_overlapping_labels(
                pred_arr, ref_arr, ref_labels, matching_metric=matching_metric
            )

    def _yaml_repr(cls, node) -> dict:
        raise NotImplementedError(
            f"Tried to get yaml representation of abstract class {cls.__name__}"
        )
        return {}


def map_instance_labels(
    processing_pair: UnmatchedInstancePair, labelmap: InstanceLabelMap
) -> MatchedInstancePair:
    """
    Map instance labels based on the provided labelmap and create a MatchedInstancePair.

    Args:
        processing_pair (UnmatchedInstancePair): The unmatched instance pair containing original labels.
        labelmap (InstanceLabelMap): The instance label map obtained from instance matching.

    Returns:
        MatchedInstancePair: The result of mapping instance labels.
    """
    prediction_arr = processing_pair.prediction_arr
    ref_labels = processing_pair.ref_labels
    pred_labels = processing_pair.pred_labels

    label_counter = int(max(ref_labels) + 1)
    pred_labelmap = labelmap.get_one_to_one_dictionary()

    # assign missed instances to next unused labels sequentially
    missed_pred_labels = [p for p in pred_labels if p not in pred_labelmap]
    for p in missed_pred_labels:
        pred_labelmap[p] = label_counter
        label_counter += 1

    assert np.all([i in pred_labelmap for i in pred_labels])

    # Using the labelmap, actually change the labels in the array here
    prediction_arr_relabeled = _map_labels(prediction_arr, pred_labelmap)

    # Build a MatchedInstancePair out of the newly derived data
    matched_instance_pair = MatchedInstancePair(
        prediction_arr=prediction_arr_relabeled,
        reference_arr=processing_pair.reference_arr,
    )
    return matched_instance_pair


class NaiveThresholdMatching(InstanceMatchingAlgorithm):
    """
    Instance matching algorithm that performs threshold-based matching.

    Attributes:
        matching_metric (Metric): The metric used for matching.
        matching_threshold (float): The threshold for matching instances.
        allow_many_to_one (bool): Whether to allow many-to-one matching.
    """

    def __init__(
        self,
        matching_metric: Metric = Metric.IOU,
        matching_threshold: float = 0.5,
        allow_many_to_one: bool = False,
    ) -> None:
        """
        Initialize the NaiveThresholdMatching instance.

        Args:
            matching_metric (Metric): The metric used for matching.
            matching_threshold (float): The threshold for matching instances.
            allow_many_to_one (bool): Whether to allow many-to-one matching.
        """
        self._allow_many_to_one = allow_many_to_one
        self._matching_metric = matching_metric
        self._matching_threshold = matching_threshold

    def _match_instances(
        self,
        unmatched_instance_pair: UnmatchedInstancePair,
        context: Optional[MatchingContext] = None,
        **kwargs,
    ) -> InstanceLabelMap:
        """
        Perform threshold-based instance matching.

        Args:
            unmatched_instance_pair (UnmatchedInstancePair): The unmatched instance pair to be matched.
            context (Optional[MatchingContext]): The matching context.
            **kwargs: Additional keyword arguments.

        Returns:
            InstanceLabelMap: The result of the instance matching.
        """
        labelmap = InstanceLabelMap()

        mm_pairs = self._calculate_matching_metric_pairs(
            unmatched_instance_pair, context, self._matching_metric
        )

        # Loop through matched instances
        for matching_score, (ref_label, pred_label) in mm_pairs:
            if pred_label in labelmap:
                # skip if prediction label is already matched
                continue
            if (
                labelmap.contains_or(pred_label, ref_label)
                and not self._allow_many_to_one
            ):
                continue

            if self._matching_metric.score_beats_threshold(
                matching_score, self._matching_threshold
            ):
                # Match found, add entry to labelmap
                labelmap.add_labelmap_entry(pred_label, ref_label)

        return labelmap

    @classmethod
    def _yaml_repr(cls, node) -> dict:
        return {
            "matching_metric": node._matching_metric,
            "matching_threshold": node._matching_threshold,
            "allow_many_to_one": node._allow_many_to_one,
        }


class MaxBipartiteMatching(InstanceMatchingAlgorithm):
    """
    Instance matching algorithm that performs optimal one-to-one matching based on maximum bipartite graph matching.

    This implementation maximizes the global matching score between predictions and references.
    """

    def __init__(
        self,
        matching_metric: Metric = Metric.IOU,
        matching_threshold: float = 0.5,
    ) -> None:
        """
        Initialize the MaxBipartiteMatching instance.

        Args:
            matching_metric (Metric): The metric to be used for matching.
            matching_threshold (float): The metric threshold for matching instances.
        """
        self._matching_metric = matching_metric
        self._matching_threshold = matching_threshold

    def _match_instances(
        self,
        unmatched_instance_pair: UnmatchedInstancePair,
        context: Optional[MatchingContext] = None,
        **kwargs,
    ) -> InstanceLabelMap:
        """
        Perform optimal instance matching based on maximum bipartite graph matching.

        Args:
            unmatched_instance_pair (UnmatchedInstancePair): The unmatched instance pair to be matched.
            context (Optional[MatchingContext]): The matching context.
            **kwargs: Additional keyword arguments.

        Returns:
            InstanceLabelMap: The result of the instance matching.
        """
        ref_labels = unmatched_instance_pair.ref_labels
        pred_labels = unmatched_instance_pair.pred_labels
        labelmap = InstanceLabelMap()

        if len(ref_labels) == 0 or len(pred_labels) == 0:
            return labelmap

        mm_pairs = self._calculate_matching_metric_pairs(
            unmatched_instance_pair, context, self._matching_metric
        )

        # Create cost matrix for bipartite matching
        cost_matrix = self._create_cost_matrix(ref_labels, pred_labels, mm_pairs)

        # Apply maximum bipartite graph matching
        labelmap = self._solve_bipartite_matching(cost_matrix, ref_labels, pred_labels)

        return labelmap

    def _create_cost_matrix(
        self,
        ref_labels: List[int],
        pred_labels: List[int],
        mm_pairs: List[Tuple[float, Tuple[int, int]]],
    ) -> np.ndarray:
        """Create cost matrix for bipartite matching."""
        # Create label to index mappings
        ref_label_to_index = {label: i for i, label in enumerate(ref_labels)}
        pred_label_to_index = {label: i for i, label in enumerate(pred_labels)}

        # Initialize cost matrix with high default cost
        small_number = 1e-6
        default_cost = 1.0 + small_number
        cost_matrix = np.ones((len(ref_labels), len(pred_labels))) * default_cost

        # Fill in known costs for overlapping instances
        for matching_score, (ref_label, pred_label) in mm_pairs:
            if not self._matching_metric.score_beats_threshold(
                matching_score, self._matching_threshold
            ):
                continue

            ref_idx = ref_label_to_index[ref_label]
            pred_idx = pred_label_to_index[pred_label]
            cost_matrix[ref_idx, pred_idx] = 1.0 - matching_score

        return cost_matrix

    def _solve_bipartite_matching(
        self, cost_matrix: np.ndarray, ref_labels: List[int], pred_labels: List[int]
    ) -> InstanceLabelMap:
        """Solve the bipartite matching problem and return labelmap."""
        from scipy.optimize import linear_sum_assignment

        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        labelmap = InstanceLabelMap()

        for i, j in zip(row_indices, col_indices):
            if cost_matrix[i, j] < 1.0:  # Valid match
                ref_label = ref_labels[i]
                pred_label = pred_labels[j]
                labelmap.add_labelmap_entry(int(pred_label), int(ref_label))

        return labelmap

    @classmethod
    def _yaml_repr(cls, node) -> dict:
        return {
            "matching_metric": node._matching_metric,
            "matching_threshold": node._matching_threshold,
        }


class MaximizeMergeMatching(InstanceMatchingAlgorithm):
    """
    Instance matching algorithm that performs many-to-one matching based on metric. Will merge if combined instance metric is greater than individual one. Only matches if at least a single instance exceeds the threshold.

    Attributes:
        matching_metric (Metric): The metric to be used for matching.
        matching_threshold (float): The threshold for matching instances.
    """

    def __init__(
        self,
        matching_metric: Metric = Metric.IOU,
        matching_threshold: float = 0.5,
    ) -> None:
        """
        Initialize the MaximizeMergeMatching instance.

        Args:
            matching_metric (Metric): The metric to be used for matching.
            matching_threshold (float): The threshold for matching instances.
        """
        self._matching_metric = matching_metric
        self._matching_threshold = matching_threshold

    def _match_instances(
        self,
        unmatched_instance_pair: UnmatchedInstancePair,
        context: Optional[MatchingContext] = None,
        **kwargs,
    ) -> InstanceLabelMap:
        """
        Perform many-to-one instance matching based on metric values.

        Args:
            unmatched_instance_pair (UnmatchedInstancePair): The unmatched instance pair to be matched.
            context (Optional[MatchingContext]): The matching context.
            **kwargs: Additional keyword arguments.

        Returns:
            InstanceLabelMap: The result of the instance matching.
        """
        ref_labels = unmatched_instance_pair.ref_labels
        # pred_labels = unmatched_instance_pair._pred_labels

        # Initialize variables for True Positives (tp) and False Positives (fp)
        labelmap = InstanceLabelMap()
        score_ref: dict[int, float] = {}

        pred_arr, ref_arr = (
            unmatched_instance_pair.prediction_arr,
            unmatched_instance_pair.reference_arr,
        )
        mm_pairs = _calc_matching_metric_of_overlapping_labels(
            prediction_arr=pred_arr,
            reference_arr=ref_arr,
            ref_labels=ref_labels,
            matching_metric=self._matching_metric,
        )

        # Loop through matched instances to compute PQ components
        for matching_score, (ref_label, pred_label) in mm_pairs:
            if labelmap.contains_pred(pred_label=pred_label):
                # skip if prediction label is already matched
                continue
            if labelmap.contains_ref(ref_label):
                pred_labels_ = labelmap.get_pred_labels_matched_to_ref(ref_label)
                new_score = self.new_combination_score(
                    pred_labels_, pred_label, ref_label, unmatched_instance_pair
                )
                if new_score > score_ref[ref_label]:
                    labelmap.add_labelmap_entry(pred_label, ref_label)
                    score_ref[ref_label] = new_score
            elif self._matching_metric.score_beats_threshold(
                matching_score, self._matching_threshold
            ):
                # Match found, increment true positive count and collect IoU and Dice values
                labelmap.add_labelmap_entry(pred_label, ref_label)
                score_ref[ref_label] = matching_score
                # map label ref_idx to pred_idx
        return labelmap

    def new_combination_score(
        self,
        pred_labels: list[int],
        new_pred_label: int,
        ref_label: int,
        unmatched_instance_pair: UnmatchedInstancePair,
    ):
        pred_labels.append(new_pred_label)
        score = self._matching_metric(
            unmatched_instance_pair.reference_arr,
            prediction_arr=unmatched_instance_pair.prediction_arr,
            ref_instance_idx=ref_label,
            pred_instance_idx=pred_labels,
        )
        return score

    @classmethod
    def _yaml_repr(cls, node) -> dict:
        return {
            "matching_metric": node._matching_metric,
            "matching_threshold": node._matching_threshold,
        }
