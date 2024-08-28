from abc import ABCMeta, abstractmethod

import numpy as np

from panoptica._functionals import (
    _calc_matching_metric_of_overlapping_labels,
    _map_labels,
)
from panoptica.metrics import Metric
from panoptica.utils.processing_pair import (
    MatchedInstancePair,
    UnmatchedInstancePair,
)
from panoptica.utils.instancelabelmap import InstanceLabelMap
from panoptica.utils.config import SupportsConfig


class InstanceMatchingAlgorithm(SupportsConfig, metaclass=ABCMeta):
    """
    Abstract base class for instance matching algorithms in panoptic segmentation evaluation.

    Attributes:
        None

    Methods:
        _match_instances(self, unmatched_instance_pair: UnmatchedInstancePair, **kwargs) -> Instance_Label_Map:
            Abstract method to be implemented by subclasses for instance matching.

        match_instances(self, unmatched_instance_pair: UnmatchedInstancePair, **kwargs) -> MatchedInstancePair:
            Perform instance matching on the given UnmatchedInstancePair.

    Example:
    >>> class CustomInstanceMatcher(InstanceMatchingAlgorithm):
    ...     def _match_instances(self, unmatched_instance_pair: UnmatchedInstancePair, **kwargs) -> Instance_Label_Map:
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
        **kwargs,
    ) -> InstanceLabelMap:
        """
        Abstract method to be implemented by subclasses for instance matching.

        Args:
            unmatched_instance_pair (UnmatchedInstancePair): The unmatched instance pair to be matched.
            **kwargs: Additional keyword arguments.

        Returns:
            Instance_Label_Map: The result of the instance matching.
        """
        pass

    def match_instances(
        self,
        unmatched_instance_pair: UnmatchedInstancePair,
        **kwargs,
    ) -> MatchedInstancePair:
        """
        Perform instance matching on the given UnmatchedInstancePair.

        Args:
            unmatched_instance_pair (UnmatchedInstancePair): The unmatched instance pair to be matched.
            **kwargs: Additional keyword arguments.

        Returns:
            MatchedInstancePair: The result of the instance matching.
        """
        instance_labelmap = self._match_instances(
            unmatched_instance_pair,
            **kwargs,
        )
        # print("instance_labelmap:", instance_labelmap)
        return map_instance_labels(unmatched_instance_pair.copy(), instance_labelmap)

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
        labelmap (Instance_Label_Map): The instance label map obtained from instance matching.

    Returns:
        MatchedInstancePair: The result of mapping instance labels.

    Example:
    >>> unmatched_instance_pair = UnmatchedInstancePair(...)
    >>> labelmap = [([1, 2], [3, 4]), ([5], [6])]
    >>> result = map_instance_labels(unmatched_instance_pair, labelmap)
    """
    prediction_arr = processing_pair.prediction_arr

    ref_labels = processing_pair.ref_labels
    pred_labels = processing_pair.pred_labels

    ref_matched_labels = []
    label_counter = int(max(ref_labels) + 1)

    pred_labelmap = labelmap.get_one_to_one_dictionary()
    ref_matched_labels = list([r for r in ref_labels if r in pred_labelmap.values()])

    # assign missed instances to next unused labels sequentially
    missed_ref_labels = list([r for r in ref_labels if r not in ref_matched_labels])
    missed_pred_labels = list([p for p in pred_labels if p not in pred_labelmap])
    for p in missed_pred_labels:
        pred_labelmap[p] = label_counter
        label_counter += 1

    assert np.all([i in pred_labelmap for i in pred_labels])

    # Using the labelmap, actually change the labels in the array here
    prediction_arr_relabeled = _map_labels(prediction_arr, pred_labelmap)  # type:ignore

    # Build a MatchedInstancePair out of the newly derived data
    matched_instance_pair = MatchedInstancePair(
        prediction_arr=prediction_arr_relabeled,
        reference_arr=processing_pair._reference_arr,
    )
    return matched_instance_pair


class NaiveThresholdMatching(InstanceMatchingAlgorithm):
    """
    Instance matching algorithm that performs one-to-one matching based on IoU values.

    Attributes:
        iou_threshold (float): The IoU threshold for matching instances.

    Methods:
        __init__(self, iou_threshold: float = 0.5) -> None:
            Initialize the NaiveOneToOneMatching instance.
        _match_instances(self, unmatched_instance_pair: UnmatchedInstancePair, **kwargs) -> Instance_Label_Map:
            Perform one-to-one instance matching based on IoU values.

    Raises:
        AssertionError: If the specified IoU threshold is not within the valid range.

    Example:
    >>> matcher = NaiveOneToOneMatching(iou_threshold=0.6)
    >>> unmatched_instance_pair = UnmatchedInstancePair(...)
    >>> result = matcher.match_instances(unmatched_instance_pair)
    """

    def __init__(
        self,
        matching_metric: Metric = Metric.IOU,
        matching_threshold: float = 0.5,
        allow_many_to_one: bool = False,
    ) -> None:
        """
        Initialize the NaiveOneToOneMatching instance.

        Args:
            iou_threshold (float, optional): The IoU threshold for matching instances. Defaults to 0.5.

        Raises:
            AssertionError: If the specified IoU threshold is not within the valid range.
        """
        self._allow_many_to_one = allow_many_to_one
        self._matching_metric = matching_metric
        self._matching_threshold = matching_threshold

    def _match_instances(
        self,
        unmatched_instance_pair: UnmatchedInstancePair,
        **kwargs,
    ) -> InstanceLabelMap:
        """
        Perform one-to-one instance matching based on IoU values.

        Args:
            unmatched_instance_pair (UnmatchedInstancePair): The unmatched instance pair to be matched.
            **kwargs: Additional keyword arguments.

        Returns:
            Instance_Label_Map: The result of the instance matching.
        """
        ref_labels = unmatched_instance_pair.ref_labels

        # Initialize variables for True Positives (tp) and False Positives (fp)
        labelmap = InstanceLabelMap()

        pred_arr, ref_arr = (
            unmatched_instance_pair.prediction_arr,
            unmatched_instance_pair.reference_arr,
        )
        mm_pairs = _calc_matching_metric_of_overlapping_labels(
            pred_arr, ref_arr, ref_labels, matching_metric=self._matching_metric
        )

        # Loop through matched instances to compute PQ components
        for matching_score, (ref_label, pred_label) in mm_pairs:
            if (
                labelmap.contains_or(pred_label, ref_label)
                and not self._allow_many_to_one
            ):
                continue  # -> doesnt make speed difference
            # TODO always go in here, but add the matching score to the pair (so evaluation over multiple thresholds becomes easy)
            if self._matching_metric.score_beats_threshold(
                matching_score, self._matching_threshold
            ):
                # Match found, increment true positive count and collect IoU and Dice values
                labelmap.add_labelmap_entry(pred_label, ref_label)
                # map label ref_idx to pred_idx
        return labelmap

    @classmethod
    def _yaml_repr(cls, node) -> dict:
        return {
            "matching_metric": node._matching_metric,
            "matching_threshold": node._matching_threshold,
            "allow_many_to_one": node._allow_many_to_one,
        }


class MaximizeMergeMatching(InstanceMatchingAlgorithm):
    """
    Instance matching algorithm that performs many-to-one matching based on metric. Will merge if combined instance metric is greater than individual one. Only matches if at least a single instance exceeds the threshold


    Methods:
        _match_instances(self, unmatched_instance_pair: UnmatchedInstancePair, **kwargs) -> Instance_Label_Map:

    Raises:
        AssertionError: If the specified IoU threshold is not within the valid range.
    """

    def __init__(
        self,
        matching_metric: Metric = Metric.IOU,
        matching_threshold: float = 0.5,
    ) -> None:
        """
        Initialize the MaximizeMergeMatching instance.

        Args:
            matching_metric (_MatchingMetric): The metric to be used for matching.
            matching_threshold (float, optional): The metric threshold for matching instances. Defaults to 0.5.

        Raises:
            AssertionError: If the specified IoU threshold is not within the valid range.
        """
        self._matching_metric = matching_metric
        self._matching_threshold = matching_threshold

    def _match_instances(
        self,
        unmatched_instance_pair: UnmatchedInstancePair,
        **kwargs,
    ) -> InstanceLabelMap:
        """
        Perform one-to-one instance matching based on IoU values.

        Args:
            unmatched_instance_pair (UnmatchedInstancePair): The unmatched instance pair to be matched.
            **kwargs: Additional keyword arguments.

        Returns:
            Instance_Label_Map: The result of the instance matching.
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


class MatchUntilConvergenceMatching(InstanceMatchingAlgorithm):
    # Match like the naive matcher (so each to their best reference) and then again and again until no overlapping labels are left
    pass


class DesperateMarriageMatching(InstanceMatchingAlgorithm):
    # Match as many predictions to references as possible, doesn't need threshold
    # Option for many-to-one or one-to-one
    # https://github.com/koseii2122/The-Stable-Matching-Algorithm
    pass
