from abc import ABC, abstractmethod

import numpy as np

from panoptica._functionals import _calc_iou_matrix, _map_labels, _calc_iou_of_overlapping_labels
from panoptica.utils.datatypes import (
    InstanceLabelMap,
    MatchedInstancePair,
    UnmatchedInstancePair,
)
from panoptica.timing import measure_time
from scipy.optimize import linear_sum_assignment


class InstanceMatchingAlgorithm(ABC):
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
    def _match_instances(self, unmatched_instance_pair: UnmatchedInstancePair, **kwargs) -> InstanceLabelMap:
        """
        Abstract method to be implemented by subclasses for instance matching.

        Args:
            unmatched_instance_pair (UnmatchedInstancePair): The unmatched instance pair to be matched.
            **kwargs: Additional keyword arguments.

        Returns:
            Instance_Label_Map: The result of the instance matching.
        """
        pass

    def match_instances(self, unmatched_instance_pair: UnmatchedInstancePair, **kwargs) -> MatchedInstancePair:
        """
        Perform instance matching on the given UnmatchedInstancePair.

        Args:
            unmatched_instance_pair (UnmatchedInstancePair): The unmatched instance pair to be matched.
            **kwargs: Additional keyword arguments.

        Returns:
            MatchedInstancePair: The result of the instance matching.
        """
        instance_labelmap = self._match_instances(unmatched_instance_pair, **kwargs)
        # print("instance_labelmap:", instance_labelmap)
        return map_instance_labels(unmatched_instance_pair.copy(), instance_labelmap)


def map_instance_labels(processing_pair: UnmatchedInstancePair, labelmap: InstanceLabelMap) -> MatchedInstancePair:
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
    prediction_arr = processing_pair._prediction_arr

    ref_labels = processing_pair._ref_labels
    pred_labels = processing_pair._pred_labels

    ref_matched_labels = []
    label_counter = int(max(ref_labels) + 1)

    pred_labelmap = labelmap.get_one_to_one_dictionary()
    ref_matched_labels = list([r for r in ref_labels if r in pred_labelmap.values()])

    n_matched_instances = len(ref_matched_labels)

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
        missed_reference_labels=missed_ref_labels,
        missed_prediction_labels=missed_pred_labels,
        n_prediction_instance=processing_pair.n_prediction_instance,
        n_reference_instance=processing_pair.n_reference_instance,
        n_matched_instances=n_matched_instances,
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

    def __init__(self, iou_threshold: float = 0.5, allow_many_to_one: bool = False) -> None:
        """
        Initialize the NaiveOneToOneMatching instance.

        Args:
            iou_threshold (float, optional): The IoU threshold for matching instances. Defaults to 0.5.

        Raises:
            AssertionError: If the specified IoU threshold is not within the valid range.
        """
        self.iou_threshold = iou_threshold
        self.allow_many_to_one = allow_many_to_one

    def _match_instances(self, unmatched_instance_pair: UnmatchedInstancePair, **kwargs) -> InstanceLabelMap:
        """
        Perform one-to-one instance matching based on IoU values.

        Args:
            unmatched_instance_pair (UnmatchedInstancePair): The unmatched instance pair to be matched.
            **kwargs: Additional keyword arguments.

        Returns:
            Instance_Label_Map: The result of the instance matching.
        """
        ref_labels = unmatched_instance_pair._ref_labels
        pred_labels = unmatched_instance_pair._pred_labels

        # Initialize variables for True Positives (tp) and False Positives (fp)
        labelmap = InstanceLabelMap()

        pred_arr, ref_arr = unmatched_instance_pair._prediction_arr, unmatched_instance_pair._reference_arr
        iou_pairs = _calc_iou_of_overlapping_labels(pred_arr, ref_arr, ref_labels, pred_labels)

        # Loop through matched instances to compute PQ components
        for iou, (ref_label, pred_label) in iou_pairs:
            if labelmap.contains_or(pred_label, ref_label) and not self.allow_many_to_one:
                continue  # -> doesnt make speed difference
            if iou >= 0.5:
                # Match found, increment true positive count and collect IoU and Dice values
                labelmap.add_labelmap_entry(pred_label, ref_label)
                # map label ref_idx to pred_idx
        return labelmap


class MaximizeMergeMatching(InstanceMatchingAlgorithm):
    """
    Instance matching algorithm that performs many-to-one matching based on IoU values. Will merge if combined instance IOU is greater than individual one

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

    def __init__(self, iou_threshold: float = 0.5, allow_many_to_one: bool = False) -> None:
        """
        Initialize the NaiveOneToOneMatching instance.

        Args:
            iou_threshold (float, optional): The IoU threshold for matching instances. Defaults to 0.5.

        Raises:
            AssertionError: If the specified IoU threshold is not within the valid range.
        """
        self.iou_threshold = iou_threshold
        self.allow_many_to_one = allow_many_to_one

    def _match_instances(self, unmatched_instance_pair: UnmatchedInstancePair, **kwargs) -> InstanceLabelMap:
        """
        Perform one-to-one instance matching based on IoU values.

        Args:
            unmatched_instance_pair (UnmatchedInstancePair): The unmatched instance pair to be matched.
            **kwargs: Additional keyword arguments.

        Returns:
            Instance_Label_Map: The result of the instance matching.
        """
        ref_labels = unmatched_instance_pair._ref_labels
        pred_labels = unmatched_instance_pair._pred_labels

        # Initialize variables for True Positives (tp) and False Positives (fp)
        labelmap = InstanceLabelMap()

        pred_arr, ref_arr = unmatched_instance_pair._prediction_arr, unmatched_instance_pair._reference_arr
        iou_pairs = _calc_iou_of_overlapping_labels(pred_arr, ref_arr, ref_labels, pred_labels)

        # Loop through matched instances to compute PQ components
        for iou, (ref_label, pred_label) in iou_pairs:
            if labelmap.contains_or(None, ref_label):
                continue  # -> doesnt make speed difference
            if iou >= 0.5:
                # Match found, increment true positive count and collect IoU and Dice values
                labelmap.add_labelmap_entry(pred_label, ref_label)
                # map label ref_idx to pred_idx
        return labelmap
