from abc import ABC, abstractmethod

import numpy as np

from panoptica._functionals import _calc_iou_matrix, _map_labels
from panoptica.utils.datatypes import (
    InstanceLabelMap,
    MatchedInstancePair,
    UnmatchedInstancePair,
)


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


class NaiveOneToOneMatching(InstanceMatchingAlgorithm):
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

    def __init__(self, iou_threshold: float = 0.5) -> None:
        """
        Initialize the NaiveOneToOneMatching instance.

        Args:
            iou_threshold (float, optional): The IoU threshold for matching instances. Defaults to 0.5.

        Raises:
            AssertionError: If the specified IoU threshold is not within the valid range.
        """
        assert iou_threshold >= 0.5, "NaiveOneToOneMatching: iou_threshold lower than 0.5 doesnt work!"
        assert iou_threshold < 1.0, "NaiveOneToOneMatching: iou_threshold greater than or equal to 1.0 doesnt work!"
        self.iou_threshold = iou_threshold

    def _match_instances(self, unmatched_instance_pair: UnmatchedInstancePair, **kwargs) -> InstanceLabelMap:
        """
        Perform one-to-one instance matching based on IoU values.

        Args:
            unmatched_instance_pair (UnmatchedInstancePair): The unmatched instance pair to be matched.
            **kwargs: Additional keyword arguments.

        Returns:
            Instance_Label_Map: The result of the instance matching.
        """
        ref_labels = unmatched_instance_pair.ref_labels
        pred_labels = unmatched_instance_pair.pred_labels
        iou_matrix = _calc_iou_matrix(
            unmatched_instance_pair.prediction_arr.flatten(),
            unmatched_instance_pair.reference_arr.flatten(),
            ref_labels,
            pred_labels,
        )
        # Use linear_sum_assignment to find the best matches
        # ref_indices, pred_indices = linear_sum_assignment(-iou_matrix)

        # Initialize variables for True Positives (tp) and False Positives (fp)
        labelmap = InstanceLabelMap()

        pairs = [(r, p) for r in range(len(ref_labels)) for p in range(len(pred_labels))]

        # Loop through matched instances to compute PQ components
        for ref_idx, pred_idx in pairs:
            iou = iou_matrix[ref_idx][pred_idx]
            if iou >= self.iou_threshold:
                # Match found, increment true positive count and collect IoU and Dice values
                labelmap.add_labelmap_entry(pred_labels[pred_idx], ref_labels[ref_idx])
                # map label ref_idx to pred_idx
        return labelmap


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
    prediction_arr = processing_pair.prediction_arr

    ref_labels = processing_pair.ref_labels
    pred_labels = processing_pair.pred_labels

    ref_matched_labels = []
    label_counter = max(ref_labels) + 1

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
        reference_arr=processing_pair.reference_arr,
        missed_reference_labels=missed_ref_labels,
        missed_prediction_labels=missed_pred_labels,
        n_prediction_instance=processing_pair.n_prediction_instance,
        n_reference_instance=processing_pair.n_reference_instance,
        n_matched_instances=n_matched_instances,
    )
    return matched_instance_pair
