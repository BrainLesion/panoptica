from abc import abstractmethod, ABC
from panoptica.utils.datatypes import UnmatchedInstancePair, MatchedInstancePair, Instance_Label_Map
import numpy as np
from panoptica._functionals import _map_labels, _calc_iou_matrix
from scipy.optimize import linear_sum_assignment


class InstanceMatchingAlgorithm(ABC):
    @abstractmethod
    def _match_instances(self, unmatched_instance_pair: UnmatchedInstancePair, **kwargs) -> Instance_Label_Map:
        pass

    def match_instances(self, unmatched_instance_pair: UnmatchedInstancePair, **kwargs) -> MatchedInstancePair:
        instance_labelmap = self._match_instances(unmatched_instance_pair, **kwargs)
        print("instance_labelmap", instance_labelmap)
        return map_instance_labels(unmatched_instance_pair.copy(), instance_labelmap)


class NaiveOneToOneMatching(InstanceMatchingAlgorithm):
    def __init__(self, iou_threshold: float = 0.5) -> None:
        assert iou_threshold >= 0.5, "NaiveOneToOneMatching: iou_threshold lower than 0.5 doesnt work!"
        assert iou_threshold < 1.0, "NaiveOneToOneMatching: iou_threshold greater than or equal to 1.0 doesnt work!"
        self.iou_threshold = iou_threshold

    def _match_instances(self, unmatched_instance_pair: UnmatchedInstancePair, **kwargs) -> Instance_Label_Map:
        ref_labels = unmatched_instance_pair.ref_labels
        pred_labels = unmatched_instance_pair.pred_labels
        iou_matrix = _calc_iou_matrix(
            unmatched_instance_pair.prediction_arr,
            unmatched_instance_pair.reference_arr,
            ref_labels,
            pred_labels,
        )
        # Use linear_sum_assignment to find the best matches
        ref_indices, pred_indices = linear_sum_assignment(-iou_matrix)

        # Initialize variables for True Positives (tp) and False Positives (fp)
        tp, iou_list = 0, []
        labelmap: Instance_Label_Map = []

        # Loop through matched instances to compute PQ components
        for ref_idx, pred_idx in zip(ref_indices, pred_indices):
            iou = iou_matrix[ref_idx][pred_idx]
            if iou >= self.iou_threshold:
                # Match found, increment true positive count and collect IoU and Dice values
                tp += 1
                iou_list.append(iou)
                labelmap.append(([ref_labels[ref_idx]], [pred_labels[pred_idx]]))
                # map label ref_idx to pred_idx
        return labelmap


def map_instance_labels(processing_pair: UnmatchedInstancePair, labelmap: Instance_Label_Map) -> MatchedInstancePair:
    prediction_arr, reference_arr = processing_pair.prediction_arr, processing_pair.reference_arr

    ref_labels = processing_pair.ref_labels
    pred_labels = processing_pair.pred_labels

    missed_ref_labels = []
    missed_pred_labels = []

    pred_labelmap = {}
    ref_labelmap = {}
    label_counter = 1
    # Go over instance labelmap and assign the matched instance sequentially
    for refs, preds in labelmap:
        for r, p in zip(refs, preds):
            ref_labelmap[r] = label_counter
            pred_labelmap[p] = label_counter
        label_counter += 1
    n_matched_instances = label_counter - 1
    # assign missed instances to next unused labels sequentially
    for r in ref_labels:
        if r not in ref_labelmap:
            ref_labelmap[r] = label_counter
            label_counter += 1
            missed_ref_labels.append(r)
    for p in pred_labels:
        if p not in pred_labelmap:
            pred_labelmap[p] = label_counter
            label_counter += 1
            missed_ref_labels.append(p)

    # Using the labelmap, actually change the labels in the array here
    prediction_arr_relabeled = _map_labels(prediction_arr, pred_labelmap)
    reference_arr_relabeled = _map_labels(reference_arr, ref_labelmap)

    # Build a MatchedInstancePair out of the newly derived data
    matched_instance_pair = MatchedInstancePair(
        prediction_arr=prediction_arr_relabeled,
        reference_arr=reference_arr_relabeled,
        missed_reference_labels=missed_ref_labels,
        missed_prediction_labels=missed_pred_labels,
        n_prediction_instance=processing_pair.n_prediction_instance,
        n_reference_instance=processing_pair.n_reference_instance,
        n_matched_instances=n_matched_instances,
    )
    return matched_instance_pair


if __name__ == "__main__":
    a = np.array([0, 1, 2, 3, 4, 5], dtype=np.uint16)
    b = np.array([0, 1, 2, 6, 3, 7], dtype=np.uint16)
    MatchedInstancePair(
        reference_arr=a,
        prediction_arr=b,
        missed_reference_labels=[4, 5],
        missed_prediction_labels=[6, 7],
        n_matched_instances=3,
        n_prediction_instance=5,
        n_reference_instance=5,
    )
