from typing import TYPE_CHECKING
from multiprocessing import Pool

import numpy as np
import math
from panoptica.utils.constants import CCABackend
from panoptica.utils.numpy_utils import _get_bbox_nd

if TYPE_CHECKING:
    from panoptica.metrics import Metric


def _calc_overlapping_labels(
    prediction_arr: np.ndarray,
    reference_arr: np.ndarray,
    ref_labels: tuple[int, ...],
) -> list[tuple[int, int]]:
    """Calculates the pairs of labels that are overlapping in at least one voxel (fast)

    Args:
        prediction_arr (np.ndarray): Numpy array containing the prediction labels.
        reference_arr (np.ndarray): Numpy array containing the reference labels.
        ref_labels (list[int]): List of unique reference labels.

    Returns:
        _type_: _description_
    """
    overlap_arr = prediction_arr.astype(np.uint32)
    max_ref = max(ref_labels) + 1
    overlap_arr = (overlap_arr * max_ref) + reference_arr
    overlap_arr[reference_arr == 0] = 0
    # overlapping_indices = [(i % (max_ref), i // (max_ref)) for i in np.unique(overlap_arr) if i > max_ref]
    # instance_pairs = [(reference_arr, prediction_arr, i, j) for i, j in overlapping_indices]

    # (ref, pred)
    return [
        (int(i % (max_ref)), int(i // (max_ref)))
        for i in np.unique(overlap_arr)
        if i > max_ref
    ]


def _calc_matching_metric_of_overlapping_labels(
    prediction_arr: np.ndarray,
    reference_arr: np.ndarray,
    ref_labels: tuple[int, ...],
    matching_metric: "Metric",
) -> list[tuple[float, tuple[int, int]]]:
    """Calculates the MatchingMetric for all overlapping labels (fast!)

    Args:
        prediction_arr (np.ndarray): Numpy array containing the prediction labels.
        reference_arr (np.ndarray): Numpy array containing the reference labels.
        ref_labels (list[int]): List of unique reference labels.

    Returns:
        list[tuple[float, tuple[int, int]]]: List of pairs in style: (iou, (ref_label, pred_label))
    """
    overlapping_labels = _calc_overlapping_labels(
        prediction_arr=prediction_arr,
        reference_arr=reference_arr,
        ref_labels=ref_labels,
    )
    # instance_pairs = [
    #    (reference_arr, prediction_arr, i[0], i[1])
    #    for i in overlapping_labels
    # ]
    ## with Pool() as pool:
    ##    mm_values = pool.starmap(matching_metric.value, instance_pairs)
    # mm_values = [matching_metric.value(*i) for i in instance_pairs]
    # mm_pairs = [(i, (instance_pairs[idx][2], instance_pairs[idx][3])) for idx, i in enumerate(mm_values)]
    mm_pairs = [
        (matching_metric.value(reference_arr, prediction_arr, i[0], i[1]), (i[0], i[1]))
        for i in overlapping_labels
    ]

    mm_pairs = sorted(
        mm_pairs, key=lambda x: x[0], reverse=not matching_metric.decreasing
    )

    return mm_pairs


def _map_labels(
    arr: np.ndarray,
    label_map: dict[np.integer, np.integer],
) -> np.ndarray:
    """
    Maps labels in the given array according to the label_map dictionary.

    Args:
        label_map (dict): A dictionary that maps the original label values (str or int) to the new label values (int).

    Returns:
        np.ndarray: Returns a copy of the remapped array
    """

    k = np.array(list(label_map.keys()), dtype=arr.dtype)
    v = np.array(list(label_map.values()), dtype=arr.dtype)

    max_value = max(arr.max(), max(k), max(v)) + 1

    mapping_ar = np.arange(max_value, dtype=arr.dtype)
    mapping_ar[k] = v
    return mapping_ar[arr]


def _connected_components(
    array: np.ndarray,
    cca_backend: CCABackend,
) -> tuple[np.ndarray, int]:
    """
    Label connected components in a binary array using a specified connected components algorithm.

    Args:
        array (np.ndarray): Binary array containing connected components.
        cca_backend (CCABackend): Enum indicating the connected components algorithm backend (CCABackend.cc3d or CCABackend.scipy).

    Returns:
        tuple[np.ndarray, int]: A tuple containing the labeled array and the number of connected components.

    Raises:
        NotImplementedError: If the specified connected components algorithm backend is not implemented.

    Example:
    >>> _connected_components(np.array([[1, 0, 1], [0, 1, 1], [1, 0, 0]]), CCABackend.scipy)
    (array([[1, 0, 2], [0, 3, 3], [4, 0, 0]]), 4)
    """
    if cca_backend == CCABackend.cc3d:
        import cc3d

        cc_arr, n_instances = cc3d.connected_components(array, return_N=True)
    elif cca_backend == CCABackend.scipy:
        from scipy.ndimage import label

        cc_arr, n_instances = label(array)
    else:
        raise NotImplementedError(cca_backend)

    return cc_arr, n_instances


def _get_paired_crop(
    prediction_arr: np.ndarray,
    reference_arr: np.ndarray,
    px_pad: int = 2,
):
    """Calculates a bounding box based on paired prediction and reference arrays.

    This function combines the prediction and reference arrays, checks if they are identical,
    and computes a bounding box around the non-zero regions. If both arrays are completely zero,
    a small value is added to ensure the bounding box is valid.

    Args:
        prediction_arr (np.ndarray): The predicted segmentation array.
        reference_arr (np.ndarray): The ground truth segmentation array.
        px_pad (int, optional): Padding to apply around the bounding box. Defaults to 2.

    Returns:
        np.ndarray: The bounding box coordinates around the combined non-zero regions.

    Raises:
        AssertionError: If the prediction and reference arrays do not have the same shape.
    """
    assert prediction_arr.shape == reference_arr.shape

    combined = prediction_arr + reference_arr
    if combined.sum() == 0:
        combined += 1
    return _get_bbox_nd(combined, px_dist=px_pad)


def _round_to_n(value: float | int, n_significant_digits: int = 2):
    """Rounds a number to a specified number of significant digits.

    This function rounds the given value to the specified number of significant digits.
    If the value is zero, it is returned unchanged.

    Args:
        value (float | int): The number to be rounded.
        n_significant_digits (int, optional): The number of significant digits to round to.
            Defaults to 2.

    Returns:
        float: The rounded value.
    """
    return (
        value
        if value == 0
        else round(
            value, -int(math.floor(math.log10(abs(value)))) + (n_significant_digits - 1)
        )
    )

def _calc_matching_metric_of_overlapping_partlabels(
    prediction_arr: np.ndarray,
    reference_arr: np.ndarray,
    processing_pair_orig_shape: tuple[int, ...],
    ref_labels: tuple[int, ...],
    matching_metric: "Metric",
) -> list[tuple[float, tuple[int, int]]]:
    """Calculates a combined matching metric for thing+part overlapping labels.

    This function calculates a combined score between the panoptic (thing) labels
    and their corresponding parts. The score is the mean of the thing instance
    score and the mean of all part instance scores.

    Args:
        prediction_arr (np.ndarray): Numpy array containing the prediction labels.
        reference_arr (np.ndarray): Numpy array containing the reference labels.
        ref_label (int): Reference panoptic/thing label.
        pred_label (int): Prediction panoptic/thing label.
        part_labels (list[int]): List of part labels to consider.
        matching_metric (Metric): The metric to use for evaluation.

    Returns:
        float: The combined matching score for the thing+parts.
    """
    
    ref_labels = list(ref_labels)    
    prediction_arr = _get_orig_onehotcc_structure(prediction_arr, len(ref_labels), processing_pair_orig_shape)
    reference_arr = _get_orig_onehotcc_structure(reference_arr, len(ref_labels), processing_pair_orig_shape)

    #1 Perform matching based on things. The way the LabelPartGroup is defined, there will always be only one thing per class and it will be the first one.

    overlapping_labels = _calc_overlapping_labels(
        prediction_arr=prediction_arr[0],
        reference_arr=reference_arr[0],
        ref_labels=[max(prediction_arr[0].max(), reference_arr[0].max())],
    )

    #! Why? Think of a human. Body + limbs (limbs is parts). You want to match with the whole body including the linbs right? 

    mm_pairs = [
        (matching_metric.value(reference_arr[0], prediction_arr[0], i[0], i[1]), (i[0], i[1]))
        for i in overlapping_labels
    ]

    thing_pairs = [
        (matching_metric.value(reference_arr[0], prediction_arr[0], i[0], i[1]), (i[0], i[1]))
        for i in overlapping_labels
    ]

    sorted_thing_pairs = sorted(
        thing_pairs, key=lambda x: x[0], reverse=not matching_metric.decreasing
    )

    #2 When calculating the metric, we need to take into account the part labels



    updated_thing_pairs = sorted_thing_pairs.copy()

    #? loop through the overlapping labels and pred, ref pairs
    for i, j in overlapping_labels:


        # isolate the matched components for the label in pred and ref
        matched_ref_component = (reference_arr[0] == i)
        matched_pred_component = (prediction_arr[0] == j)

        #? Isolate the part labels for the matched components
        #! Remember there can be multiple part labels for a thing label
        for part_label in ref_labels[1:]:
            pred_part_slice = prediction_arr[part_label]
            ref_part_slice = reference_arr[part_label]

            encompassed_pred_parts = []
            encompassed_ref_parts = []

            #? isolate the part labels for the matched components
            # Loop over unique predicted part instances (excluding 0)
            for pred_part_instance in np.unique(pred_part_slice):
                if pred_part_instance == 0:
                    continue
                curr_pred_part_instance = pred_part_slice == (pred_part_instance)
                flag = _is_part_encompassed(
                    part_component=curr_pred_part_instance,
                    thing_mask=matched_pred_component,
                )
                if flag:
                    encompassed_pred_parts.append(pred_part_instance)

            # Loop over unique reference part instances (excluding 0)
            for ref_part_instance in np.unique(ref_part_slice):
                if ref_part_instance == 0:
                    continue
                curr_ref_part_instance = ref_part_slice == (ref_part_instance)
                flag = _is_part_encompassed(
                    part_component=curr_ref_part_instance,
                    thing_mask=matched_ref_component,
                )
                if flag:
                    encompassed_ref_parts.append(ref_part_instance)



            # If there are multiple encompassed predicted parts, relabel all encompassed parts to the lowest label
            if len(encompassed_pred_parts) > 1:
                lowest_label = min(encompassed_pred_parts)
                for label in encompassed_pred_parts:
                    pred_part_slice[pred_part_slice == (label)] = lowest_label

            # If there are multiple encompassed reference parts, relabel all encompassed parts to the lowest label
            if len(encompassed_ref_parts) > 1:
                lowest_label = min(encompassed_ref_parts)
                for label in encompassed_ref_parts:
                    ref_part_slice[ref_part_slice == (label)] = lowest_label

        ref_unique_labels = [int(label) for label in np.unique(ref_part_slice) if label > 0]
        all_part_labels = calculate_all_label_pairs(
            prediction_arr=pred_part_slice,
            reference_arr=ref_part_slice,
            ref_labels=ref_unique_labels,
        )

        curr_part_pairs = [
            (
                matching_metric.value(
                    ref_part_slice, pred_part_slice, i[0], i[1]
                ),
                (i[0], i[1]),
            )
            for i in all_part_labels
        ]
        
        chosen_part_pairs = sorted(
            curr_part_pairs,
            key=lambda x: x[0],
            reverse=not matching_metric.decreasing,
        )

        # discard all pairs which are not the pair in cosideration for this loop
        chosen_part_pairs = [
            (score, pair)
            for score, pair in chosen_part_pairs
            if pair[0] == i and pair[1] == j
        ]


        # add the class values to the thing pairs
        def _update_thing_pairs_with_part_scores(thing_pairs, part_pairs):
            # Convert part pairs to dictionary for lookup
            part_scores = {pair: score for score, pair in part_pairs}
            
            # Return list with updated scores where applicable
            return [
                ((score + part_scores[pair]) / 2, pair) if pair in part_scores else (score, pair)
                for score, pair in thing_pairs
            ]
                
        # Update the thing pairs with the mean of the part pairs
        updated_thing_pairs = _update_thing_pairs_with_part_scores(
            thing_pairs=updated_thing_pairs,
            part_pairs=chosen_part_pairs,
        )


    return updated_thing_pairs

def _is_part_encompassed(part_component: np.ndarray, thing_mask: np.ndarray) -> bool:
    """Checks if a part is encompassed by the thing label.

    A part is valid if:
    1. It overlaps with the dilated thing OR
    2. It's completely surrounded by the thing.

    Args:
        part_component (np.ndarray): The connected component representing a part.
        thing_mask (np.ndarray): The binary mask representing the thing label.

    Returns:
        bool: True if the part is valid (encompassed by the thing), otherwise False.
    """
    from scipy import ndimage

    dilated_thing = ndimage.binary_dilation(thing_mask)

    # Check if part touches the dilated thing
    if np.any(part_component & dilated_thing):
        return True
    else:
        # Check if the part is completely surrounded by the thing
        dilated_part = ndimage.binary_dilation(part_component)
        boundary = dilated_part & ~part_component

        # If all boundary pixels are thing pixels, the part is surrounded
        return np.all((boundary & thing_mask) == boundary)
    
def _get_orig_onehotcc_structure(
    arr_onehot: np.ndarray,
    num_ref_labels: int,
    processing_pair_orig_shape: tuple[int, ...],
) -> np.ndarray:
        return arr_onehot.reshape((num_ref_labels + 1,) + processing_pair_orig_shape)

def _remove_isolated_parts(
    array: np.ndarray, thing_label: int, part_labels: list[int]
) -> np.ndarray:
    """Checks if part labels are connected to thing labels and creates a mask for valid parts.

    Args:
        array (np.ndarray): The array containing thing and part labels.
        thing_label (int): The label representing the thing/semantic class.
        part_labels (list[int]): The labels representing the part classes.

    Returns:
        np.ndarray: A binary mask where True indicates valid regions (thing and valid parts)
                   and False indicates regions to be zeroed out.
    """
    from scipy import ndimage

    # Create a mask for the thing
    thing_mask = array == thing_label

    # Create a mask for all parts
    parts_mask = np.zeros_like(array, dtype=bool)
    for part_label in part_labels:
        parts_mask[array == part_label] = True

    # Label the connected components in the parts mask
    labeled_parts, num_parts = ndimage.label(parts_mask)

    # For each connected component in the parts mask, check if it's encompassed
    encompassed_parts = np.zeros_like(parts_mask)
    for i in range(1, num_parts + 1):
        part_component = labeled_parts == i

        # Check if the part is encompassed
        if _is_part_encompassed(part_component, thing_mask):
            encompassed_parts |= part_component

    # Final encompassed regions are the thing plus encompassed parts
    encompassed_regions = thing_mask | encompassed_parts

    return encompassed_regions


def calculate_all_label_pairs(
    prediction_arr: np.ndarray,
    reference_arr: np.ndarray,
    ref_labels: tuple[int, ...],
) -> list[tuple[int, int]]:
    """Returns all possible pairs of reference and prediction labels

    Args:
        prediction_arr (np.ndarray): Numpy array containing the prediction labels.
        reference_arr (np.ndarray): Numpy array containing the reference labels.
        ref_labels (tuple[int, ...]): Tuple of unique reference labels.

    Returns:
        list[tuple[int, int]]: List of all (reference_label, prediction_label) pairs
    """
    # Get non-zero prediction and reference labels
    pred_labels = [int(label) for label in np.unique(prediction_arr) if label > 0]
    ref_labels = [int(label) for label in ref_labels if label > 0]
    
    # Create all possible pairs using list comprehension
    return [(ref_label, pred_label) for ref_label in ref_labels for pred_label in pred_labels]