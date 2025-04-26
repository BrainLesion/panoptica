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

    print(mm_pairs)

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

def _remove_isolated_parts(array: np.ndarray, thing_label: int, part_labels: list[int]) -> np.ndarray:
    """Checks if part labels are connected to thing labels and creates a mask for valid parts.
    
    Args:
        array (np.ndarray): The array containing thing and part labels.
        thing_label (int): The label representing the thing/semantic class.
        part_labels (list[int]): The labels representing the part classes.
        
    Returns:
        np.ndarray: A binary mask where True indicates valid regions (thing and valid parts)
                   and False indicates regions to be zeroed out.
    """
    # Create a mask for the thing
    thing_mask = array == thing_label
    
    # Create a mask for all parts
    parts_mask = np.zeros_like(array, dtype=bool)
    for part_label in part_labels:
        parts_mask[array == part_label] = True
    
    from scipy import ndimage
    
    # Label the connected components in the parts mask
    labeled_parts, num_parts = ndimage.label(parts_mask)
    
    # Create a dilated thing mask
    dilated_thing = ndimage.binary_dilation(thing_mask)
    
    # For each connected component in the parts mask, check if it's valid
    valid_parts = np.zeros_like(parts_mask)
    for i in range(1, num_parts + 1):
        part_component = labeled_parts == i
        
        # A part is valid if:
        # 1. It overlaps with the dilated thing OR
        # 2. It's completely surrounded by the thing
        
        # Check if part touches the dilated thing
        if np.any(part_component & dilated_thing):
            valid_parts |= part_component
        else:
            # Additionally check if the part is completely surrounded by the thing
            # by dilating the part and checking if all its boundaries touch the thing
            dilated_part = ndimage.binary_dilation(part_component)
            boundary = dilated_part & ~part_component
            
            # If all boundary pixels are thing pixels, the part is surrounded
            if np.all((boundary & thing_mask) == boundary):
                valid_parts |= part_component
    
    # Final valid regions are the thing plus valid parts
    valid_regions = thing_mask | valid_parts
    
    return valid_regions

def _calc_matching_metric_of_overlapping_partlabels(
    prediction_arr: np.ndarray,
    reference_arr: np.ndarray,
    ref_label: int,
    pred_label: int,
    part_labels: list[int],
    matching_metric: "Metric",
) -> float:
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
    # First calculate the score for the thing/panoptic label
    thing_score = matching_metric.value(reference_arr, prediction_arr, ref_label, pred_label)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(reference_arr)
    ax[0].set_title('Reference')
    ax[1].imshow(prediction_arr)
    ax[1].set_title('Prediction')
    plt.show()

    # Get masks for the reference and prediction regions of interest
    ref_mask = reference_arr == ref_label
    pred_mask = prediction_arr == pred_label

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(ref_mask)
    ax[0].set_title('Reference Mask')
    ax[1].imshow(pred_mask)
    ax[1].set_title('Prediction Mask')
    plt.show()

    # Create a combined mask for the region of interest (where to look for parts)
    region_of_interest = ref_mask | pred_mask

    import matplotlib.pyplot as plt
    plt.imshow(region_of_interest)
    plt.title('Region of Interest')
    plt.show()
    
    # Extract parts only in the region of interest to avoid parts from other things
    part_scores = []
    for part_label in part_labels:
        # Check if this part exists in both reference and prediction
        ref_part = np.where(reference_arr == part_label, 1, 0) * region_of_interest
        pred_part = np.where(prediction_arr == part_label, 1, 0) * region_of_interest

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(ref_part)
        ax[0].set_title('Reference Part')
        ax[1].imshow(pred_part)
        ax[1].set_title('Prediction Part')
        plt.show()
        
        # Only calculate score if part exists in at least one of them
        if np.any(ref_part) or np.any(pred_part):
            part_score = matching_metric.value(ref_part, pred_part)
            part_scores.append(part_score)
    
    print(f"Part scores: {part_scores}")
    # Calculate the mean part score if there are any parts
    if part_scores:
        mean_part_score = sum(part_scores) / len(part_scores)
        # Return the mean of the thing score and the mean part score
        print('Final Score:', (thing_score + mean_part_score) / 2)
        return (thing_score + mean_part_score) / 2
    else:
        # If no parts are present, return just the thing score
        return thing_score