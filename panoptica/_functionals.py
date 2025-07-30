from typing import TYPE_CHECKING
from multiprocessing import Pool
import numpy as np
import math
from panoptica.utils.constants import CCABackend
from panoptica.utils.numpy_utils import _get_bbox_nd
from scipy import ndimage

if TYPE_CHECKING:
    from panoptica.metrics import Metric

# -------------------- LABEL OPERATIONS --------------------


def _map_labels(
    arr: np.ndarray,
    label_map: dict[np.integer, np.integer],
) -> np.ndarray:
    """
    Maps labels in the given array according to the label_map dictionary.

    Args:
        arr: Array containing labels to be mapped
        label_map: A dictionary that maps original label values to new label values

    Returns:
        np.ndarray: Copy of the remapped array
    """
    if len(label_map) == 0:
        return arr.copy()
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
        array: Binary array containing connected components
        cca_backend: Enum indicating the connected components algorithm backend

    Returns:
        tuple: Labeled array and the number of connected components

    Raises:
        NotImplementedError: If the specified backend is not implemented
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
) -> np.ndarray:
    """
    Calculates a bounding box based on paired prediction and reference arrays.

    Args:
        prediction_arr: The predicted segmentation array
        reference_arr: The ground truth segmentation array
        px_pad: Padding to apply around the bounding box

    Returns:
        np.ndarray: The bounding box coordinates around the combined non-zero regions
    """
    assert prediction_arr.shape == reference_arr.shape

    combined = prediction_arr + reference_arr
    if combined.sum() == 0:
        combined += 1
    return _get_bbox_nd(combined, px_dist=px_pad)


def _round_to_n(value: float | int, n_significant_digits: int = 2) -> float:
    """
    Rounds a number to a specified number of significant digits.

    Args:
        value: The number to be rounded
        n_significant_digits: The number of significant digits to round to

    Returns:
        float: The rounded value
    """
    return (
        value
        if value == 0
        else round(
            value, -int(math.floor(math.log10(abs(value)))) + (n_significant_digits - 1)
        )
    )


def _get_orig_onehotcc_structure(
    arr_onehot: np.ndarray,
    num_ref_labels: int,
    processing_pair_orig_shape: tuple[int, ...],
) -> np.ndarray:
    """
    Reshapes a one-hot encoded array back to its original structure.

    Args:
        arr_onehot: One-hot encoded array
        num_ref_labels: Number of reference labels
        processing_pair_orig_shape: Original shape of the array

    Returns:
        np.ndarray: Reshaped array
    """
    return arr_onehot.reshape((num_ref_labels + 1,) + processing_pair_orig_shape)


# -------------------- LABEL MATCHING & OVERLAP CALCULATION --------------------


def _calc_overlapping_labels(
    prediction_arr: np.ndarray,
    reference_arr: np.ndarray,
    ref_labels: tuple[int, ...],
) -> list[tuple[int, int]]:
    """
    Calculates the pairs of labels that overlap in at least one voxel.

    Args:
        prediction_arr: Array containing prediction labels
        reference_arr: Array containing reference labels
        ref_labels: List of unique reference labels

    Returns:
        list: Pairs of (ref_label, pred_label) that overlap
    """
    overlap_arr = prediction_arr.astype(np.uint32)
    max_ref = max(ref_labels) + 1
    overlap_arr = (overlap_arr * max_ref) + reference_arr
    overlap_arr[reference_arr == 0] = 0

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
    """
    Calculates the matching metric for all overlapping labels.

    Args:
        prediction_arr: Array containing prediction labels
        reference_arr: Array containing reference labels
        ref_labels: List of unique reference labels
        matching_metric: Metric to use for matching

    Returns:
        list: Sorted list of (metric_value, (ref_label, pred_label)) pairs
    """
    overlapping_labels = _calc_overlapping_labels(
        prediction_arr=prediction_arr,
        reference_arr=reference_arr,
        ref_labels=ref_labels,
    )

    mm_pairs = [
        (matching_metric.value(reference_arr, prediction_arr, i[0], i[1]), (i[0], i[1]))
        for i in overlapping_labels
    ]

    mm_pairs = sorted(
        mm_pairs, key=lambda x: x[0], reverse=not matching_metric.decreasing
    )

    return mm_pairs


def calculate_all_label_pairs(
    prediction_arr: np.ndarray,
    ref_labels: tuple[int, ...],
) -> list[tuple[int, int]]:
    """
    Returns all possible pairs of reference and prediction labels.

    Args:
        prediction_arr: Array containing prediction labels
        ref_labels: Tuple of unique reference labels

    Returns:
        list: All (reference_label, prediction_label) pairs
    """
    pred_labels = [int(label) for label in np.unique(prediction_arr) if label > 0]
    ref_labels = [int(label) for label in ref_labels if label > 0]

    return [
        (ref_label, pred_label)
        for ref_label in ref_labels
        for pred_label in pred_labels
    ]


# -------------------- PART LABEL PROCESSING --------------------


def _is_part_encompassed(part_component: np.ndarray, thing_mask: np.ndarray) -> bool:
    """
    Checks if a part is encompassed by the thing label.

    Args:
        part_component: Connected component representing a part
        thing_mask: Binary mask representing the thing label

    Returns:
        bool: True if the part is valid (encompassed by the thing)
    """
    dilated_thing = ndimage.binary_dilation(thing_mask)

    # Check if part touches the dilated thing
    if np.any(part_component & dilated_thing):
        return True

    # Check if the part is completely surrounded by the thing
    dilated_part = ndimage.binary_dilation(part_component)
    boundary = dilated_part & ~part_component

    # If all boundary pixels are thing pixels, the part is surrounded
    return np.all((boundary & thing_mask) == boundary)


def _get_encompassed_parts(part_slice: np.ndarray, thing_mask: np.ndarray) -> list[int]:
    """
    Get all part instances that are encompassed by the thing mask.

    Args:
        part_slice: Array containing part labels
        thing_mask: Binary mask of the thing instance

    Returns:
        list: Labels of parts encompassed by the thing mask
    """
    encompassed_parts = []
    for part_instance in np.unique(part_slice):
        if part_instance == 0:
            continue
        curr_part_instance = part_slice == part_instance
        if _is_part_encompassed(curr_part_instance, thing_mask):
            encompassed_parts.append(part_instance)
    return encompassed_parts


def _consolidate_multiple_parts(
    part_slice: np.ndarray, encompassed_parts: list[int]
) -> np.ndarray:
    """
    Consolidate multiple encompassed parts to the lowest label.

    Args:
        part_slice: Array containing part labels
        encompassed_parts: List of part labels to consolidate

    Returns:
        np.ndarray: Updated part slice with consolidated labels
    """
    if len(encompassed_parts) > 1:
        lowest_label = min(encompassed_parts)
        part_slice[np.isin(part_slice, encompassed_parts)] = lowest_label
    return part_slice


def _find_optimal_part_matching_per_type(
    part_pairs: list[tuple[float, tuple[int, int]]], matching_metric: "Metric"
) -> list[float]:
    """
    Find optimal matching between reference and predicted parts for a single part type.

    Args:
        part_pairs: List of (score, (ref_label, pred_label)) tuples
        matching_metric: Metric used for scoring

    Returns:
        list: Optimal matching scores
    """
    if not part_pairs:
        return []

    # Handle mismatch case
    if any(pair[1] == (-1, -1) for pair in part_pairs):
        return [0.0]  # Penalty for part type mismatch

    # Get unique parts
    ref_parts = set(pair[1][0] for pair in part_pairs)
    pred_parts = set(pair[1][1] for pair in part_pairs)

    # Sort part pairs by score
    sorted_pairs = sorted(
        part_pairs, key=lambda x: x[0], reverse=not matching_metric.decreasing
    )

    matching_scores = []
    used_pred_parts = set()

    for ref_part in sorted(ref_parts):
        # Find best available predicted part for this reference part
        candidates = [
            (score, p_part)
            for score, (r_part, p_part) in sorted_pairs
            if r_part == ref_part and p_part not in used_pred_parts
        ]

        if candidates:
            # Prefer positive scores, then sort by metric preference
            positive_candidates = [(s, p) for s, p in candidates if s > 0]
            final_candidates = (
                positive_candidates if positive_candidates else candidates
            )

            final_candidates.sort(
                key=lambda x: x[0], reverse=not matching_metric.decreasing
            )
            best_score, best_pred_part = final_candidates[0]

            matching_scores.append(best_score)
            used_pred_parts.add(best_pred_part)

    return matching_scores


def _calculate_part_scores_for_all_types(
    thing_pair: tuple[int, int],
    prediction_arr: np.ndarray,
    reference_arr: np.ndarray,
    num_ref_labels: int,
    matching_metric: "Metric",
) -> dict[int, list[tuple[float, tuple[int, int]]]]:
    """
    Calculate part scores for all part types for a given thing pair.

    Args:
        thing_pair: Tuple of (ref_thing_label, pred_thing_label)
        prediction_arr: Prediction array
        reference_arr: Reference array
        num_ref_labels: Number of reference labels
        matching_metric: Metric used for scoring

    Returns:
        dict: Dictionary mapping part type to list of part scores
    """
    i, j = thing_pair
    matched_ref_component = reference_arr[0] == i
    matched_pred_component = prediction_arr[0] == j

    part_scores_by_type = {}

    # Process each part label (starting from channel 2)
    for part_label in range(2, num_ref_labels + 1):
        # Create isolated part slices for just this thing pair
        pred_part_slice = prediction_arr[part_label].copy()
        ref_part_slice = reference_arr[part_label].copy()

        # Zero out everything outside the thing masks
        pred_part_slice[~matched_pred_component] = 0
        ref_part_slice[~matched_ref_component] = 0

        # Get encompassed parts for both prediction and reference
        encompassed_pred_parts = _get_encompassed_parts(
            pred_part_slice, matched_pred_component
        )
        encompassed_ref_parts = _get_encompassed_parts(
            ref_part_slice, matched_ref_component
        )

        # Skip if no parts found in either
        if not encompassed_pred_parts and not encompassed_ref_parts:
            continue

        # Consolidate multiple parts to lowest label
        pred_part_slice = _consolidate_multiple_parts(
            pred_part_slice, encompassed_pred_parts
        )
        ref_part_slice = _consolidate_multiple_parts(
            ref_part_slice, encompassed_ref_parts
        )

        part_pairs_for_type = []

        # Check for part existence mismatch for this part type
        ref_has_parts = len(encompassed_ref_parts) > 0
        pred_has_parts = len(encompassed_pred_parts) > 0

        if ref_has_parts and pred_has_parts:
            # Both have parts - calculate all possible pairs
            ref_unique_labels = list(set(encompassed_ref_parts))
            pred_unique_labels = list(set(encompassed_pred_parts))

            for ref_label in ref_unique_labels:
                for pred_label in pred_unique_labels:
                    score = matching_metric.value(
                        ref_part_slice, pred_part_slice, ref_label, pred_label
                    )
                    part_pairs_for_type.append((score, (ref_label, pred_label)))

        elif ref_has_parts != pred_has_parts:
            # Part mismatch for this part type - assign penalty score
            part_pairs_for_type.append((0.0, (-1, -1)))  # Use -1 to indicate mismatch

        # Store results for this part type
        if part_pairs_for_type:
            part_scores_by_type[part_label] = part_pairs_for_type

    return part_scores_by_type


def _calculate_combined_part_score(
    part_scores_by_type: dict[int, list[tuple[float, tuple[int, int]]]],
    matching_metric: "Metric",
) -> tuple[float, int]:
    """
    Calculate combined score across all part types.

    Args:
        part_scores_by_type: Dictionary mapping part type to part scores
        matching_metric: Metric used for scoring

    Returns:
        tuple: (total_part_score, total_part_count)
    """
    if not part_scores_by_type:
        return 0.0, 0

    all_part_scores = []

    for part_type, part_pairs in part_scores_by_type.items():
        # Get optimal matching for this part type
        matching_scores = _find_optimal_part_matching_per_type(
            part_pairs, matching_metric
        )

        if matching_scores:
            # Add all individual part scores (no averaging within part type)
            all_part_scores.extend(matching_scores)

    total_part_score = sum(all_part_scores) if all_part_scores else 0.0
    total_part_count = len(all_part_scores)

    return total_part_score, total_part_count


def _calc_matching_metric_of_overlapping_partlabels(
    prediction_arr: np.ndarray,
    reference_arr: np.ndarray,
    processing_pair_orig_shape: tuple[int, ...],
    num_ref_labels: int,
    matching_metric: "Metric",
) -> list[tuple[float, tuple[int, int]]]:
    """
    Calculates a combined matching metric for thing+part overlapping labels.

    This function calculates a combined score between the panoptic (thing) labels
    and their corresponding parts.

    Args:
        prediction_arr: Array containing prediction labels
        reference_arr: Array containing reference labels
        processing_pair_orig_shape: Original shape for processing
        num_ref_labels: Number of reference labels
        matching_metric: Metric to use for evaluation

    Returns:
        list: List of (score, (ref_label, pred_label)) pairs
    """
    # Reshape arrays to original structure
    prediction_arr = _get_orig_onehotcc_structure(
        prediction_arr, num_ref_labels, processing_pair_orig_shape
    )
    reference_arr = _get_orig_onehotcc_structure(
        reference_arr, num_ref_labels, processing_pair_orig_shape
    )

    # Calculate overlapping labels for things (channel 0)
    overlapping_labels = _calc_overlapping_labels(
        prediction_arr=prediction_arr[0],
        reference_arr=reference_arr[0],
        ref_labels=[max(prediction_arr[0].max(), reference_arr[0].max())],
    )

    # Calculate thing scores
    thing_pairs = [
        (matching_metric.value(reference_arr[0], prediction_arr[0], i, j), (i, j))
        for i, j in overlapping_labels
    ]

    sorted_thing_pairs = sorted(
        thing_pairs, key=lambda x: x[0], reverse=not matching_metric.decreasing
    )

    # Process each thing pair to include part scores
    final_pairs = []

    for thing_score, thing_pair in sorted_thing_pairs:
        i, j = thing_pair
        matched_ref_component = reference_arr[0] == i
        matched_pred_component = prediction_arr[0] == j

        # Check if parts exist within this specific thing pair's regions for each part type
        pred_part_types_in_region = set()
        ref_part_types_in_region = set()

        for part_label in range(2, num_ref_labels + 1):
            # Check prediction parts within this thing's region
            pred_part_in_region = prediction_arr[part_label][matched_pred_component]
            if pred_part_in_region.max() > 0:
                pred_part_types_in_region.add(part_label)

            # Check reference parts within this thing's region
            ref_part_in_region = reference_arr[part_label][matched_ref_component]
            if ref_part_in_region.max() > 0:
                ref_part_types_in_region.add(part_label)

        # Get part scores for all part types
        part_scores_by_type = _calculate_part_scores_for_all_types(
            thing_pair, prediction_arr, reference_arr, num_ref_labels, matching_metric
        )

        # Determine scoring strategy based on part type presence
        if pred_part_types_in_region and ref_part_types_in_region:
            # Both have some parts - calculate combined part score
            total_part_score, total_part_count = _calculate_combined_part_score(
                part_scores_by_type, matching_metric
            )

            # Calculate combined score: (thing_score + sum_of_all_part_scores) / (1 + total_part_count)
            if total_part_count > 0:
                updated_score = (thing_score + total_part_score) / (
                    1 + total_part_count
                )
            else:
                updated_score = thing_score

        elif pred_part_types_in_region != ref_part_types_in_region:
            # Part type mismatch - penalize
            updated_score = thing_score / 2

        else:
            # Neither has parts - use thing score directly
            updated_score = thing_score

        final_pairs.append((updated_score, thing_pair))

    # Final sort by updated scores
    final_sorted_pairs = sorted(
        final_pairs, key=lambda x: x[0], reverse=not matching_metric.decreasing
    )

    return final_sorted_pairs
