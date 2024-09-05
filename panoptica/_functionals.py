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
    instance_pairs = [
        (reference_arr, prediction_arr, i[0], i[1])
        for i in _calc_overlapping_labels(
            prediction_arr=prediction_arr,
            reference_arr=reference_arr,
            ref_labels=ref_labels,
        )
    ]
    with Pool() as pool:
        mm_values = pool.starmap(matching_metric.value, instance_pairs)

    mm_pairs = [
        (i, (instance_pairs[idx][2], instance_pairs[idx][3]))
        for idx, i in enumerate(mm_values)
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
    assert prediction_arr.shape == reference_arr.shape

    combined = prediction_arr + reference_arr
    if combined.sum() == 0:
        combined += 1
    return _get_bbox_nd(combined, px_dist=px_pad)


def _round_to_n(value: float | int, n_significant_digits: int = 2):
    return (
        value
        if value == 0
        else round(
            value, -int(math.floor(math.log10(abs(value)))) + (n_significant_digits - 1)
        )
    )
