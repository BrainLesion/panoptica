import itertools
import warnings

import numpy as np


def _unique_without_zeros(arr: np.ndarray) -> np.ndarray:
    """
    Get unique non-zero values from a NumPy array.

    Parameters:
        arr (np.ndarray): Input NumPy array.

    Returns:
        np.ndarray: Unique non-zero values from the input array.

    Issues a warning if negative values are present.
    """
    if np.any(arr < 0):
        warnings.warn("Negative values are present in the input array.")

    return np.unique(arr[arr != 0])


def _count_unique_without_zeros(arr: np.ndarray) -> int:
    """
    Count the number of unique elements in the input NumPy array, excluding zeros.

    Args:
        arr (np.ndarray): Input array.

    Returns:
        int: Number of unique elements excluding zeros.
    """
    if np.any(arr < 0):
        warnings.warn("Negative values are present in the input array.")

    return len(_unique_without_zeros(arr))


def _get_smallest_fitting_uint(max_value: int) -> type:
    """
    Determine the smallest unsigned integer type that can accommodate the given maximum value.

    Args:
        max_value (int): The maximum value to be accommodated.

    Returns:
        type: The NumPy data type (e.g., np.uint8, np.uint16, np.uint32, np.uint64).

    Example:
    >>> _get_smallest_fitting_uint(255)
    <class 'numpy.uint8'>
    """
    if max_value < 256:
        dtype = np.uint8
    elif max_value < 65536:
        dtype = np.uint16
    elif max_value < 4294967295:
        dtype = np.uint32
    else:
        dtype = np.uint64
    return dtype


def _get_bbox_nd(
    img: np.ndarray,
    px_dist: int | tuple[int, ...] = 0,
) -> tuple[slice, ...]:
    """calculates a bounding box in n dimensions given a image (factor ~2 times faster than compute_crop_slice)

    Args:
        img: input array
        px_dist: int | tuple[int]: dist (int): The amount of padding to be added to the cropped image. If int, will apply the same padding to each dim. Default value is 0.

    Returns:
        list of boundary coordinates [x_min, x_max, y_min, y_max, z_min, z_max]
    """
    assert img is not None, "bbox_nd: received None as image"
    assert np.count_nonzero(img) > 0, "bbox_nd: img is empty, cannot calculate a bbox"
    N = img.ndim
    shp = img.shape
    if isinstance(px_dist, int):
        px_dist = np.ones(N, dtype=np.uint8) * px_dist
    assert (
        len(px_dist) == N
    ), f"dimension mismatch, got img shape {shp} and px_dist {px_dist}"

    out = []
    for ax in itertools.combinations(reversed(range(N)), N - 1):
        nonzero = np.any(a=img, axis=ax)
        out.extend(np.where(nonzero)[0][[0, -1]])
    out = tuple(
        slice(
            max(out[i] - px_dist[i // 2], 0),
            min(out[i + 1] + px_dist[i // 2], shp[i // 2]) + 1,
        )
        for i in range(0, len(out), 2)
    )
    return out
