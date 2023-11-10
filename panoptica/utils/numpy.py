import warnings
from typing import Tuple

import cc3d
import numpy as np
from scipy import ndimage


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
    if max_value < 256:
        dtype = np.uint8
    elif max_value < 65536:
        dtype = np.uint16
    elif max_value < 4294967295:
        dtype = np.uint32
    else:
        dtype = np.uint64
    return dtype