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
        warnings.warn("Negative values are present in the input array.", UserWarning)

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
        warnings.warn("Negative values are present in the input array.", UserWarning)

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
    dtype: type
    if max_value < 256:
        dtype = np.uint8
    elif max_value < 65536:
        dtype = np.uint16
    elif max_value < 4294967295:
        dtype = np.uint32
    else:
        dtype = np.uint64
    return dtype


def recall_by_volume_bins(
    volumes: list[float],
    matched_flags: list[float],
    thresholds: list[float],
) -> dict[str, float]:
    """Instance detection recall stratified into user-supplied volume bins.

    ``thresholds`` are volume cut points (e.g. ``[160, 271, 451]`` voxels), given by
    the user rather than estimated from the data. They define ``len(thresholds) + 1``
    bins: ``rec_q0`` is ``volume < thresholds[0]``, ``rec_qi`` is
    ``thresholds[i - 1] <= volume < thresholds[i]``, and the last bin is
    ``volume >= thresholds[-1]``. The recall of a bin is the fraction of reference
    instances falling in it that were matched (the mean of their 0/1 matched flags);
    an empty bin yields ``nan``.

    Args:
        volumes: Per-reference-instance volumes (or voxel counts).
        matched_flags: Per-reference-instance matched indicator (1.0 matched, 0.0 not),
            aligned with ``volumes``.
        thresholds: Volume bin edges.

    Returns:
        dict[str, float]: ``{"rec_q0": ..., "rec_q1": ..., ...}`` with one entry per bin.
    """
    if len(volumes) != len(matched_flags):
        raise ValueError("volumes and matched_flags must have equal length")
    edges = sorted(thresholds)
    n_bins = len(edges) + 1
    sums = [0.0] * n_bins
    counts = [0] * n_bins
    if volumes:
        bin_indices = np.digitize(np.asarray(volumes, dtype=float), edges)
        for b, flag in zip(bin_indices.tolist(), matched_flags):
            sums[b] += float(flag)
            counts[b] += 1
    return {
        f"rec_q{b}": (sums[b] / counts[b] if counts[b] > 0 else float("nan"))
        for b in range(n_bins)
    }


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
    if img is None:
        raise ValueError("bbox_nd: received None as image")
    if np.count_nonzero(img) <= 0:
        raise ValueError("bbox_nd: img is empty, cannot calculate a bbox")
    N = img.ndim
    shp = img.shape
    pad: list[int] = [px_dist] * N if isinstance(px_dist, int) else list(px_dist)
    if len(pad) != N:
        raise ValueError(
            f"dimension mismatch, got img shape {shp} and px_dist {px_dist}"
        )

    bounds: list[int] = []
    for ax in itertools.combinations(reversed(range(N)), N - 1):
        nonzero = np.any(a=img, axis=ax)
        idx = np.where(nonzero)[0]
        bounds.extend((int(idx[0]), int(idx[-1])))
    return tuple(
        slice(
            max(bounds[i] - pad[i // 2], 0),
            min(bounds[i + 1] + pad[i // 2], shp[i // 2]) + 1,
        )
        for i in range(0, len(bounds), 2)
    )
