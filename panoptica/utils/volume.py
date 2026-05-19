import numpy as np


def compute_ref_voxel_count_and_volume(
    reference_arr: np.ndarray,
    ref_idx: int,
    voxelspacing: tuple[float, ...] | None = None,
) -> tuple[int, float]:
    """Compute voxel count and physical volume of a single reference instance.

    Used for unmatched reference instances, where no prediction comparison is needed.
    """
    if voxelspacing is None:
        voxelspacing = (1.0,) * reference_arr.ndim
    voxel_count = int(np.count_nonzero(reference_arr == ref_idx))
    volume = float(voxel_count * np.prod(voxelspacing))
    return voxel_count, volume
