import numpy as np

def _compute_instance_physical_volume(
    reference_arr: np.ndarray,
    prediction_arr: np.ndarray,
    voxelspacing: tuple = (1.0, 1.0, 1.0),
    *args,
    **kwargs,
) -> float:
    """Calculates the physical volume of the reference instance. Computed exclusively using the reference_arr.
        
    Args:
        reference_arr (np.ndarray): The reference (ground truth) instance mask.
        prediction_arr (np.ndarray): The prediction instance mask (ignored).
        voxelspacing (tuple, optional): The physical size of voxels in each dimension. 
            Defaults to (1.0, 1.0, 1.0).

    Returns:
        float: The physical volume of the reference instance (voxel_count * unit_volume).
    """

    voxel_count = np.count_nonzero(reference_arr)
    voxel_volume = float(np.prod(voxelspacing))
    return voxel_count * voxel_volume