import numpy as np

def _compute_instance_physical_volume(
    reference_arr: np.ndarray,
    prediction_arr: np.ndarray,
    voxelspacing: tuple[float, ...] | None = None,
    *args,
    **kwargs,
) -> float:
    """
    Calculates the physical volume of the reference instance. Computed exclusively using the reference_arr.

    Args:
        reference_arr (np.ndarray): The reference (ground truth) instance mask.
        prediction_arr (np.ndarray): The prediction instance mask (ignored).
        voxelspacing (tuple[float, ...] | None, optional): The physical size of voxels per dimension. If None, defaults to unit spacing (1.0) matching the dimensionality of reference_arr.

    Returns:
        float: The physical volume of the reference instance.

    Raises:
        ValueError: If the length of voxelspacing does not match the dimensionality of reference_arr.
    """
    if voxelspacing is None:
        voxelspacing = (1.0,) * reference_arr.ndim
    
    if len(voxelspacing) != reference_arr.ndim:
        raise ValueError(
            f"Voxelspacing dimension ({len(voxelspacing)}) does not match "
            f"reference_arr dimensionality ({reference_arr.ndim})."
        )

    voxel_count = np.count_nonzero(reference_arr)
    unit_volume = float(np.prod(voxelspacing))
    
    return voxel_count * unit_volume
