import sys, warnings
import numpy as np
from panoptica.metrics.assd import __surface_distances


def _compute_normalized_surface_dice(
    reference,
    prediction,
    voxelspacing=None,
    connectivity=1,
    threshold=None,
    *args,
) -> float:
    """
    Computes the normalized surface dice between two instances.
    Copied from https://github.com/mlcommons/GaNDLF/blob/641e6c75e698ce2b52a2acf12bbdcb819804c3e2/GANDLF/metrics/segmentation.py#L142

    This implementation differs from the official surface dice implementation! These two are not comparable!!!!!
    The normalized surface dice is symmetric, so it should not matter whether a or b is the reference image
    This implementation natively supports 2D and 3D images.

    Args:
        reference (np.ndarray): The reference mask
        prediction (np.ndarray): The prediction mask
        voxelspacing (_type_, optional): The voxel spacing. Defaults to None.
        connectivity (int, optional): The connectivity. Defaults to 1.
        threshold (float): The threshold for the surface distance in real world coordinates. If None, the minimum voxel spacing is used. If voxelspacing is None, the threshold is set to 0.5.

    Returns:
        float: the normalized surface dice between a and b
    """
    # calculate the surface distances between the two masks
    a_to_b = __surface_distances(prediction, reference, voxelspacing, connectivity)
    b_to_a = __surface_distances(reference, prediction, voxelspacing, connectivity)

    # if threshold is None, use the minimum voxel spacing as the threshold
    if threshold is None:
        threshold = min(voxelspacing) if voxelspacing is not None else 0.5
    if threshold == 0.5:
        warnings.warn(
            "The threshold is set to 0.5, which is the default value, which may not be appropriate for your data."
        )

    if isinstance(a_to_b, int):
        return 0
    if isinstance(b_to_a, int):
        return 0
    numel_a = len(a_to_b)
    numel_b = len(b_to_a)
    tp_a = np.sum(a_to_b <= threshold) / numel_a
    tp_b = np.sum(b_to_a <= threshold) / numel_b
    fp = np.sum(a_to_b > threshold) / numel_a
    fn = np.sum(b_to_a > threshold) / numel_b
    dc = (tp_a + tp_b) / (tp_a + tp_b + fp + fn + sys.float_info.min)
    return dc


def _compute_instance_normalized_surface_dice(
    ref_labels: np.ndarray,
    pred_labels: np.ndarray,
    ref_instance_idx: int | None = None,
    pred_instance_idx: int | None = None,
    voxelspacing=None,
    connectivity=1,
    threshold=None,
    *args,
):
    """Computes the instance-wise normalized surface dice between two instances.

    Args:
        reference (np.ndarray): The reference mask
        prediction (np.ndarray): The prediction mask
        ref_instance_idx (int | None, optional): The reference instance index. Defaults to None.
        pred_instance_idx (int | None, optional): The prediction instance index. Defaults to None.
        voxelspacing (_type_, optional): The voxel spacing. Defaults to None.
        connectivity (int, optional): The connectivity. Defaults to 1.
        threshold (float): The threshold for the surface distance in real world coordinates. If None, the minimum voxel spacing is used. If voxelspacing is None, the threshold is set to 0.5.

    Returns:
        float: Normalized Surface Dice
    """
    if ref_instance_idx is None and pred_instance_idx is None:
        return _compute_normalized_surface_dice(
            reference=ref_labels,
            prediction=pred_labels,
            voxelspacing=voxelspacing,
            connectivity=connectivity,
            threshold=threshold,
        )
    ref_instance_mask = ref_labels == ref_instance_idx
    pred_instance_mask = pred_labels == pred_instance_idx
    return _compute_normalized_surface_dice(
        reference=ref_instance_mask,
        prediction=pred_instance_mask,
        voxelspacing=voxelspacing,
        connectivity=connectivity,
        threshold=threshold,
    )
