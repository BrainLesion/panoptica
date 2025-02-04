import numpy as np
from scipy.spatial import cKDTree
from panoptica.metrics.assd import __surface_distances


def _compute_instance_hausdorff_distance(
    ref_labels: np.ndarray,
    pred_labels: np.ndarray,
    ref_instance_idx: int | None = None,
    pred_instance_idx: int | None = None,
    voxelspacing=None,
    connectivity=1,
):
    """Computes the hausdroff distance between two instances.

    Args:
        ref_labels (np.ndarray): _description_
        pred_labels (np.ndarray): _description_
        ref_instance_idx (int | None, optional): _description_. Defaults to None.
        pred_instance_idx (int | None, optional): _description_. Defaults to None.
        voxelspacing (_type_, optional): _description_. Defaults to None.
        connectivity (int, optional): _description_. Defaults to 1.

    Returns:
        float: Hausdorff Distance
    """
    if ref_instance_idx is None and pred_instance_idx is None:
        return _compute_hausdorff_distance(
            reference=ref_labels,
            prediction=pred_labels,
            voxelspacing=voxelspacing,
            connectivity=connectivity,
        )
    ref_instance_mask = ref_labels == ref_instance_idx
    pred_instance_mask = pred_labels == pred_instance_idx
    return _compute_hausdorff_distance(
        reference=ref_instance_mask,
        prediction=pred_instance_mask,
        voxelspacing=voxelspacing,
        connectivity=connectivity,
    )


def _compute_hausdorff_distance(
    reference,
    prediction,
    voxelspacing=None,
    connectivity=1,
    *args,
) -> float:
    """Computes the hausdroff distance between two instances.

    Args:
        ref_labels (np.ndarray): _description_
        pred_labels (np.ndarray): _description_
        voxelspacing (_type_, optional): _description_. Defaults to None.
        connectivity (int, optional): _description_. Defaults to 1.

    Returns:
        float: Hausdorff Distance
    """
    hd1 = __surface_distances(prediction, reference, voxelspacing, connectivity).max()
    hd2 = __surface_distances(reference, prediction, voxelspacing, connectivity).max()
    hd = max(hd1, hd2)
    return hd


def _compute_instance_hausdorff_distance95(
    ref_labels: np.ndarray,
    pred_labels: np.ndarray,
    ref_instance_idx: int | None = None,
    pred_instance_idx: int | None = None,
    voxelspacing=None,
    connectivity=1,
):
    """Computes the hausdroff distance between two instances.

    Args:
        ref_labels (np.ndarray): _description_
        pred_labels (np.ndarray): _description_
        ref_instance_idx (int | None, optional): _description_. Defaults to None.
        pred_instance_idx (int | None, optional): _description_. Defaults to None.
        voxelspacing (_type_, optional): _description_. Defaults to None.
        connectivity (int, optional): _description_. Defaults to 1.

    Returns:
        float: Hausdorff Distance
    """
    if ref_instance_idx is None and pred_instance_idx is None:
        return _compute_hausdorff_distance95(
            reference=ref_labels,
            prediction=pred_labels,
            voxelspacing=voxelspacing,
            connectivity=connectivity,
        )
    ref_instance_mask = ref_labels == ref_instance_idx
    pred_instance_mask = pred_labels == pred_instance_idx
    return _compute_hausdorff_distance95(
        reference=ref_instance_mask,
        prediction=pred_instance_mask,
        voxelspacing=voxelspacing,
        connectivity=connectivity,
    )


def _compute_hausdorff_distance95(
    reference,
    prediction,
    voxelspacing=None,
    connectivity=1,
    *args,
) -> float:
    """Computes the hausdroff distance 95 between two instances.

    Args:
        ref_labels (np.ndarray): _description_
        pred_labels (np.ndarray): _description_
        voxelspacing (_type_, optional): _description_. Defaults to None.
        connectivity (int, optional): _description_. Defaults to 1.

    Returns:
        float: Hausdorff Distance
    """
    hd1 = __surface_distances(prediction, reference, voxelspacing, connectivity)
    hd2 = __surface_distances(reference, prediction, voxelspacing, connectivity)
    hd95 = np.percentile(np.hstack((hd1, hd2)), 95)
    return hd95
