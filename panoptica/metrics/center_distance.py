import numpy as np
from scipy.ndimage import center_of_mass


def _compute_center_of_mass(arr: np.ndarray):
    # enforce binary
    arr[arr != 0] = 1
    com = center_of_mass(arr)
    return com


def _compute_instance_center_distance(
    ref_labels: np.ndarray,
    pred_labels: np.ndarray,
    ref_instance_idx: int | None = None,
    pred_instance_idx: int | None = None,
    voxelspacing=None,
) -> float:
    """
    Compute the Center Distance between a specific pair of instances.

    The Center Distance is the distance between the center of mass of the compared instances.

    Args:
        ref_labels (np.ndarray): Reference instance labels.
        pred_labels (np.ndarray): Prediction instance labels.
        ref_instance_idx (int): Index of the reference instance.
        pred_instance_idx (int): Index of the prediction instance.

    Returns:
        float: Center Distance between the specified instances. Higher values
        indicate worse localization quality.
    """
    if ref_instance_idx is None and pred_instance_idx is None:
        return _compute_center_distance(
            reference=ref_labels,
            prediction=pred_labels,
        )
    ref_instance_mask = ref_labels == ref_instance_idx
    pred_instance_mask = pred_labels == pred_instance_idx
    return _compute_center_distance(
        reference=ref_instance_mask,
        prediction=pred_instance_mask,
    )


def _compute_center_distance(
    reference: np.ndarray,
    prediction: np.ndarray,
    voxelspacing=None,
) -> float:
    """
    Compute the Center Distance between a specific pair of instances.

    The Center Distance is the distance between the center of mass of the compared instances.

    Args:
        reference (np.ndarray): Reference binary mask.
        prediction (np.ndarray): Prediction binary mask.

    Returns:
        float: Center Distance between the specified instances. Higher values
        indicate worse localization quality.
    """
    ref_com = _compute_center_of_mass(reference)
    pred_com = _compute_center_of_mass(prediction)

    # Handle division by zero
    if reference.sum() == 0 and prediction.sum() == 0:
        return np.nan

    # Calculate metric
    diff_vector = np.subtract(pred_com, ref_com)
    if voxelspacing is not None:
        assert len(voxelspacing) == len(
            pred_com
        ), "Voxelspacing must have same dimensionality than the input"
        diff_vector = np.multiply(diff_vector, voxelspacing)
    center_distance = float(np.linalg.norm(diff_vector))
    return center_distance
