import numpy as np
from skimage.morphology import skeletonize

try:
    from skimage.morphology import skeletonize_3d
except ImportError:
    # In newer skimage versions (0.23+), skeletonize handles both 2D and 3D
    skeletonize_3d = None


def _get_skeleton(array: np.ndarray) -> np.ndarray:
    """
    Get skeleton for 2D or 3D array.

    Args:
        array (np.ndarray): Input array (2D or 3D).

    Returns:
        np.ndarray: Skeletonized array.
    """
    if array.ndim == 2:
        return skeletonize(array)
    elif array.ndim == 3:
        if skeletonize_3d is not None:
            # Older versions: use dedicated 3D function
            return skeletonize_3d(array)
        else:
            # Newer versions: skeletonize handles 3D automatically
            return skeletonize(array)
    else:
        raise ValueError(f"Unsupported array dimension: {array.ndim}")


def cl_score(volume: np.ndarray, skeleton: np.ndarray):
    """Computes the skeleton volume overlap

    Args:
        volume (np.ndarray): volume
        skeleton (np.ndarray): skeleton

    Returns:
        _type_: skeleton overlap
    """
    return np.sum(volume * skeleton) / np.sum(skeleton)


def _compute_centerline_dice(
    ref_labels: np.ndarray,
    pred_labels: np.ndarray,
    ref_instance_idx: int | None = None,
    pred_instance_idx: int | None = None,
) -> float:
    """Compute the centerline Dice (clDice) coefficient between a specific pair of instances.

    Args:
        ref_labels (np.ndarray): Reference instance labels.
        pred_labels (np.ndarray): Prediction instance labels.
        ref_instance_idx (int): Index of the reference instance.
        pred_instance_idx (int): Index of the prediction instance.

    Returns:
        float: clDice coefficient
    """
    if ref_instance_idx is None and pred_instance_idx is None:
        return _compute_centerline_dice_coefficient(
            reference=ref_labels,
            prediction=pred_labels,
        )
    ref_instance_mask = ref_labels == ref_instance_idx
    pred_instance_mask = pred_labels == pred_instance_idx
    return _compute_centerline_dice_coefficient(
        reference=ref_instance_mask,
        prediction=pred_instance_mask,
    )


def _compute_centerline_dice_coefficient(
    reference: np.ndarray,
    prediction: np.ndarray,
    *args,
) -> float:
    ndim = reference.ndim
    assert 2 <= ndim <= 3, "clDice only implemented for 2D or 3D"
    tprec = cl_score(prediction, _get_skeleton(reference))
    tsens = cl_score(reference, _get_skeleton(prediction))

    return 2 * tprec * tsens / (tprec + tsens)
