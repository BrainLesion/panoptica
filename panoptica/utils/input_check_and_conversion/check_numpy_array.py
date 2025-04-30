import numpy as np
from pathlib import Path


def sanity_checker_numpy_array(
    prediction_arr: np.ndarray | str | Path,
    reference_arr: np.ndarray | str | Path,
) -> tuple[bool, str | tuple[np.ndarray, np.ndarray]]:
    """
    This function performs sanity check on 2 image arrays.

    Args:
        image_array_baseline (np.ndarray): The first image array to be used as a baseline.
        image_array_compare (np.ndarray): The second image array for comparison.

    Returns:
        bool: True if the images pass the sanity check, False otherwise.
    """
    # load if necessary
    if isinstance(prediction_arr, (str, Path)):
        prediction_arr = np.load(prediction_arr)
    if isinstance(reference_arr, (str, Path)):
        reference_arr = np.load(reference_arr)

    # assert correct datatype
    if not isinstance(prediction_arr, np.ndarray) or not isinstance(reference_arr, np.ndarray):
        return False, "Input images must be of type np.ndarray"

    # dimensions need to be exact
    if prediction_arr.shape != reference_arr.shape:
        return False, "Dimension Mismatch: {} vs {}".format(prediction_arr.shape, reference_arr.shape)

    return True, (prediction_arr, reference_arr)
