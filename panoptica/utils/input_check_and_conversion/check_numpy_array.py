import numpy as np
from pathlib import Path


def sanity_checker_numpy_array(
    prediction_arr: np.ndarray | str | Path,
    reference_arr: np.ndarray | str | Path,
) -> tuple[bool, str | tuple[np.ndarray, np.ndarray]]:
    """
    This function performs sanity check on 2 image arrays.

    Args:
        prediction_arr (np.ndarray): The prediction_array to be used as a baseline.
        reference_arr (np.ndarray): The reference_array for comparison.

    Returns:
        tuple[bool, tuple[np.ndarray, np.ndarray] | str]: A tuple where the first element is a boolean indicating if the images pass the sanity check, and the second element is either the numpy arrays of the images or an error message.
    """
    # load if necessary
    if isinstance(prediction_arr, (str, Path)):
        prediction_arr = np.load(prediction_arr)
    if isinstance(reference_arr, (str, Path)):
        reference_arr = np.load(reference_arr)

    # assert correct datatype
    assert isinstance(prediction_arr, np.ndarray) and isinstance(
        reference_arr, np.ndarray
    ), "prediction and reference must be of type np.ndarray."

    # dimensions need to be exact
    if prediction_arr.shape != reference_arr.shape:
        return False, "Dimension Mismatch: {} vs {}".format(
            prediction_arr.shape, reference_arr.shape
        )

    return True, (prediction_arr, reference_arr)
