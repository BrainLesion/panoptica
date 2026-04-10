from typing import Any
import numpy as np
from pathlib import Path
from warnings import warn

from panoptica.utils.constants import _Enum_Compare
from panoptica.utils.input_check_and_conversion.input_data_type_checker import (
    _InputDataTypeChecker,
)
from panoptica.utils.input_check_and_conversion.check_numpy_array import (
    NumpyImageChecker,
)
from panoptica.utils.input_check_and_conversion.check_sitk_image import (
    SITKImageChecker,
)
from panoptica.utils.input_check_and_conversion.check_torch_image import (
    TorchImageChecker,
)
from panoptica.utils.input_check_and_conversion.check_nibabel_image import (
    NibabelImageChecker,
)
from panoptica.utils.input_check_and_conversion.check_nrrd_image import (
    NRRDImageChecker,
)
from panoptica.utils.numpy_utils import _get_smallest_fitting_uint


class INPUTDTYPE(_Enum_Compare):
    NUMPY = NumpyImageChecker()
    SITK = SITKImageChecker()
    NIBABEL = NibabelImageChecker()
    TORCH = TorchImageChecker()
    NRRD = NRRDImageChecker()


def print_available_package_to_input_handlers():
    for d in INPUTDTYPE:
        if d.value.are_requirements_fulfilled():
            print(f"{d.name} is available for input handling.")
        else:
            print(
                f"{d.name} is not available for input handling. Missing packages: {d.value.missing_packages}"
            )


def sanity_check_and_convert_to_array(
    prediction: Any,
    reference: Any,
) -> tuple[
    tuple[
        tuple[np.ndarray, np.ndarray],
        dict,
    ],
    INPUTDTYPE,
]:
    """
    This function is a wrapper that performs sanity check on 2 images.

    Args:
        prediction (Any): The first image to be used as a baseline.
        reference (Any): The second image for comparison.

    Returns:
        tuple[np.ndarray, np.ndarray], InputDType, dict: Will return the prediction array, reference array, the INPUTDTYPE and any metadata if the sanity check passes, otherwise it raises a corresponding Exception.
    """
    assert (
        prediction is not None and reference is not None
    ), "prediction and reference cannot be None."
    assert type(prediction) is type(
        reference
    ), "prediction and reference must be of the same type."

    is_path = isinstance(prediction, (str, Path))
    file_ending = None
    if is_path:
        prediction = Path(prediction)
        reference = Path(reference)
        file_ending = "".join(prediction.suffixes)
        assert file_ending == "".join(
            reference.suffixes
        ), f"prediction and reference must have the same file ending. Got {file_ending, reference.suffix}"

    missing_package_for_this = []

    for inputdtype in INPUTDTYPE:
        checker = inputdtype.value
        if checker.are_requirements_fulfilled():
            if is_path and file_ending in checker.supported_file_endings:
                r, msg, (pred, ref) = checker(prediction, reference)
                if not r:
                    raise ValueError(
                        f"Sanity check failed for {inputdtype.name}: {msg}. Please check the input files."
                    )
                return (
                    convert_to_numpy_array_and_extract_metadata(pred, ref, checker),
                    inputdtype,
                )
            elif not is_path:
                try:
                    r, msg, (pred, ref) = checker(prediction, reference)
                except AssertionError as e:
                    continue
                if not r:
                    raise ValueError(
                        f"Sanity check failed for {inputdtype.name}: {msg}. Please check the input files."
                    )
                else:
                    return (
                        convert_to_numpy_array_and_extract_metadata(pred, ref, checker),
                        inputdtype,
                    )
        elif is_path and file_ending in checker.supported_file_endings:
            missing_package_for_this.append(checker)
    if len(missing_package_for_this) > 0:
        missing_package_names = [
            checker.missing_packages for checker in missing_package_for_this
        ]
        raise ImportError(
            f"Missing packages for the given file ending {file_ending}: Any of these sets of packages is missing: {missing_package_names}. Please install the required packages."
        )

    if is_path:
        raise ValueError(
            f"Unsupported file ending {file_ending} for reference and compare. Either panoptica is not supporting it, otherwise maybe a package is missing to handle these?"
        )
    raise ValueError(
        f"Unsupported input types {type(prediction), prediction} for reference and compare. Either panoptica is not supporting it, otherwise maybe a package is missing to handle these? You can always pass as numpy arrays directly."
    )


def convert_to_numpy_array_and_extract_metadata(
    prediction: object,
    reference: object,
    checker: _InputDataTypeChecker,
):
    np_prediction = checker.convert_to_numpy_array(prediction)
    np_reference = checker.convert_to_numpy_array(reference)

    metadata_ref: dict = checker.extract_metadata_from_image(reference)
    metadata_pred: dict = checker.extract_metadata_from_image(prediction)

    assert (
        metadata_ref == metadata_pred
    ), f"Metadata of prediction and reference do not match. Got Reference={metadata_ref}, and prediction={metadata_pred}"

    return post_check(np_prediction, np_reference), metadata_ref


def post_check(
    prediction_array: np.ndarray,
    reference_array: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    This function performs a post check on the sanity check result.

    Args:
        prediction_reference_array_pair (tuple[np.ndarray, np.ndarray]): The prediction and reference arrays.

    Returns:
        bool: True if the post check passes, False otherwise.
    """
    assert isinstance(prediction_array, np.ndarray) and isinstance(
        reference_array, np.ndarray
    ), f"prediction_array and reference_array must be numpy arrays. Got {type(prediction_array), type(reference_array)}"

    min_value = min(
        prediction_array.min(),
        reference_array.min(),
    )
    assert (
        min_value >= 0
    ), "There are negative values in the segmentation maps. This is not allowed!"

    if not np.issubdtype(prediction_array.dtype, np.integer) or not np.issubdtype(
        reference_array.dtype, np.integer
    ):
        warn(
            "The input arrays are not of integer type. This may lead to unexpected behavior in the segmentation maps.",
            UserWarning,
        )

    max_value = max(
        prediction_array.max(),
        reference_array.max(),
    )
    dtype = _get_smallest_fitting_uint(max_value)

    return prediction_array.astype(dtype), reference_array.astype(dtype)
