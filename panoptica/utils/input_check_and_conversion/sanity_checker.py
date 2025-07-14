from typing import Any
import numpy as np
from pathlib import Path
from warnings import warn

from panoptica.utils.constants import _Enum_Compare
from panoptica.utils.input_check_and_conversion.input_data_type_checker import (
    _InputDataTypeChecker,
)
from panoptica.utils.input_check_and_conversion.check_numpy_array import (
    sanity_checker_numpy_array,
)
from panoptica.utils.input_check_and_conversion.check_sitk_image import (
    sanity_checker_sitk_image,
)
from panoptica.utils.input_check_and_conversion.check_torch_image import (
    sanity_checker_torch_image,
)
from panoptica.utils.input_check_and_conversion.check_nibabel_image import (
    sanity_checker_nibabel_image,
)
from panoptica.utils.input_check_and_conversion.check_nrrd_image import (
    sanity_checker_nrrd_image,
)
from panoptica.utils.numpy_utils import _get_smallest_fitting_uint


class INPUTDTYPE(_Enum_Compare):
    NUMPY = _InputDataTypeChecker(
        supported_file_endings=[
            ".npy",
            ".npz",
        ],
        required_package_names=["numpy"],
        sanity_check_handler=sanity_checker_numpy_array,
    )
    SITK = _InputDataTypeChecker(
        supported_file_endings=[
            ".nii",
            ".nii.gz",
            ".mha",
            ".mhd",
            ".dcm",
            ".bmp",
            ".pic",
            ".gipl",
            ".gipl.gz",
            ".jpg",
            ".JPG",
            ".jpeg",
            ".JPEG",
            ".png",
            ".PNG",
            ".tiff",
            ".TIFF",
            ".tif",
            ".TIF",
            ".hdr",
            ".mnc",
            ".MNC",
            ".img",
            ".img.gz",
            ".vtk",
        ],
        required_package_names=["SimpleITK"],
        sanity_check_handler=sanity_checker_sitk_image,
    )
    TORCH = _InputDataTypeChecker(
        supported_file_endings=[
            ".pt",
            ".pth",
        ],
        required_package_names=["torch"],
        sanity_check_handler=sanity_checker_torch_image,
    )
    NIBABEL = _InputDataTypeChecker(
        supported_file_endings=[
            ".nii",
            ".nii.gz",
        ],
        required_package_names=["nibabel"],
        sanity_check_handler=sanity_checker_nibabel_image,
    )
    NRRD = _InputDataTypeChecker(
        supported_file_endings=[
            ".nrrd",
        ],
        required_package_names=["nrrd"],
        sanity_check_handler=sanity_checker_nrrd_image,
    )


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
) -> tuple[tuple[np.ndarray, np.ndarray], INPUTDTYPE]:
    """
    This function is a wrapper that performs sanity check on 2 images.

    Args:
        prediction (Any): The first image to be used as a baseline.
        reference (Any): The second image for comparison.

    Returns:
        tuple[np.ndarray, np.ndarray]: Will return the prediction array and reference array if the sanity check passes, otherwise it raises a corresponding Exception.
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
                r, s = checker(prediction, reference)
                if not r:
                    raise ValueError(
                        f"Sanity check failed for {inputdtype.name}: {s}. Please check the input files."
                    )
                return post_check(s), inputdtype
            elif not is_path:
                try:
                    r, s = checker(prediction, reference)
                except AssertionError as e:
                    continue
                if not r:
                    raise ValueError(
                        f"Sanity check failed for {inputdtype.name}: {s}. Please check the input files."
                    )
                else:
                    return post_check(s), inputdtype
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


def post_check(
    prediction_reference_array_pair: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """
    This function performs a post check on the sanity check result.

    Args:
        prediction_reference_array_pair (tuple[np.ndarray, np.ndarray]): The prediction and reference arrays.

    Returns:
        bool: True if the post check passes, False otherwise.
    """
    assert (
        isinstance(prediction_reference_array_pair, tuple)
        and len(prediction_reference_array_pair) == 2
    ), f"prediction_reference_array_pair must be a tuple of 2 elements. Got {type(prediction_reference_array_pair), len(prediction_reference_array_pair)}"
    assert isinstance(prediction_reference_array_pair[0], np.ndarray) and isinstance(
        prediction_reference_array_pair[1], np.ndarray
    ), f"prediction_reference_array_pair must be a tuple of 2 numpy arrays. Got {type(prediction_reference_array_pair[0]), type(prediction_reference_array_pair[0])}"

    min_value = min(
        prediction_reference_array_pair[0].min(),
        prediction_reference_array_pair[1].min(),
    )
    assert (
        min_value >= 0
    ), "There are negative values in the segmentation maps. This is not allowed!"

    if not np.issubdtype(
        prediction_reference_array_pair[0].dtype, np.integer
    ) or not np.issubdtype(prediction_reference_array_pair[1].dtype, np.integer):
        warn(
            "The input arrays are not of integer type. This may lead to unexpected behavior in the segmentation maps.",
            UserWarning,
        )

    max_value = max(
        prediction_reference_array_pair[0].max(),
        prediction_reference_array_pair[1].max(),
    )
    dtype = _get_smallest_fitting_uint(max_value)
    prediction_reference_array_pair = (
        prediction_reference_array_pair[0].astype(dtype),
        prediction_reference_array_pair[1].astype(dtype),
    )

    return prediction_reference_array_pair
