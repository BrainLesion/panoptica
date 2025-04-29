from typing import Union
import SimpleITK as sitk
import numpy as np


def sanity_checker_with_arrays(
    image_array_baseline: np.ndarray,
    image_array_compare: np.ndarray,
) -> bool:
    """
    This function performs sanity check on 2 image arrays.

    Args:
        image_array_baseline (np.ndarray): The first image array to be used as a baseline.
        image_array_compare (np.ndarray): The second image array for comparison.
        threshold (float): Threshold for checking image data consistency. This is needed because different packages use different precisions for metadata.

    Returns:
        bool: True if the images pass the sanity check, False otherwise.
    """
    # dimensions need to be exact
    if image_array_baseline.shape != image_array_compare.shape:
        return False

    return True


def sanity_checker_with_images(
    image_baseline: sitk.Image, image_compare: sitk.Image, threshold: float = 1e-5
) -> bool:
    """
    This function performs sanity check on 2 SimpleITK images.

    Args:
        image_baseline (sitk.Image): The first image to be used as a baseline.
        image_compare (sitk.Image): The second image for comparison.

    Returns:
        bool: True if the images pass the sanity check, False otherwise.
    """
    # start necessary comparisons
    # dimensions need to be exact
    if image_baseline.GetDimension() != image_compare.GetDimension():
        return False
    # size need to be exact
    if image_baseline.GetSize() != image_compare.GetSize():
        return False

    # origin, direction, and spacing need to be "similar" enough
    # this is needed because different packages use different precisions for metadata
    if (
        np.array(image_baseline.GetOrigin()) - np.array(image_compare.GetOrigin())
    ).sum() > threshold:
        return False
    if (
        np.array(image_baseline.GetSpacing()) - np.array(image_compare.GetSpacing())
    ).sum() > threshold:
        return False
    if (
        np.array(image_baseline.GetDirection()) - np.array(image_compare.GetDirection())
    ).sum() > threshold:
        return False

    # check if the number of components is the same - this is needed for multi-channel or vector images
    if (
        image_baseline.GetNumberOfComponentsPerPixel()
        != image_compare.GetNumberOfComponentsPerPixel()
    ):
        return False

    return True


def sanity_checker_with_files(
    image_file_baseline: str, image_file_compare: str, threshold: float = 1e-5
) -> bool:
    """
    This function performs sanity check on 2 image files WITHOUT loading images into memory.

    Args:
        image_file_baseline (str): Path to the first image file to be used as a baseline.
        image_file_compare (str): Path to the second image file for comparison.
        threshold (float): Threshold for checking image data consistency. This is needed because different packages use different precisions for metadata.

    Returns:
        bool: True if the images pass the sanity check, False otherwise.
    """
    try:
        image_baseline = sitk.ReadImage(image_file_baseline)
        image_compare = sitk.ReadImage(image_file_compare)
    except Exception as e:
        print(f"Error reading images: {e}")
        return False

    return sanity_checker_with_images(
        image_baseline, image_compare, threshold=threshold
    )


def sanity_checker(
    reference: Union[str, sitk.Image, np.ndarray],
    compare: Union[str, sitk.Image, np.ndarray],
    threshold: float = 1e-5,
) -> bool:
    """
    This function is a wrapper that performs sanity check on 2 images.

    Args:
        reference (Union[str, sitk.Image, np.ndarray]): The first image to be used as a baseline.
        compare (Union[str, sitk.Image, np.ndarray]): The second image for comparison.
        threshold (float): Threshold for checking image data consistency. This is needed because different packages use different precisions for metadata.

    Returns:
        bool: True if the images pass the sanity check, False otherwise.
    """
    if isinstance(reference, str) and isinstance(compare, str):
        return sanity_checker_with_files(reference, compare, threshold=threshold)
    elif isinstance(reference, sitk.Image) and isinstance(compare, sitk.Image):
        return sanity_checker_with_images(reference, compare, threshold=threshold)
    elif isinstance(reference, sitk.Image) and isinstance(compare, str):
        try:
            compare_img = sitk.ReadImage(compare)
        except Exception as e:
            print(f"Error reading image: {e}")
            return False
        return sanity_checker_with_images(reference, compare_img, threshold=threshold)
    elif isinstance(reference, str) and isinstance(compare, sitk.Image):
        try:
            reference_img = sitk.ReadImage(reference)
        except Exception as e:
            print(f"Error reading image: {e}")
            return False
        return sanity_checker_with_images(reference_img, compare, threshold=threshold)
    elif isinstance(reference, np.ndarray) and isinstance(compare, np.ndarray):
        return sanity_checker_with_arrays(reference, compare)
    else:
        raise ValueError("Unsupported input types for reference and compare.")
