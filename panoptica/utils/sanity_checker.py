import SimpleITK as sitk
import numpy as np


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
    # initialize image readers
    file_reader_baseline = sitk.ImageFileReader()
    file_reader_baseline.SetFileName(image_file_baseline)
    file_reader_baseline.ReadImageInformation()

    file_reader_compare = sitk.ImageFileReader()
    file_reader_compare.SetFileName(image_file_compare)
    file_reader_compare.ReadImageInformation()

    # start necessary comparisons
    # dimensions need to be exact
    if file_reader_baseline.GetDimension() != file_reader_compare.GetDimension():
        return False
    # size need to be exact
    if file_reader_baseline.GetSize() != file_reader_compare.GetSize():
        return False

    # origin, direction, and spacing need to be similar enough
    if (
        np.array(file_reader_baseline.GetOrigin())
        - np.array(file_reader_compare.GetOrigin())
    ).sum() > threshold:
        return False
    if (
        np.array(file_reader_baseline.GetSpacing())
        - np.array(file_reader_compare.GetSpacing())
    ).sum() > threshold:
        return False
    if (
        np.array(file_reader_baseline.GetDirection())
        - np.array(file_reader_compare.GetDirection())
    ).sum() > threshold:
        return False

    return True
