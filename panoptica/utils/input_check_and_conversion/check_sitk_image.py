import numpy as np
from importlib.util import find_spec
from pathlib import Path

# Optional sitk import
sitk_spec = find_spec("SimpleITK")
if sitk_spec is not None:
    import SimpleITK as sitk


def load_sitk_image(image_path: str | Path) -> sitk.Image:
    try:
        image = sitk.ReadImage(image_path)
    except Exception as e:
        print(f"Error reading images: {e}")
        return None
    return image


def sanity_checker_sitk_image(
    prediction_image: sitk.Image | str | Path,
    reference_image: sitk.Image | str | Path,
    threshold: float = 1e-5,
) -> tuple[bool, tuple[np.ndarray, np.ndarray] | str]:
    """
    This function performs sanity check on 2 SimpleITK images.

    Args:
        image_baseline (sitk.Image): The first image to be used as a baseline.
        image_compare (sitk.Image): The second image for comparison.

    Returns:
        bool: True if the images pass the sanity check, False otherwise.
    """
    # load if necessary
    if isinstance(prediction_image, (str, Path)):
        prediction_image = load_sitk_image(prediction_image)
    if isinstance(reference_image, (str, Path)):
        reference_image = load_sitk_image(reference_image)

    # assert correct datatype
    if not isinstance(prediction_image, sitk.Image) or not isinstance(reference_image, sitk.Image):
        return False, "Input images must be of type sitk.Image"

    # start necessary comparisons
    # dimensions need to be exact
    if prediction_image.GetDimension() != reference_image.GetDimension():
        return False, "Dimension Mismatch: {} vs {}".format(prediction_image.GetDimension(), reference_image.GetDimension())
    # size need to be exact
    if prediction_image.GetSize() != reference_image.GetSize():
        return False, "Size Mismatch: {} vs {}".format(prediction_image.GetSize(), reference_image.GetSize())

    # origin, direction, and spacing need to be "similar" enough
    # this is needed because different packages use different precisions for metadata
    if (np.array(prediction_image.GetOrigin()) - np.array(reference_image.GetOrigin())).sum() > threshold:
        return False, "Origin Mismatch: {} vs {}".format(prediction_image.GetOrigin(), reference_image.GetOrigin())
    if (np.array(prediction_image.GetSpacing()) - np.array(reference_image.GetSpacing())).sum() > threshold:
        return False, "Spacing Mismatch: {} vs {}".format(prediction_image.GetSpacing(), reference_image.GetSpacing())
    if (np.array(prediction_image.GetDirection()) - np.array(reference_image.GetDirection())).sum() > threshold:
        return False, "Direction Mismatch: {} vs {}".format(prediction_image.GetDirection(), reference_image.GetDirection())

    # check if the number of components is the same - this is needed for multi-channel or vector images
    if prediction_image.GetNumberOfComponentsPerPixel() != reference_image.GetNumberOfComponentsPerPixel():
        return False, "Number of Components Mismatch: {} vs {}".format(
            prediction_image.GetNumberOfComponentsPerPixel(), reference_image.GetNumberOfComponentsPerPixel()
        )

    return True, (sitk.GetArrayFromImage(prediction_image), sitk.GetArrayFromImage(reference_image))
