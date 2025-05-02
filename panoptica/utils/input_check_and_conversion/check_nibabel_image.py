import numpy as np
from importlib.util import find_spec
from pathlib import Path

# Optional sitk import
spec = find_spec("nibabel")
if spec is not None:
    import nibabel as nib


def load_nibabel_image(image_path: str | Path) -> nib.Nifti1Image:
    try:
        image = nib.load(image_path)
    except Exception as e:
        print(f"Error reading images: {e}")
        return None
    return image


def sanity_checker_nibabel_image(
    prediction_image: nib.Nifti1Image | str | Path,
    reference_image: nib.Nifti1Image | str | Path,
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
        prediction_image = load_nibabel_image(prediction_image)
    if isinstance(reference_image, (str, Path)):
        reference_image = load_nibabel_image(reference_image)

    # assert correct datatype
    assert isinstance(prediction_image, nib.Nifti1Image) and isinstance(
        reference_image, nib.Nifti1Image
    ), "Input images must be of type nibabel.Nifti1Image"
    # start necessary comparisons
    # dimensions need to be exact
    if prediction_image.shape != reference_image.shape:
        return False, "Dimension Mismatch: {} vs {}".format(
            prediction_image.shape, reference_image.shape
        )

    # check if the affine matrices are similar
    if (
        np.array(prediction_image.affine) - np.array(reference_image.affine)
    ).sum() > threshold:
        return False, "Affine Mismatch: {} vs {}".format(
            prediction_image.affine, reference_image.affine
        )

    return True, (
        np.asanyarray(
            prediction_image.dataobj, dtype=prediction_image.dataobj.dtype
        ).copy(),
        np.asanyarray(
            reference_image.dataobj, dtype=reference_image.dataobj.dtype
        ).copy(),
    )
