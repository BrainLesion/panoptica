import numpy as np
from importlib.util import find_spec
from pathlib import Path
from panoptica.utils.input_check_and_conversion.input_data_type_checker import (
    _InputDataTypeChecker,
)

# Optional sitk import
spec = find_spec("nibabel")
if spec is not None:
    import nibabel as nib


class NibabelImageChecker(_InputDataTypeChecker):
    def __init__(self):
        super().__init__(
            supported_file_endings=[".nii", ".nii.gz"],
            required_package_names=["nibabel"],
        )
        self.threshold: float = 1e-5

    def load_image_from_path(self, image_path: str | Path) -> nib.Nifti1Image | None:
        try:
            image = nib.load(image_path)
        except Exception as e:
            print(f"Error reading images: {e}")
            return None
        return image

    def sanity_check_images(
        self,
        prediction_image: nib.Nifti1Image,
        reference_image: nib.Nifti1Image,
        *args,
        **kwargs,
    ) -> tuple[bool, str]:
        """
        This function performs sanity check on 2 Nibabel Nifti1Image objects.

        Args:
            prediction_image (nib.Nifti1Image): The prediction_image to be used as a baseline.
            reference_image (nib.Nifti1Image): The reference_image for comparison.
            threshold (float): The threshold for comparing the images. Default is 1e-5.

        Returns:
            tuple[bool, tuple[np.ndarray, np.ndarray] | str]: A tuple where the first element is a boolean indicating if the images pass the sanity check, and the second element is either the numpy arrays of the images or an error message.
        """
        # assert correct datatype
        assert isinstance(prediction_image, nib.Nifti1Image) and isinstance(
            reference_image, nib.Nifti1Image
        ), "Input images must be of type nibabel.Nifti1Image"
        # start necessary comparisons
        # dimensions need to be exact
        if prediction_image.shape != reference_image.shape:
            return (
                False,
                "Dimension Mismatch: {} vs {}".format(
                    prediction_image.shape, reference_image.shape
                ),
            )

        # check if the affine matrices are similar
        if (
            np.array(prediction_image.affine) - np.array(reference_image.affine)
        ).sum() > self.threshold:
            return (
                False,
                "Affine Mismatch: {} vs {}".format(
                    prediction_image.affine, reference_image.affine
                ),
            )

        return True, ""

    def convert_to_numpy_array(self, image: nib.Nifti1Image) -> np.ndarray:
        return np.asanyarray(image.dataobj, dtype=image.dataobj.dtype).copy()

    def extract_metadata_from_image(self, image: nib.Nifti1Image) -> dict:
        """
        Extract metadata from a Nibabel Nifti1Image object.

        Args:
            image (nib.Nifti1Image): The Nibabel image object.

        Returns:
            dict: A dictionary containing the extracted metadata.
        """
        if not isinstance(image, nib.Nifti1Image):
            raise TypeError("Input must be a Nibabel Nifti1Image object.")

        metadata = {
            "voxelspacing": image.header.get_zooms(),
        }
        return metadata
