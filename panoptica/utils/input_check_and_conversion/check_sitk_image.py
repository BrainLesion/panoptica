import numpy as np
from importlib.util import find_spec
from pathlib import Path
from panoptica.utils.input_check_and_conversion.input_data_type_checker import (
    _InputDataTypeChecker,
)

# Optional sitk import
sitk_spec = find_spec("SimpleITK")
if sitk_spec is not None:
    import SimpleITK as sitk


class SITKImageChecker(_InputDataTypeChecker):
    def __init__(self):
        super().__init__(
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
        )
        self.threshold = 1e-5

    def load_image_from_path(self, image_path: str | Path) -> sitk.Image | None:
        try:
            image = sitk.ReadImage(image_path)
        except Exception as e:
            print(f"Error reading images: {e}")
            return None
        return image

    def sanity_check_images(
        self, prediction_image: sitk.Image, reference_image: sitk.Image, *args, **kwargs
    ) -> tuple[bool, str]:
        # assert correct datatype
        assert isinstance(prediction_image, sitk.Image) and isinstance(
            reference_image, sitk.Image
        ), "Input images must be of type sitk.Image"

        # dimensions need to be exact
        if prediction_image.GetDimension() != reference_image.GetDimension():
            return False, "Dimension Mismatch: {} vs {}".format(
                prediction_image.GetDimension(), reference_image.GetDimension()
            )

        # size need to be exact
        if prediction_image.GetSize() != reference_image.GetSize():
            return False, "Size Mismatch: {} vs {}".format(
                prediction_image.GetSize(), reference_image.GetSize()
            )

        # origin, direction, and spacing need to be "similar" enough
        # this is needed because different packages use different precisions for metadata
        if (
            np.array(prediction_image.GetOrigin())
            - np.array(reference_image.GetOrigin())
        ).sum() > self.threshold:
            return False, "Origin Mismatch: {} vs {}".format(
                prediction_image.GetOrigin(), reference_image.GetOrigin()
            )

        if (
            np.array(prediction_image.GetSpacing())
            - np.array(reference_image.GetSpacing())
        ).sum() > self.threshold:
            return False, "Spacing Mismatch: {} vs {}".format(
                prediction_image.GetSpacing(), reference_image.GetSpacing()
            )

        if (
            np.array(prediction_image.GetDirection())
            - np.array(reference_image.GetDirection())
        ).sum() > self.threshold:
            return False, "Direction Mismatch: {} vs {}".format(
                prediction_image.GetDirection(), reference_image.GetDirection()
            )

        # check if the number of components is the same - this is needed for multi-channel or vector images
        if (
            prediction_image.GetNumberOfComponentsPerPixel()
            != reference_image.GetNumberOfComponentsPerPixel()
        ):
            return False, "Number of Components Mismatch: {} vs {}".format(
                prediction_image.GetNumberOfComponentsPerPixel(),
                reference_image.GetNumberOfComponentsPerPixel(),
            )

        return True, ""

    def convert_to_numpy_array(self, image: sitk.Image) -> np.ndarray:
        return sitk.GetArrayFromImage(image)

    def extract_metadata_from_image(self, image: sitk.Image) -> dict:
        metadata = {
            "voxelspacing": image.GetSpacing(),
        }
        return metadata
