import numpy as np
from pathlib import Path
from panoptica.utils.input_check_and_conversion.input_data_type_checker import (
    _InputDataTypeChecker,
)


class NumpyImageChecker(_InputDataTypeChecker):
    def __init__(self):
        super().__init__(
            supported_file_endings=[
                ".npy",
                ".npz",
            ],
            required_package_names=["numpy"],
        )

    def load_image_from_path(self, image_path: str | Path) -> np.ndarray:
        return np.load(image_path)

    def sanity_check_images(
        self, prediction_image: np.ndarray, reference_image: np.ndarray, *args, **kwargs
    ) -> tuple[bool, str]:
        return _sanity_check_images(prediction_image, reference_image)

    def convert_to_numpy_array(self, image: np.ndarray) -> np.ndarray:
        return image

    def extract_metadata_from_image(self, image: np.ndarray) -> dict:
        """
        Extracts basic metadata from a numpy array.
        """
        return {}


def _sanity_check_images(
    prediction_image: np.ndarray, reference_image: np.ndarray, *args, **kwargs
) -> tuple[bool, str]:
    # assert correct datatype
    assert isinstance(prediction_image, np.ndarray) and isinstance(
        reference_image, np.ndarray
    ), "prediction and reference must be of type np.ndarray."

    # dimensions need to be exact
    if prediction_image.shape != reference_image.shape:
        return False, "Dimension Mismatch: {} vs {}".format(
            prediction_image.shape, reference_image.shape
        )

    return True, ""
