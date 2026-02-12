import numpy as np
from importlib.util import find_spec
from pathlib import Path
from panoptica.utils.input_check_and_conversion.input_data_type_checker import (
    _InputDataTypeChecker,
)

# Optional sitk import
spec = find_spec("nrrd")
if spec is not None:
    import nrrd


class NRRDImage:
    """
    A class to represent a NRRD image.
    """

    def __init__(self, data: np.ndarray, header: dict):
        self.__data: np.ndarray = data
        self.__header: dict = header
        try:
            self.__space_directions: np.ndarray = np.array(header["space directions"])
            self.__space_origin: np.ndarray = np.array(header["space origin"])
            ndim: int = header["dimension"]
        except KeyError as e:
            raise KeyError(f"Missing key in header: {e}")

        if self.__space_directions.shape != (ndim, ndim):
            raise ValueError(
                f"Expected 'space directions' to be a nxn matrix. n = {ndim} is not {self.__space_directions.shape}",
                self.__space_directions,
            )
        if self.__space_origin.shape != (ndim,):
            raise ValueError(
                "Expected 'space origin' to be a n-element vector. n = ",
                ndim,
                "is not",
                self.__space_origin.shape,
            )
        affine = np.eye(ndim + 1)  # Initialize 4x4 identity matrix
        affine[:ndim, :ndim] = self.__space_directions  # Set rotation and scaling
        affine[:ndim, ndim] = self.__space_origin  # Set translation
        self.__affine = affine

    @property
    def shape(self):
        return self.__data.shape

    @property
    def affine(self):
        return self.__affine

    @property
    def array(self):
        return self.__data

    @property
    def header(self):
        return self.__header

    @property
    def space_directions(self):
        return self.__space_directions

    @property
    def space_origin(self):
        return self.__space_origin


class NRRDImageChecker(_InputDataTypeChecker):
    def __init__(self):
        super().__init__(
            supported_file_endings=[
                ".nrrd",
            ],
            required_package_names=["nrrd"],
        )
        self.threshold = 1e-5

    def load_image_from_path(self, image_path: str | Path) -> NRRDImage | None:
        try:
            readdata, header = nrrd.read(str(image_path))
        except Exception as e:
            print(f"Error reading images: {e}")
            return None
        return NRRDImage(readdata, header)

    def sanity_check_images(
        self, prediction_image: NRRDImage, reference_image: NRRDImage, *args, **kwargs
    ) -> tuple[bool, str]:
        # assert correct datatype
        assert isinstance(prediction_image, NRRDImage) and isinstance(
            reference_image, NRRDImage
        ), "Input images must be of type NRRD_IMAGE"
        # start necessary comparisons
        # dimensions need to be exact
        if prediction_image.shape != reference_image.shape:
            return False, "Dimension Mismatch: {} vs {}".format(
                prediction_image.shape, reference_image.shape
            )

        # check if the affine matrices are similar
        if (
            np.array(prediction_image.affine) - np.array(reference_image.affine)
        ).sum() > self.threshold:
            return False, "Affine Mismatch: {} vs {}".format(
                prediction_image.affine, reference_image.affine
            )

        return True, ""

    def convert_to_numpy_array(self, image: NRRDImage) -> np.ndarray:
        return np.asanyarray(image.array, dtype=image.array.dtype).copy()

    def extract_metadata_from_image(self, image: NRRDImage) -> dict:
        voxel_spacing = [np.linalg.norm(v) for v in image.space_directions]
        return {"voxelspacing": voxel_spacing}
