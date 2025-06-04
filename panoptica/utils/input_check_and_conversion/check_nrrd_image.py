import numpy as np
from importlib.util import find_spec
from pathlib import Path

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


def load_nrrd_image(image_path: str | Path) -> NRRDImage:
    try:
        readdata, header = nrrd.read(str(image_path))
    except Exception as e:
        print(f"Error reading images: {e}")
        return None
    return NRRDImage(readdata, header)


def sanity_checker_nrrd_image(
    prediction_image: NRRDImage | str | Path,
    reference_image: NRRDImage | str | Path,
    threshold: float = 1e-5,
) -> tuple[bool, tuple[np.ndarray, np.ndarray] | str]:
    """
    This function performs sanity check on 2 NRRD images.

    Args:
        prediction_image (NRRD_IMAGE): The prediction_image to be used as a baseline.
        reference_image (NRRD_IMAGE): The reference_image for comparison.
        threshold (float): The threshold for comparing the images. Default is 1e-5.

    Returns:
        tuple[bool, tuple[np.ndarray, np.ndarray] | str]: A tuple where the first element is a boolean indicating if the images pass the sanity check, and the second element is either the numpy arrays of the images or an error message.
    """
    # load if necessary
    if isinstance(prediction_image, (str, Path)):
        prediction_image = load_nrrd_image(prediction_image)
    if isinstance(reference_image, (str, Path)):
        reference_image = load_nrrd_image(reference_image)

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
    ).sum() > threshold:
        return False, "Affine Mismatch: {} vs {}".format(
            prediction_image.affine, reference_image.affine
        )

    return True, (
        np.asanyarray(
            prediction_image.array, dtype=prediction_image.array.dtype
        ).copy(),
        np.asanyarray(reference_image.array, dtype=reference_image.array.dtype).copy(),
    )
