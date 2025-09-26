import sys
from importlib.util import find_spec
import numpy as np
from abc import abstractmethod, ABC
from pathlib import Path


def _is_package_available(package_name: str) -> bool:
    """
    Check if a package is available in the current environment.

    Args:
        package_name (str): The name of the package to check.

    Returns:
        bool: True if the package is available, False otherwise.
    """
    if package_name in sys.modules:
        return True

    # Check if the package can be imported
    spec = find_spec(package_name)
    if spec is not None:
        # import it to be able to access it
        # module = module_from_spec(spec)
        # sys.modules[package_name] = module
        # spec.loader.exec_module(module)
        return True
    return False


class _InputDataTypeChecker(ABC):
    """
    This class is used to define an input data type.
    It defines the supporting file endings, required package names, and the functions for loading images of that data type from a path, converting it to numpy array, extracting metadata and sanity checking the input.
    """

    __supported_file_endings: list[str] = []

    def __init__(
        self,
        supported_file_endings: list[str],
        required_package_names: list[str],
    ):
        self.__supported_file_endings = supported_file_endings
        self.__missing_packages = []
        for package_name in required_package_names:
            if not _is_package_available(package_name):
                self.__missing_packages.append(package_name)

    @property
    def missing_packages(self) -> list[str]:
        """
        Get the list of missing packages.

        Returns:
            list[str]: List of missing packages.
        """
        return self.__missing_packages

    @property
    def supported_file_endings(self) -> list[str]:
        """
        Get the supported file endings.

        Returns:
            list[str]: List of supported file endings.
        """
        return self.__supported_file_endings

    def are_requirements_fulfilled(self) -> bool:
        """
        Check if all required packages are available.

        Returns:
            bool: True if all required packages are available, False otherwise.
        """
        return len(self.__missing_packages) == 0

    def __call__(
        self, prediction, reference, **kwargs
    ) -> tuple[bool, str, tuple[object, object]]:
        """Tries to load the images of that data type and then checks if they are compatible.

        Args:
            prediction (_type_): Prediction image or path to prediction image.
            reference (_type_): Reference image or path to reference image.

        Returns:
            tuple[bool, str, tuple[object, object]]: A tuple containing a boolean indicating success, an (error) message string, and a tuple of the loaded images.
        """

        # load from path if necessary
        if isinstance(prediction, (str, Path)):
            prediction = self.load_image_from_path(prediction)
        if isinstance(reference, (str, Path)):
            reference = self.load_image_from_path(reference)

        assert (
            prediction is not None and reference is not None
        ), "Could not load images from the given paths."

        c, msg = self.sanity_check_images(prediction, reference)
        return c, msg, (prediction, reference)

    @abstractmethod
    def load_image_from_path(self, image_path: str | Path) -> object | None:
        pass

    @abstractmethod
    def sanity_check_images(
        self, prediction_image, reference_image, *args, **kwargs
    ) -> tuple[bool, str]:
        pass

    @abstractmethod
    def convert_to_numpy_array(self, image) -> np.ndarray:
        pass

    @abstractmethod
    def extract_metadata_from_image(self, image) -> dict:
        pass
