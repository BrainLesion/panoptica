import sys
from importlib.util import find_spec, module_from_spec


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


class _InputDataTypeChecker:
    """
    This class is used to define the input data type for the sanity checker.
    It defines the supporting file endings, required package names, and the sanity check handler
    """

    __supported_file_endings: list[str] = []
    __sanity_check_handler: callable = None

    def __init__(
        self,
        supported_file_endings: list[str],
        required_package_names: list[str],
        sanity_check_handler: callable,
    ):
        self.__supported_file_endings = supported_file_endings
        self.__sanity_check_handler = sanity_check_handler
        self.__missing_packages = []
        for package_name in required_package_names:
            if not _is_package_available(package_name):
                self.__missing_packages.append(package_name)
        if not callable(self.__sanity_check_handler):
            raise TypeError("Sanity check handler must be a callable function.")

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

    def __call__(self, data1, data2, **kwds) -> bool:
        return self.__sanity_check_handler(data1, data2, **kwds)
