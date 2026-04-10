import numpy as np
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Union
from panoptica.utils.input_check_and_conversion.check_numpy_array import (
    _sanity_check_images,
)
from panoptica.utils.input_check_and_conversion.input_data_type_checker import (
    _InputDataTypeChecker,
)

# Optional torch import
_spec = find_spec("torch")
if _spec is not None:
    import torch
else:
    torch = None

if TYPE_CHECKING:
    import torch


class TorchImageChecker(_InputDataTypeChecker):
    def __init__(self):
        super().__init__(
            supported_file_endings=[
                ".pt",
                ".pth",
            ],
            required_package_names=["torch"],
        )

    def load_image_from_path(
        self, image_path: Union[str, Path]
    ) -> Union["torch.Tensor", None]:
        try:
            image = torch.load(
                image_path,
                weights_only=True,
                map_location="cpu",
            )
        except Exception as e:
            print(f"Error reading images: {e}")
            return None
        return image

    def sanity_check_images(
        self,
        prediction_image: "torch.Tensor",
        reference_image: "torch.Tensor",
        *args,
        **kwargs,
    ) -> tuple[bool, str]:
        # assert correct datatype
        assert isinstance(prediction_image, torch.Tensor) and isinstance(
            reference_image, torch.Tensor
        ), "Input images must be of type torch.Tensor"

        return _sanity_check_images(
            prediction_image.numpy(),
            reference_image.numpy(),
        )

    def convert_to_numpy_array(self, image: "torch.Tensor") -> np.ndarray:
        return image.numpy()

    def extract_metadata_from_image(self, image: "torch.Tensor") -> dict:
        """
        Extracts metadata from a torch.Tensor image.
        Returns a dictionary.
        If no further metadata is available, this is documented here.
        """
        return {}
