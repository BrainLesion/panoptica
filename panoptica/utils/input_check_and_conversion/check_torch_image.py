import numpy as np
from importlib.util import find_spec
from pathlib import Path
from typing import Union, TYPE_CHECKING
from panoptica.utils.input_check_and_conversion.check_numpy_array import (
    sanity_checker_numpy_array,
)

# Optional torch import
_spec = find_spec("torch")
if _spec is not None:
    import torch
else:
    torch = None

if TYPE_CHECKING:
    import torch


def load_torch_image(image_path: Union[str, Path]):
    if torch is None:
        raise ImportError(
            "torch is not available. Please install torch to use this functionality."
        )
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


def sanity_checker_torch_image(
    prediction_image: Union["torch.Tensor", str, Path],
    reference_image: Union["torch.Tensor", str, Path],
) -> tuple[bool, Union[tuple[np.ndarray, np.ndarray], str]]:
    """
    This function performs sanity check on 2 Torch tensors by converting them to numpy arrays and using that check.

    Args:
        prediction_image (torch.Tensor): The prediction_image.
        reference_image (torch.Tensor): The reference_image.

    Returns:
        tuple[bool, tuple[np.ndarray, np.ndarray] | str]: A tuple where the first element is a boolean indicating if the images pass the sanity check, and the second element is either the numpy arrays of the images or an error message.
    """
    if torch is None:
        raise ImportError(
            "torch is not available. Please install torch to use this functionality."
        )

    # load if necessary
    if isinstance(prediction_image, (str, Path)):
        prediction_image = load_torch_image(prediction_image)
    if isinstance(reference_image, (str, Path)):
        reference_image = load_torch_image(reference_image)

    # assert correct datatype
    assert isinstance(prediction_image, torch.Tensor) and isinstance(
        reference_image, torch.Tensor
    ), "Input images must be of type torch.Tensor"

    return sanity_checker_numpy_array(
        prediction_image.numpy(),
        reference_image.numpy(),
    )
