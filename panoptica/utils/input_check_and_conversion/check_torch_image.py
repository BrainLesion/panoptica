import numpy as np
from importlib.util import find_spec
from pathlib import Path
from panoptica.utils.input_check_and_conversion.check_numpy_array import (
    sanity_checker_numpy_array,
)

# Optional sitk import
_spec = find_spec("torch")
if _spec is not None:
    import torch


def load_torch_image(image_path: str | Path) -> torch.Tensor:
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
    prediction_image: torch.Tensor | str | Path,
    reference_image: torch.Tensor | str | Path,
) -> tuple[bool, tuple[np.ndarray, np.ndarray] | str]:
    """
    This function performs sanity check on 2 Torch tensors by converting them to numpy arrays and using that check.

    Args:
        prediction_image (torch.Tensor): The prediction_image.
        reference_image (torch.Tensor): The reference_image.

    Returns:
        tuple[bool, tuple[np.ndarray, np.ndarray] | str]: A tuple where the first element is a boolean indicating if the images pass the sanity check, and the second element is either the numpy arrays of the images or an error message.
    """
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
