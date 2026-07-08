"""Converter for SimpleITK images."""

from __future__ import annotations

from typing import Any

import numpy as np

from panoptica.core.errors import BackendUnavailable, InputValidationError


def convert_to_numpy(obj: Any) -> tuple[np.ndarray, tuple[float, ...] | None]:
    """Convert a SimpleITK image to a numpy array with spacing.

    Args:
        obj: A SimpleITK Image object.

    Returns:
        (array, spacing): The numpy array and voxel spacing (in mm), or None if uniform 1.0.

    Raises:
        BackendUnavailable: If SimpleITK is not installed.
        InputValidationError: If obj is not a SimpleITK Image or fails conversion.
    """
    try:
        import SimpleITK as sitk
    except ImportError as e:
        raise BackendUnavailable(
            "SimpleITK is not installed. Install it with: pip install SimpleITK"
        ) from e

    if not isinstance(obj, sitk.Image):
        raise InputValidationError(
            f"Expected SimpleITK Image, got {type(obj).__name__}. "
            "Pass a SimpleITK Image object or use the appropriate converter."
        )

    # Extract array; sitk.GetArrayFromImage returns (Z, Y, X) for 3D or (Y, X) for 2D
    array = sitk.GetArrayFromImage(obj)

    if not np.issubdtype(array.dtype, np.integer):
        raise InputValidationError(
            f"SimpleITK image must have integer dtype, got {array.dtype}. "
            "Label and mask arrays must contain integer values."
        )

    spacing_tuple = obj.GetSpacing()
    spacing = tuple(float(s) for s in spacing_tuple) if spacing_tuple else None

    return array, spacing
