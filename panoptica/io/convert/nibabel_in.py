"""Converter for Nibabel Nifti images."""

from __future__ import annotations

from typing import Any

import numpy as np

from panoptica.core.errors import BackendUnavailable, InputValidationError


def convert_to_numpy(obj: Any) -> tuple[np.ndarray, tuple[float, ...] | None]:
    """Convert a Nibabel Nifti image to a numpy array with spacing.

    Args:
        obj: A Nibabel Nifti1Image or Nifti2Image object.

    Returns:
        (array, spacing): The numpy array and voxel spacing (in mm), or None if uniform 1.0.

    Raises:
        BackendUnavailable: If nibabel is not installed.
        InputValidationError: If obj is not a Nibabel image or fails conversion.
    """
    try:
        import nibabel as nib
    except ImportError as e:
        raise BackendUnavailable(
            "nibabel is not installed. Install it with: pip install nibabel"
        ) from e

    if not isinstance(obj, (nib.Nifti1Image, nib.Nifti2Image)):
        raise InputValidationError(
            f"Expected Nibabel Nifti1Image or Nifti2Image, got {type(obj).__name__}. "
            "Pass a Nibabel image object or use the appropriate converter."
        )

    # Extract array and ensure it's a copy (not a lazy-loaded proxy)
    array = np.asanyarray(obj.dataobj, dtype=obj.dataobj.dtype).copy()  # pyrefly: ignore

    if not np.issubdtype(array.dtype, np.integer):
        raise InputValidationError(
            f"Nibabel image must have integer dtype, got {array.dtype}. "
            "Label and mask arrays must contain integer values."
        )

    # Extract spacing from header; get_zooms() returns spacing (in mm)
    spacing_tuple = obj.header.get_zooms()
    spacing = tuple(float(s) for s in spacing_tuple) if spacing_tuple else None

    return array, spacing
