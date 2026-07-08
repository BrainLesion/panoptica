"""Converter for NRRD images."""

from __future__ import annotations

from typing import Any

import numpy as np

from panoptica.core.errors import BackendUnavailable, InputValidationError


def convert_to_numpy(obj: Any) -> tuple[np.ndarray, tuple[float, ...] | None]:
    """Convert a NRRD image to a numpy array with spacing.

    NRRD files carry spatial metadata (space directions) from which we extract
    voxel spacing by computing the norm of each direction vector.

    Args:
        obj: An object with nrrd metadata (data array, header dict).

    Returns:
        (array, spacing): The numpy array and voxel spacing (in mm), or None if not available.

    Raises:
        BackendUnavailable: If nrrd is not installed.
        InputValidationError: If obj is not a valid NRRD image or fails conversion.
    """
    try:
        import nrrd  # noqa: F401
    except ImportError as e:
        raise BackendUnavailable(
            "nrrd is not installed. Install it with: pip install pynrrd"
        ) from e

    # Check for NRRD-like objects: should have .array, .header (or __data, __header if private)
    if hasattr(obj, "array"):
        array = obj.array
        header = obj.header if hasattr(obj, "header") else None
    elif hasattr(obj, "__dict__") and "_NRRDImage__data" in obj.__dict__:
        # Handle private attributes from a custom NRRDImage-like class
        array = obj.__dict__["_NRRDImage__data"]
        header = obj.__dict__.get("_NRRDImage__header")
    else:
        raise InputValidationError(
            f"Expected NRRD image object with .array and .header attributes, "
            f"got {type(obj).__name__}. Pass a nrrd image object or use the appropriate converter."
        )

    if not isinstance(array, np.ndarray):
        raise InputValidationError(
            f"NRRD data must be a numpy array, got {type(array).__name__}."
        )

    if not np.issubdtype(array.dtype, np.integer):
        raise InputValidationError(
            f"NRRD image must have integer dtype, got {array.dtype}. "
            "Label and mask arrays must contain integer values."
        )

    # Extract spacing from header's space_directions
    spacing: tuple[float, ...] | None = None
    if header and "space directions" in header:
        try:
            space_directions = np.array(header["space directions"])
            # Compute norm of each direction vector to get voxel spacing
            spacing = tuple(float(np.linalg.norm(v)) for v in space_directions)
        except (TypeError, ValueError) as e:
            raise InputValidationError(
                f"Failed to extract spacing from NRRD space directions: {e}"
            ) from e

    return array, spacing
