"""Converter for numpy arrays (passthrough with validation)."""

from __future__ import annotations

from typing import Any

import numpy as np

from panoptica.core.errors import InputValidationError


def convert_to_numpy(obj: Any) -> tuple[np.ndarray, None]:
    """Convert a numpy array to canonical form.

    Args:
        obj: A numpy ndarray.

    Returns:
        (array, spacing): The array (validated) and None for spacing (numpy has no metadata).

    Raises:
        InputValidationError: If obj is not a numpy array.
    """
    if not isinstance(obj, np.ndarray):
        raise InputValidationError(
            f"Expected numpy.ndarray, got {type(obj).__name__}. "
            "Pass the array directly or use the appropriate converter for your input type."
        )

    if not np.issubdtype(obj.dtype, np.integer):
        raise InputValidationError(
            f"Array must have integer dtype, got {obj.dtype}. "
            "Label and mask arrays must contain integer values."
        )

    return obj, None
