"""Sanity checks for segmentation label and mask arrays."""

from __future__ import annotations

import numpy as np

from panoptica.core.errors import InputValidationError
from panoptica.core.protocols import Array


def check_arrays_same_shape(ref: Array, pred: Array) -> None:
    """Verify that ref and pred have the same shape.

    Args:
        ref: Reference array.
        pred: Prediction array.

    Raises:
        InputValidationError: If shapes do not match.
    """
    if ref.shape != pred.shape:
        raise InputValidationError(
            f"Reference and prediction arrays must have the same shape. "
            f"Got reference shape {ref.shape} and prediction shape {pred.shape}."
        )


def check_arrays_integer_dtype(ref: Array, pred: Array) -> None:
    """Verify that both arrays have integer dtype.

    Args:
        ref: Reference array.
        pred: Prediction array.

    Raises:
        InputValidationError: If either array is not integer.
    """
    xp = type(ref).__module__.split(".")[0]
    if xp == "cupy":
        import cupy as cp

        if not cp.issubdtype(ref.dtype, cp.integer):
            raise InputValidationError(
                f"Reference array must have integer dtype, got {ref.dtype}."
            )
        if not cp.issubdtype(pred.dtype, cp.integer):
            raise InputValidationError(
                f"Prediction array must have integer dtype, got {pred.dtype}."
            )
    else:
        if not np.issubdtype(ref.dtype, np.integer):
            raise InputValidationError(
                f"Reference array must have integer dtype, got {ref.dtype}."
            )
        if not np.issubdtype(pred.dtype, np.integer):
            raise InputValidationError(
                f"Prediction array must have integer dtype, got {pred.dtype}."
            )


def check_arrays_non_negative(ref: Array, pred: Array) -> None:
    """Verify that both arrays are non-negative (no negative labels).

    Args:
        ref: Reference array.
        pred: Prediction array.

    Raises:
        InputValidationError: If either array contains negative values.
    """
    if ref.min() < 0:
        raise InputValidationError(
            f"Reference array contains negative values (min={ref.min()}). "
            "Label and mask arrays must be non-negative."
        )
    if pred.min() < 0:
        raise InputValidationError(
            f"Prediction array contains negative values (min={pred.min()}). "
            "Label and mask arrays must be non-negative."
        )


def sanity_check(ref: Array, pred: Array) -> None:
    """Run all sanity checks on reference and prediction arrays.

    Performs:
    - Shape equality check
    - Integer dtype check
    - Non-negative value check

    Args:
        ref: Reference array (semantic mask or labeled instance map).
        pred: Prediction array (semantic mask or labeled instance map).

    Raises:
        InputValidationError: If any check fails.
    """
    check_arrays_same_shape(ref, pred)
    check_arrays_integer_dtype(ref, pred)
    check_arrays_non_negative(ref, pred)
