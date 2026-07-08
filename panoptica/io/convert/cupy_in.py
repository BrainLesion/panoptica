"""Converter for CuPy arrays and GPU-resident arrays (__cuda_array_interface__ / DLPack)."""

from __future__ import annotations

from typing import Any

from panoptica.core.errors import BackendUnavailable, InputValidationError


def convert_to_cupy(obj: Any) -> tuple[Any, None]:
    """Convert a CuPy array or GPU-resident array to canonical form.

    Accepts cupy arrays or any object exposing __cuda_array_interface__ or DLPack,
    keeping the array on-device (no .numpy() forced copy to CPU).

    Args:
        obj: A cupy array, or an object with __cuda_array_interface__ or DLPack.

    Returns:
        (array, spacing): The GPU-resident array (validated) and None for spacing.

    Raises:
        BackendUnavailable: If cupy is not installed.
        InputValidationError: If obj is not a supported GPU array type.
    """
    try:
        import cupy as cp
    except ImportError as e:
        raise BackendUnavailable(
            "cupy is not installed. Install it with: pip install cupy-cuda12x"
        ) from e

    if isinstance(obj, cp.ndarray):
        if not cp.issubdtype(obj.dtype, cp.integer):
            raise InputValidationError(
                f"CuPy array must have integer dtype, got {obj.dtype}. "
                "Label and mask arrays must contain integer values."
            )
        return obj, None

    # Try to adopt via DLPack (zero-copy from other GPU frameworks)
    if hasattr(obj, "__dlpack__"):
        try:
            array = cp.asarray(obj)
            if not cp.issubdtype(array.dtype, cp.integer):
                raise InputValidationError(
                    f"DLPack array must have integer dtype, got {array.dtype}. "
                    "Label and mask arrays must contain integer values."
                )
            return array, None
        except Exception as e:
            raise InputValidationError(f"Failed to adopt DLPack array: {e}") from e

    # Try to adopt via __cuda_array_interface__ (zero-copy from other GPU frameworks)
    if hasattr(obj, "__cuda_array_interface__"):
        try:
            array = cp.asarray(obj)
            if not cp.issubdtype(array.dtype, cp.integer):
                raise InputValidationError(
                    f"__cuda_array_interface__ array must have integer dtype, got {array.dtype}. "
                    "Label and mask arrays must contain integer values."
                )
            return array, None
        except Exception as e:
            raise InputValidationError(
                f"Failed to adopt __cuda_array_interface__ array: {e}"
            ) from e

    raise InputValidationError(
        f"Expected cupy array or GPU array (with __cuda_array_interface__ or DLPack), "
        f"got {type(obj).__name__}. Pass the array directly or use the appropriate converter."
    )
