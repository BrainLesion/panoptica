"""Input converters for various segmentation image formats.

Each converter exposes:
    convert_to_numpy(obj) -> (array, spacing_or_None)

Converters handle type checking, dtype validation, and metadata extraction
for numpy, SimpleITK, Nibabel, NRRD, and CuPy inputs. Optional dependencies
are lazily imported and guarded with clear error messages if missing.
"""

from __future__ import annotations

__all__ = [
    "convert_to_numpy",
]


def convert_to_numpy(obj):
    """Convert a segmentation image to numpy array + spacing.

    Dispatches to the appropriate converter based on object type.
    Raises InputValidationError if the type is not supported.

    Args:
        obj: A segmentation image (numpy, SimpleITK, nibabel, nrrd, or cupy).

    Returns:
        (array, spacing): The array and voxel spacing (or None if not available).

    Raises:
        InputValidationError: If obj type is not supported.
        BackendUnavailable: If the required library is not installed.
    """
    import numpy as np

    from panoptica.io.convert import numpy_in

    if isinstance(obj, np.ndarray):
        return numpy_in.convert_to_numpy(obj)

    try:
        import SimpleITK as sitk

        if isinstance(obj, sitk.Image):
            from panoptica.io.convert import sitk_in

            return sitk_in.convert_to_numpy(obj)
    except ImportError:
        pass

    try:
        import nibabel as nib

        if isinstance(obj, (nib.Nifti1Image, nib.Nifti2Image)):
            from panoptica.io.convert import nibabel_in

            return nibabel_in.convert_to_numpy(obj)
    except ImportError:
        pass

    try:
        import cupy as cp

        if isinstance(obj, cp.ndarray):
            from panoptica.io.convert import cupy_in

            return cupy_in.convert_to_cupy(obj)
    except ImportError:
        pass

    # NRRD is duck-typed by its .array/.header attributes (no import to check).
    if hasattr(obj, "array") and hasattr(obj, "header"):
        from panoptica.io.convert import nrrd_in

        return nrrd_in.convert_to_numpy(obj)

    if hasattr(obj, "__dlpack__") or hasattr(obj, "__cuda_array_interface__"):
        from panoptica.io.convert import cupy_in

        return cupy_in.convert_to_cupy(obj)

    from panoptica.core.errors import InputValidationError

    raise InputValidationError(
        f"Unsupported input type: {type(obj).__name__}. "
        "Supported types: numpy.ndarray, SimpleITK.Image, nibabel.Nifti1Image/Nifti2Image, "
        "NRRD image, cupy.ndarray, or any array with __cuda_array_interface__ or DLPack."
    )
