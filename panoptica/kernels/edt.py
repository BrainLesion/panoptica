"""Euclidean distance transform.

The big 3D distance field is the main GPU-memory concern, so it is float32
on GPU and float64 on CPU — never a hardcoded dtype.
"""

from __future__ import annotations

from panoptica.core.protocols import Array, Xp


def edt(mask: Array, xp: Xp, *, spacing: tuple[float, ...] | None = None) -> Array:
    """Euclidean distance transform of ``mask``: distance to the nearest zero voxel.

    ``spacing`` is physical voxel spacing per axis (``None`` == isotropic unit
    spacing).
    """
    if xp.__name__ == "cupy":
        return _edt_gpu(mask, xp, spacing=spacing)
    return _edt_cpu(mask, xp, spacing=spacing)


def _edt_gpu(mask: Array, xp: Xp, *, spacing: tuple[float, ...] | None) -> Array:
    from cupyx.scipy import ndimage as cundi

    field = cundi.distance_transform_edt(mask.astype(bool), sampling=spacing)
    return field.astype(xp.float32)


def _edt_cpu(mask: Array, xp: Xp, *, spacing: tuple[float, ...] | None) -> Array:
    from scipy import ndimage

    field = ndimage.distance_transform_edt(mask.astype(bool), sampling=spacing)
    return field.astype(xp.float64)  # pyrefly: ignore
