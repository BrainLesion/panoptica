"""Voronoi partition of a volume by nearest labelled reference region.

A single Euclidean feature transform of the background (``mask == 0``) yields,
for every voxel, the index of its nearest labelled voxel in one pass — an
O(n_voxels) computation replacing a naive per-region distance transform.

``spacing`` lets the nearest-region assignment respect anisotropic voxels.
"""

from __future__ import annotations

from panoptica.backends.capability import warn_once
from panoptica.core.protocols import Array, Xp


def voronoi_regions(
    mask: Array, xp: Xp, *, spacing: tuple[float, ...] | None = None
) -> tuple[Array, int]:
    """Assign every voxel to its nearest labelled region in ``mask``.

    Returns:
        ``(region_map, n_instances)`` where ``region_map`` is an ``int32`` array
        the shape of ``mask`` holding, per voxel, the label of the nearest
        nonzero region (``0`` everywhere if ``mask`` has no instances).
    """
    n_instances = int(mask.max())
    if n_instances <= 0 or not bool(xp.any(mask)):
        return xp.zeros(mask.shape, dtype=xp.int32), n_instances

    if xp.__name__ == "cupy":
        region_map = _voronoi_gpu(mask, xp, spacing=spacing)
    else:
        region_map = _voronoi_cpu(mask, xp, spacing=spacing)

    return region_map.astype(xp.int32), n_instances


def _voronoi_cpu(mask: Array, xp: Xp, *, spacing: tuple[float, ...] | None) -> Array:
    from scipy.ndimage import distance_transform_edt

    indices = distance_transform_edt(
        mask == 0, sampling=spacing, return_distances=False, return_indices=True
    )
    return mask[tuple(indices)]


def _voronoi_gpu(mask: Array, xp: Xp, *, spacing: tuple[float, ...] | None) -> Array:
    from cupyx.scipy import ndimage as cundi

    try:
        indices = cundi.distance_transform_edt(
            mask == 0, sampling=spacing, return_distances=False, return_indices=True
        )
    except (TypeError, NotImplementedError):
        import numpy as np

        warn_once(
            "voronoi",
            "cupyx.scipy.ndimage.distance_transform_edt has no "
            "return_indices support; falling back to CPU",
        )
        host_mask = mask.get()
        region_map_host = _voronoi_cpu(host_mask, np, spacing=spacing)
        return xp.asarray(region_map_host)

    return mask[tuple(indices)]
