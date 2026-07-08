"""Surface (border) voxel extraction, shared by the ASSD/HD/HD95/NSD family.

``mask XOR erosion(mask, connectivity=1 structuring element, iterations=1)``
gives the 1-voxel-thick surface shell.

The structuring element itself (``generate_binary_structure``) is tiny,
data-independent host-side metadata, so it is built with scipy on the host
and moved to the active namespace — the actual erosion runs natively on GPU
via ``cupyx.scipy.ndimage``.
"""

from __future__ import annotations

from panoptica.core.protocols import Array, Xp


def surface_border(mask: Array, xp: Xp) -> Array:
    """Return the 1-voxel-thick boolean surface shell of ``mask``."""
    from scipy.ndimage import generate_binary_structure

    mask_bool = mask.astype(bool)
    structure = xp.asarray(generate_binary_structure(mask_bool.ndim, 1))

    if xp.__name__ == "cupy":
        from cupyx.scipy import ndimage as cundi

        eroded = cundi.binary_erosion(mask_bool, structure=structure, iterations=1)
    else:
        from scipy.ndimage import binary_erosion

        eroded = binary_erosion(mask_bool, structure=structure, iterations=1)

    return mask_bool ^ eroded
