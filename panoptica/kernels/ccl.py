"""Connected-components labeling.

cc3d for arrays with ``ndim >= 3``, ``scipy.ndimage.label`` for 1D/2D, default
connectivity being whatever each library's default is (cc3d: 26-connectivity
in 3D; scipy: the default cross/star structuring element, i.e. connectivity=1).

GPU path: cuCIM's ``skimage.measure.label`` runs natively on device —
connected components is NOT in ``backends.capability.GPU_UNSUPPORTED``, so it
is expected to stay GPU-resident. If cuCIM is unavailable we fall back to
host (cc3d/scipy) with a single deduplicated warning.
"""

from __future__ import annotations

from typing import Literal, cast

from panoptica.backends.capability import warn_once
from panoptica.core.protocols import Array, Xp


def connected_components(
    mask: Array, xp: Xp, *, connectivity: int | None = None
) -> tuple[Array, int]:
    """Label connected components of a binary/foreground mask.

    Returns:
        ``(labeled, n)`` where ``labeled`` has the same shape/backend as ``mask``
        (background stays ``0``, components are ``1..n``) and ``n`` is the
        component count.
    """
    if xp.__name__ == "cupy":
        return _connected_components_gpu(mask, xp, connectivity=connectivity)
    return _connected_components_cpu(mask, xp, connectivity=connectivity)


def _connected_components_gpu(
    mask: Array, xp: Xp, *, connectivity: int | None
) -> tuple[Array, int]:
    try:
        from cucim.skimage.measure import label as cucim_label
    except Exception:
        warn_once("ccl", "cucim is not installed; falling back to CPU")
        return _connected_components_via_host_fallback(
            mask, xp, connectivity=connectivity
        )

    # Pass the raw array (not binarized): cucim/skimage label separates connected
    # regions of equal value, matching cc3d's value-respecting CC on the CPU path.
    labeled = cucim_label(mask, connectivity=connectivity)
    n = int(labeled.max())
    return labeled, n


def _connected_components_via_host_fallback(
    mask: Array, xp: Xp, *, connectivity: int | None
) -> tuple[Array, int]:
    import numpy as np

    host_mask = mask.get() if hasattr(mask, "get") else np.asarray(mask)
    labeled_host, n = _connected_components_cpu(
        host_mask, np, connectivity=connectivity
    )
    return xp.asarray(labeled_host), n


def _connected_components_cpu(
    mask: Array, xp: Xp, *, connectivity: int | None
) -> tuple[Array, int]:
    if mask.ndim >= 3:
        import cc3d

        # Default: cc3d's own default (26-connectivity in 3D) when none is requested.
        if connectivity is None:
            labeled, n = cc3d.connected_components(mask, return_N=True)
        else:
            conn = cast(Literal[4, 6, 8, 18, 26], connectivity)
            labeled, n = cc3d.connected_components(
                mask, connectivity=conn, return_N=True
            )
        return labeled, int(n)

    from scipy import ndimage

    structure = None
    if connectivity is not None:
        structure = ndimage.generate_binary_structure(mask.ndim, connectivity)
    labeled, n = ndimage.label(mask, structure=structure)
    return labeled, int(n)


__all__ = ["connected_components"]
