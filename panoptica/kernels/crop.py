"""Bounding-box extraction for a labelled/binary mask.

Padding is a pipeline-level concern and not handled here. The paired-crop
convenience wrapper also lives in the pipeline layer, which can call this
kernel on ``ref + pred``.
"""

from __future__ import annotations

import itertools

from panoptica.core.errors import InputValidationError
from panoptica.core.protocols import Array, Xp


def bounding_box(mask: Array, xp: Xp) -> tuple[slice, ...]:
    """Return the tight bounding box (as a tuple of per-axis slices) of nonzero voxels.

    Raises:
        InputValidationError: if ``mask`` has no nonzero voxels.
    """
    ndim = mask.ndim
    shape = mask.shape

    bounds: list[int] = []
    for axes in itertools.combinations(reversed(range(ndim)), ndim - 1):
        nonzero = xp.any(mask, axis=axes)
        idx = xp.where(nonzero)[0]
        if idx.size == 0:
            raise InputValidationError(
                "bounding_box: mask is empty, cannot compute a bounding box"
            )
        bounds.append(int(idx[0]))
        bounds.append(int(idx[-1]))

    return tuple(
        slice(bounds[i], min(bounds[i + 1] + 1, shape[i // 2]))
        for i in range(0, len(bounds), 2)
    )
