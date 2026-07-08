"""Label remapping via a lookup table.

Builds a dense lookup array sized to the max label/key/value seen and indexes
into it, rather than a per-voxel dict lookup (vectorized, works identically
for numpy and cupy).
"""

from __future__ import annotations

from panoptica.core.protocols import Array, Xp


def map_labels(arr: Array, mapping: dict[int, int], xp: Xp) -> Array:
    """Return a copy of ``arr`` with labels remapped according to ``mapping``.

    Labels not present as keys in ``mapping`` are left unchanged. Pure: ``arr`` is
    never mutated.
    """
    if len(mapping) == 0:
        return xp.array(arr, copy=True)

    keys = xp.asarray(list(mapping.keys()), dtype=arr.dtype)
    values = xp.asarray(list(mapping.values()), dtype=arr.dtype)

    max_value = int(max(int(arr.max()), int(keys.max()), int(values.max()))) + 1

    lookup = xp.arange(max_value, dtype=arr.dtype)
    lookup[keys] = values
    return lookup[arr]
