"""Instance approximation: SemanticPair -> UnmatchedInstancePair via connected components.

Each side of the semantic pair is reduced to a foreground mask (optionally
restricted to a ``LabelGroup``'s value labels) and labelled with connected
components; an empty side is left at ``0`` instances without invoking the
kernel.
"""

from __future__ import annotations

from panoptica.core.labels import LabelGroup
from panoptica.core.pairs import SemanticPair, UnmatchedInstancePair
from panoptica.core.protocols import Array, Xp
from panoptica.kernels.ccl import connected_components


def _foreground_mask(arr: Array, xp: Xp, label_group: LabelGroup | None) -> Array:
    if label_group is not None:
        values = xp.asarray(list(label_group.value_labels))
        return xp.isin(arr, values)
    # Pass the raw array; binarizing would merge distinct touching labels.
    return arr


def _approximate_side(
    arr: Array, xp: Xp, label_group: LabelGroup | None, connectivity: int | None
) -> tuple[Array, int]:
    mask = _foreground_mask(arr, xp, label_group)
    if not bool(xp.any(mask)):
        return xp.zeros(arr.shape, dtype=xp.int64), 0
    labeled, n = connected_components(mask, xp, connectivity=connectivity)
    return labeled, n


def approximate(
    pair: SemanticPair,
    xp: Xp,
    *,
    label_group: LabelGroup | None = None,
    connectivity: int | None = None,
    **cfg: object,
) -> UnmatchedInstancePair:
    """Extract instances from both sides of ``pair`` via connected components.

    When ``label_group`` is given, only voxels whose value is one of its
    ``value_labels`` are treated as foreground before labelling; otherwise any
    nonzero voxel is foreground (the semantic pair is assumed already
    restricted to the group of interest by the caller).
    """
    ref_labeled, n_ref = _approximate_side(pair.ref, xp, label_group, connectivity)
    pred_labeled, n_pred = _approximate_side(pair.pred, xp, label_group, connectivity)

    return UnmatchedInstancePair(
        ref=ref_labeled,
        pred=pred_labeled,
        n_ref=n_ref,
        n_pred=n_pred,
        spacing=pair.spacing,
    )


__all__ = ["approximate"]
