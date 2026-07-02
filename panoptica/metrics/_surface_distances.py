"""Shared surface-distance computation for the surface-based metric family.

ASSD, Hausdorff distance (HD), HD95 and the Normalized Surface Dice (NSD) are all
reductions of the *same* two directional surface-distance arrays between a reference
and a prediction mask. Computed independently (as the per-metric functions do) the
same Euclidean distance transforms are recomputed up to four times for one instance
pair. :func:`_surface_distance_pair` computes them once; the ``_*_from_pair`` reducers
turn that single result into each metric, so requesting several surface metrics for an
instance costs two distance transforms instead of up to eight.

The reducers intentionally mirror the standalone implementations in :mod:`.assd`,
:mod:`.hausdorff_distance` and :mod:`.normalized_surface_dice` exactly (same border
extraction, same distance transform, same arithmetic), so results are bit-identical.
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
from scipy.ndimage import _ni_support, binary_erosion, generate_binary_structure

from panoptica.metrics.assd import _distance_transform_edt

# Metrics that are pure reductions of the directional surface-distance pair.
SURFACE_DISTANCE_METRIC_NAMES = frozenset({"ASSD", "HD", "HD95", "NSD"})


def _surface_distance_pair(
    reference: np.ndarray,
    prediction: np.ndarray,
    voxelspacing=None,
    connectivity: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute both directional surface-distance arrays in a single pass each.

    Args:
        reference: Boolean (or label) reference mask.
        prediction: Boolean (or label) prediction mask.
        voxelspacing: Physical spacing per axis; ``None`` means isotropic unit spacing.
        connectivity: Connectivity for the border-extracting structuring element.

    Returns:
        ``(sd_ref, sd_pred)`` where ``sd_ref`` holds, for every reference-surface voxel,
        the distance to the nearest prediction-surface voxel, and ``sd_pred`` the
        symmetric quantity. These equal ``__surface_distances(prediction, reference)``
        and ``__surface_distances(reference, prediction)`` respectively.
    """
    prediction = np.atleast_1d(prediction.astype(bool))
    reference = np.atleast_1d(reference.astype(bool))
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, prediction.ndim)
        voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()

    footprint = generate_binary_structure(prediction.ndim, connectivity)
    prediction_border = prediction ^ binary_erosion(
        prediction, structure=footprint, iterations=1
    )
    reference_border = reference ^ binary_erosion(
        reference, structure=footprint, iterations=1
    )

    # scipy's EDT measures distance to the nearest background voxel, so each border is
    # inverted before transforming (as in the original __surface_distances).
    sd_ref = _distance_transform_edt(~prediction_border, sampling=voxelspacing)[
        reference_border
    ]
    sd_pred = _distance_transform_edt(~reference_border, sampling=voxelspacing)[
        prediction_border
    ]
    return sd_ref, sd_pred


def _assd_from_pair(sd_ref: np.ndarray, sd_pred: np.ndarray) -> float:
    """Average Symmetric Surface Distance: mean of the two mean surface distances."""
    return float(np.mean((sd_ref.mean(), sd_pred.mean())))


def _hd_from_pair(sd_ref: np.ndarray, sd_pred: np.ndarray) -> float:
    """Hausdorff distance: the larger of the two directional maxima."""
    return max(sd_ref.max(), sd_pred.max())


def _hd95_from_pair(sd_ref: np.ndarray, sd_pred: np.ndarray) -> float:
    """95th-percentile Hausdorff distance over the pooled surface distances."""
    return np.percentile(np.hstack((sd_ref, sd_pred)), 95)


def _nsd_from_pair(
    sd_ref: np.ndarray,
    sd_pred: np.ndarray,
    voxelspacing=None,
    threshold=None,
) -> float:
    """Normalized Surface Dice at ``threshold`` (default: min voxel spacing, else 0.5)."""
    if threshold is None:
        threshold = min(voxelspacing) if voxelspacing is not None else 0.5
    if threshold == 0.5:
        warnings.warn(
            "The threshold is set to 0.5, which is the default value, which may not be appropriate for your data."
        )
    numel_a = len(sd_ref)
    numel_b = len(sd_pred)
    tp_a = np.sum(sd_ref <= threshold) / numel_a
    tp_b = np.sum(sd_pred <= threshold) / numel_b
    fp = np.sum(sd_ref > threshold) / numel_a
    fn = np.sum(sd_pred > threshold) / numel_b
    return (tp_a + tp_b) / (tp_a + tp_b + fp + fn + sys.float_info.min)


def _reduce_surface_metric(
    metric_name: str,
    sd_ref: np.ndarray,
    sd_pred: np.ndarray,
    voxelspacing=None,
    threshold=None,
) -> float:
    """Dispatch a surface-distance metric by name onto the precomputed pair.

    ``threshold`` is the NSD distance threshold; it is ignored by the distance-based
    reducers (ASSD/HD/HD95). Since the surface-distance pair is shared across all
    surface metrics for an instance, per-metric ``connectivity`` is not configurable
    here (the pair is built with the default connectivity).
    """
    if metric_name == "ASSD":
        return _assd_from_pair(sd_ref, sd_pred)
    if metric_name == "HD":
        return _hd_from_pair(sd_ref, sd_pred)
    if metric_name == "HD95":
        return _hd95_from_pair(sd_ref, sd_pred)
    if metric_name == "NSD":
        return _nsd_from_pair(
            sd_ref, sd_pred, voxelspacing=voxelspacing, threshold=threshold
        )
    raise ValueError(f"{metric_name} is not a surface-distance metric")
