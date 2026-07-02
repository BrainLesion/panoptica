"""Per-instance metric evaluation for matched instance pairs."""

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import find_objects

from panoptica._functionals import _get_orig_onehotcc_structure, _get_paired_crop
from panoptica.metrics import Metric
from panoptica.metrics._surface_distances import (
    SURFACE_DISTANCE_METRIC_NAMES,
    _reduce_surface_metric,
    _surface_distance_pair,
)
from panoptica.utils.processing_pair import EvaluateInstancePair, MatchedInstancePair


def _extended_voxelspacing(voxelspacing: tuple[float, ...], ndim: int):
    """Match a voxelspacing to a spatial array, padding non-spatial (label) axes.

    Flattened one-hot arrays gain leading label axes when reshaped to
    ``(num_labels + 1, *spatial_shape)``; spatial metrics need a spacing entry per
    axis, so unit spacing is prepended for those extra dimensions.
    """
    if len(voxelspacing) < ndim:
        extra_dims = ndim - len(voxelspacing)
        return (1.0,) * extra_dims + tuple(voxelspacing)
    if len(voxelspacing) > ndim:
        raise ValueError(
            f"Voxelspacing has {len(voxelspacing)} dimensions but the spatial array "
            f"only has {ndim} dimensions. Voxelspacing should match the original "
            f"spatial dimensions of the data."
        )
    return voxelspacing


def _union_instance_slice(
    ref_slices: list,
    pred_slices: list,
    label: int,
    shape: tuple[int, ...],
    px_pad: int = 2,
) -> tuple[slice, ...] | None:
    """Padded union of a label's reference and prediction bounding boxes.

    ``ref_slices`` / ``pred_slices`` are ``scipy.ndimage.find_objects`` outputs (indexed
    by ``label - 1``, ``None`` where the label is absent). Returns a per-axis slice tuple
    that bounds the instance in both arrays plus ``px_pad`` voxels (so the downstream
    ``_get_paired_crop`` reproduces exactly the crop it would compute on the full array),
    or ``None`` when the label is absent from both.
    """
    boxes = []
    for slices in (ref_slices, pred_slices):
        box = slices[label - 1] if 0 < label <= len(slices) else None
        if box is not None:
            boxes.append(box)
    if not boxes:
        return None
    return tuple(
        slice(
            max(min(b[ax].start for b in boxes) - px_pad, 0),
            min(max(b[ax].stop for b in boxes) + px_pad, shape[ax]),
        )
        for ax in range(len(shape))
    )


@dataclass(frozen=True)
class _InstanceEvaluation:
    """Result of evaluating a single matched reference instance.

    Attributes:
        metrics: Per-metric scores keyed by ``Metric``. Empty dict signals no overlap (the instance is filtered out by the caller's ``if not metrics`` check).
        voxel_count_ref: Raw voxel count of the reference instance (``np.count_nonzero`` of the cropped reference mask).
        volume_ref: Physical volume of the reference instance, computed as ``voxel_count_ref * prod(voxelspacing)``.
    """

    metrics: dict[Metric, float] = field(default_factory=dict)
    voxel_count_ref: int = 0
    volume_ref: float = 0.0


def evaluate_matched_instance(
    matched_instance_pair: MatchedInstancePair,
    eval_metrics: list[Metric] | None = None,
    decision_metric: Metric | None = Metric.IOU,
    decision_threshold: float | None = None,
    voxelspacing: tuple[float, ...] | None = None,
    processing_pair_orig_shape: tuple[int, ...] | None = None,
    n_ref_labels: int | None = None,
    instance_metric_params: dict[Metric, dict] | None = None,
    **kwargs,
) -> EvaluateInstancePair:
    """
    Evaluate a given MatchedInstancePair given metrics and decision threshold.

    Args:
        processing_pair (MatchedInstancePair): The matched instance pair containing original labels.
        labelmap (Instance_Label_Map): The instance label map obtained from instance matching.

    Returns:
        EvaluateInstancePair: Evaluated pair of instances

    """
    if eval_metrics is None:
        eval_metrics = [Metric.DSC, Metric.IOU, Metric.ASSD]
    if decision_metric is not None:
        if decision_metric.name not in [v.name for v in eval_metrics]:
            raise ValueError("decision metric not contained in eval_metrics")
        if decision_threshold is None:
            raise ValueError("decision metric set but no threshold")
    # Initialize variables for True Positives (tp)
    score_dict: dict[Metric, list[float]] = {m: [] for m in eval_metrics}

    reference_arr, prediction_arr = (
        matched_instance_pair.reference_arr,
        matched_instance_pair.prediction_arr,
    )
    ref_matched_labels = matched_instance_pair.matched_instances

    # Precompute every label's bounding box once (a single pass over each array) so each
    # instance is evaluated within its own crop instead of rescanning the full array.
    ref_slices = find_objects(reference_arr)
    pred_slices = find_objects(prediction_arr)

    per_instance_results: list[_InstanceEvaluation] = [
        _evaluate_instance(
            reference_arr,
            prediction_arr,
            ref_idx,
            eval_metrics,
            voxelspacing,
            processing_pair_orig_shape,
            n_ref_labels,
            instance_slice=_union_instance_slice(
                ref_slices, pred_slices, ref_idx, reference_arr.shape
            ),
            metric_params=instance_metric_params,
        )
        for ref_idx in ref_matched_labels
    ]

    # TODO if instance matcher already gives matching metric, adapt here!
    tp = 0
    instance_voxel_count_matched_ref: list[int] = []
    instance_volume_matched_ref: list[float] = []
    instance_voxel_count_unmatched_ref: list[int] = []
    instance_volume_unmatched_ref: list[float] = []
    for instance_result in per_instance_results:
        # Decision-threshold rejection and the no-overlap safety guard both demote
        # the ref to unmatched, so n_matched + n_unmatched == n_ref_instances.
        if not instance_result.metrics:
            instance_voxel_count_unmatched_ref.append(instance_result.voxel_count_ref)
            instance_volume_unmatched_ref.append(instance_result.volume_ref)
            continue
        accepted = decision_metric is None or (
            decision_threshold is not None
            and decision_metric.score_beats_threshold(
                instance_result.metrics[decision_metric], decision_threshold
            )
        )
        if not accepted:
            instance_voxel_count_unmatched_ref.append(instance_result.voxel_count_ref)
            instance_volume_unmatched_ref.append(instance_result.volume_ref)
            continue
        tp += 1
        instance_voxel_count_matched_ref.append(instance_result.voxel_count_ref)
        instance_volume_matched_ref.append(instance_result.volume_ref)
        for metric, score in instance_result.metrics.items():
            score_dict[metric].append(score)

    if matched_instance_pair.missed_reference_labels:
        unique_labels, counts = np.unique(reference_arr, return_counts=True)
        label_to_count = dict(zip(unique_labels.tolist(), counts.tolist()))
        missed_voxel_size = float(
            np.prod(
                voxelspacing
                if voxelspacing is not None
                else (1.0,) * reference_arr.ndim
            )
        )
        for ref_idx in matched_instance_pair.missed_reference_labels:
            voxel_count = int(label_to_count.get(ref_idx, 0))
            instance_voxel_count_unmatched_ref.append(voxel_count)
            instance_volume_unmatched_ref.append(float(voxel_count) * missed_voxel_size)

    # Create and return the EvaluateInstancePair object with computed metrics
    return EvaluateInstancePair(
        reference_arr=matched_instance_pair.reference_arr,
        prediction_arr=matched_instance_pair.prediction_arr,
        n_pred_instances=matched_instance_pair.n_pred_instances,
        n_ref_instances=matched_instance_pair.n_ref_instances,
        tp=tp,
        list_metrics=score_dict,
        instance_voxel_count_matched_ref=instance_voxel_count_matched_ref,
        instance_volume_matched_ref=instance_volume_matched_ref,
        instance_voxel_count_unmatched_ref=instance_voxel_count_unmatched_ref,
        instance_volume_unmatched_ref=instance_volume_unmatched_ref,
    )


def _evaluate_instance(
    reference_arr: np.ndarray,
    prediction_arr: np.ndarray,
    ref_idx: int,
    eval_metrics: list[Metric],
    voxelspacing: tuple[float, ...] | None = None,
    processing_pair_orig_shape: tuple[int, ...] | None = None,
    n_ref_labels: int | None = None,
    instance_slice: tuple[slice, ...] | None = None,
    metric_params: dict[Metric, dict] | None = None,
) -> _InstanceEvaluation:
    """
    Evaluate a single instance.

    Args:
        ref_labels (np.ndarray): Reference instance segmentation mask.
        pred_labels (np.ndarray): Predicted instance segmentation mask.
        ref_idx (int): The label of the current instance.
        iou_threshold (float): The IoU threshold for considering a match.

    Returns:
        _InstanceEvaluation: Per-metric scores, raw voxel count of the reference instance, and physical volume (voxel count * prod(voxelspacing)).
        If the reference label is absent from ``reference_arr`` (``voxel_count_ref == 0``), the result has empty metrics and zero count/volume.
        If the reference is present but the prediction has no voxels for this label, the result has empty metrics but still carries the reference's true voxel count and volume, so the caller can record the ref as unmatched with its actual size.
    """
    # Restrict the per-instance mask extraction to the instance's bounding box when one
    # was precomputed; the one-hot spatial path below still reshapes the full arrays.
    if instance_slice is not None:
        ref_arr = reference_arr[instance_slice] == ref_idx
        pred_arr = prediction_arr[instance_slice] == ref_idx
    else:
        ref_arr = reference_arr == ref_idx
        pred_arr = prediction_arr == ref_idx

    # Detect if we have flattened one-hot arrays that need reshaping for spatial metrics
    is_flattened_onehot = (
        reference_arr.ndim == 1
        and processing_pair_orig_shape is not None
        and n_ref_labels is not None
    )

    # Set default voxelspacing based on original or current array dimensions
    if voxelspacing is None:
        if is_flattened_onehot:
            assert processing_pair_orig_shape is not None
            voxelspacing = (1.0,) * len(processing_pair_orig_shape)
        else:
            voxelspacing = (1.0,) * reference_arr.ndim

    voxel_size = float(np.prod(voxelspacing))
    voxel_count_ref = int(ref_arr.sum())
    volume_ref = float(voxel_count_ref) * voxel_size

    if voxel_count_ref == 0 or pred_arr.sum() == 0:
        return _InstanceEvaluation(
            voxel_count_ref=voxel_count_ref,
            volume_ref=volume_ref,
        )

    # Crop down for speedup
    crop = _get_paired_crop(
        pred_arr,
        ref_arr,
    )

    ref_arr = ref_arr[crop]
    pred_arr = pred_arr[crop]

    result: dict[Metric, float] = {}

    # Resolve the mask view spatial metrics operate on, once instead of per metric.
    # For flattened one-hot (LabelPartGroup) data this means reshaping back to
    # (num_classes, *spatial_shape), extracting this instance and cropping; for normal
    # data the already-cropped instance masks are used directly.
    needs_spatial = any(
        metric.requires_spatial and is_flattened_onehot for metric in eval_metrics
    )
    if needs_spatial:
        assert n_ref_labels is not None and processing_pair_orig_shape is not None
        ref_spatial = _get_orig_onehotcc_structure(
            reference_arr, n_ref_labels, processing_pair_orig_shape
        )
        pred_spatial = _get_orig_onehotcc_structure(
            prediction_arr, n_ref_labels, processing_pair_orig_shape
        )
        ref_spatial_instance = ref_spatial == ref_idx
        pred_spatial_instance = pred_spatial == ref_idx
        spatial_crop = _get_paired_crop(pred_spatial_instance, ref_spatial_instance)
        spatial_ref = ref_spatial_instance[spatial_crop]
        spatial_pred = pred_spatial_instance[spatial_crop]
        spatial_voxelspacing = _extended_voxelspacing(voxelspacing, spatial_ref.ndim)
    else:
        spatial_ref, spatial_pred = ref_arr, pred_arr
        spatial_voxelspacing = voxelspacing

    # ASSD/HD/HD95/NSD are all reductions of the same directional surface-distance
    # pair, so compute it once for this instance and reduce, rather than recomputing
    # the distance transforms inside every surface metric.
    surface_pair = None
    if any(m.name in SURFACE_DISTANCE_METRIC_NAMES for m in eval_metrics):
        surface_pair = _surface_distance_pair(
            spatial_ref, spatial_pred, voxelspacing=spatial_voxelspacing
        )

    for metric in eval_metrics:
        # Fixed per-metric parameters (e.g. NSD threshold) configured via
        # ConfiguredMetric. Default-parameter metrics get an empty dict and behave
        # exactly as before.
        params = metric_params.get(metric, {}) if metric_params else {}
        if metric.name in SURFACE_DISTANCE_METRIC_NAMES:
            assert surface_pair is not None
            metric_value = _reduce_surface_metric(
                metric.name,
                surface_pair[0],
                surface_pair[1],
                voxelspacing=spatial_voxelspacing,
                **params,
            )
        elif metric.requires_spatial and is_flattened_onehot:
            # Spatial-but-not-surface metrics (e.g. center distance) on one-hot data.
            metric_value = metric(
                spatial_ref, spatial_pred, voxelspacing=spatial_voxelspacing, **params
            )
        else:
            # Non-spatial metrics, or any metric on normal (non-one-hot) arrays.
            metric_value = metric(
                ref_arr, pred_arr, voxelspacing=voxelspacing, **params
            )

        result[metric] = metric_value

    return _InstanceEvaluation(
        metrics=result,
        voxel_count_ref=voxel_count_ref,
        volume_ref=volume_ref,
    )
