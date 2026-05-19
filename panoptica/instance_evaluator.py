from dataclasses import dataclass, field
from multiprocessing import Pool
import numpy as np

from panoptica.metrics import Metric
from panoptica.utils.processing_pair import MatchedInstancePair, EvaluateInstancePair
from panoptica._functionals import _get_paired_crop, _get_orig_onehotcc_structure
from panoptica.utils import compute_ref_voxel_count_and_volume


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
    eval_metrics: list[Metric] = [Metric.DSC, Metric.IOU, Metric.ASSD],
    decision_metric: Metric | None = Metric.IOU,
    decision_threshold: float | None = None,
    voxelspacing: tuple[float, ...] | None = None,
    processing_pair_orig_shape: tuple[int, ...] | None = None,
    n_ref_labels: int | None = None,
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

    per_instance_results: list[_InstanceEvaluation] = [
        _evaluate_instance(
            reference_arr,
            prediction_arr,
            ref_idx,
            eval_metrics,
            voxelspacing,
            processing_pair_orig_shape,
            n_ref_labels,
        )
        for ref_idx in ref_matched_labels
    ]
    # instance_pairs = [(reference_arr, prediction_arr, ref_idx, eval_metrics) for ref_idx in ref_matched_labels]
    # with Pool() as pool:
    #    metric_dicts: list[_InstanceEvaluation] = pool.starmap(
    #        _evaluate_instance, instance_pairs
    #    )

    # TODO if instance matcher already gives matching metric, adapt here!
    tp = 0
    instance_voxel_count_matched_ref: list[int] = []
    instance_volume_matched_ref: list[float] = []
    for instance_result in per_instance_results:
        if not instance_result.metrics:
            continue
        accepted = decision_metric is None or (
            decision_threshold is not None
            and decision_metric.score_beats_threshold(
                instance_result.metrics[decision_metric], decision_threshold
            )
        )
        if not accepted:
            continue
        tp += 1
        instance_voxel_count_matched_ref.append(instance_result.voxel_count_ref)
        instance_volume_matched_ref.append(instance_result.volume_ref)
        for metric, score in instance_result.metrics.items():
            score_dict[metric].append(score)

    instance_voxel_count_unmatched_ref: list[int] = []
    instance_volume_unmatched_ref: list[float] = []
    for ref_idx in matched_instance_pair.missed_reference_labels:
        voxel_count, volume = compute_ref_voxel_count_and_volume(
            reference_arr, ref_idx, voxelspacing
        )
        instance_voxel_count_unmatched_ref.append(voxel_count)
        instance_volume_unmatched_ref.append(volume)

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
        If the instance has no overlap, returns the default ``_InstanceEvaluation()`` (empty metrics, zero count and volume).
    """
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
            voxelspacing = (1.0,) * len(processing_pair_orig_shape)
        else:
            voxelspacing = (1.0,) * reference_arr.ndim

    if ref_arr.sum() == 0 or pred_arr.sum() == 0:
        return _InstanceEvaluation()

    # Crop down for speedup
    crop = _get_paired_crop(
        pred_arr,
        ref_arr,
    )

    ref_arr = ref_arr[crop]
    pred_arr = pred_arr[crop]

    voxel_count_ref = int(np.count_nonzero(ref_arr))
    volume_ref = float(voxel_count_ref * np.prod(voxelspacing))

    result: dict[Metric, float] = {}

    # Cache spatial structures if any metric requires them and is_flattened_onehot is True
    needs_spatial = any(
        metric.requires_spatial and is_flattened_onehot for metric in eval_metrics
    )
    ref_spatial = pred_spatial = None
    if needs_spatial:
        # Reshape full arrays back to (num_classes, *spatial_shape) only once
        ref_spatial = _get_orig_onehotcc_structure(
            reference_arr, n_ref_labels, processing_pair_orig_shape
        )
        pred_spatial = _get_orig_onehotcc_structure(
            prediction_arr, n_ref_labels, processing_pair_orig_shape
        )

    for metric in eval_metrics:
        if metric.requires_spatial and is_flattened_onehot:
            # For spatial metrics on flattened one-hot data, use cached spatial structure

            # Extract the specific instance from the spatial structure
            ref_spatial_instance = ref_spatial == ref_idx
            pred_spatial_instance = pred_spatial == ref_idx

            # Crop for performance (only on spatial dimensions)
            crop = _get_paired_crop(pred_spatial_instance, ref_spatial_instance)
            ref_spatial_cropped = ref_spatial_instance[crop]
            pred_spatial_cropped = pred_spatial_instance[crop]

            # Adjust voxelspacing to match spatial array dimensions if needed
            if len(voxelspacing) < ref_spatial_cropped.ndim:
                # After reshaping from flattened one-hot, arrays have shape (num_labels+1, *spatial_shape)
                # The instance extraction preserves this shape as a boolean mask
                # Spatial metrics require voxelspacing for ALL dimensions, so we prepend 1.0
                # for the non-spatial label dimension(s)
                extra_dims = ref_spatial_cropped.ndim - len(voxelspacing)
                extended_voxelspacing = (1.0,) * extra_dims + tuple(voxelspacing)
            elif len(voxelspacing) > ref_spatial_cropped.ndim:
                raise ValueError(
                    f"Voxelspacing has {len(voxelspacing)} dimensions but the spatial array "
                    f"only has {ref_spatial_cropped.ndim} dimensions. Voxelspacing should match "
                    f"the original spatial dimensions of the data."
                )
            else:
                extended_voxelspacing = voxelspacing

            metric_value = metric(
                ref_spatial_cropped,
                pred_spatial_cropped,
                voxelspacing=extended_voxelspacing,
            )
        else:
            # For non-spatial metrics or normal arrays, use standard computation
            metric_value = metric(ref_arr, pred_arr, voxelspacing=voxelspacing)

        result[metric] = metric_value

    return _InstanceEvaluation(
        metrics=result,
        voxel_count_ref=voxel_count_ref,
        volume_ref=volume_ref,
    )
