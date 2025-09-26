from multiprocessing import Pool
import numpy as np

from panoptica.metrics import Metric
from panoptica.utils.processing_pair import MatchedInstancePair, EvaluateInstancePair
from panoptica._functionals import _get_paired_crop, _get_orig_onehotcc_structure


def evaluate_matched_instance(
    matched_instance_pair: MatchedInstancePair,
    eval_metrics: list[Metric] = [Metric.DSC, Metric.IOU, Metric.ASSD],
    decision_metric: Metric | None = Metric.IOU,
    decision_threshold: float | None = None,
    voxelspacing: tuple[float, ...] | None = None,
    processing_pair_orig_shape: tuple[int, ...] | None = None,
    num_ref_labels: int | None = None,
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
        assert decision_metric.name in [
            v.name for v in eval_metrics
        ], "decision metric not contained in eval_metrics"
        assert decision_threshold is not None, "decision metric set but no threshold"
    # Initialize variables for True Positives (tp)
    tp = len(matched_instance_pair.matched_instances)
    score_dict: dict[Metric, list[float]] = {m: [] for m in eval_metrics}

    reference_arr, prediction_arr = (
        matched_instance_pair.reference_arr,
        matched_instance_pair.prediction_arr,
    )
    ref_matched_labels = matched_instance_pair.matched_instances

    metric_dicts: list[dict[Metric, float]] = [
        _evaluate_instance(
            reference_arr,
            prediction_arr,
            ref_idx,
            eval_metrics,
            voxelspacing,
            processing_pair_orig_shape,
            num_ref_labels,
        )
        for ref_idx in ref_matched_labels
    ]
    # instance_pairs = [(reference_arr, prediction_arr, ref_idx, eval_metrics) for ref_idx in ref_matched_labels]
    # with Pool() as pool:
    #    metric_dicts: list[dict[Metric, float]] = pool.starmap(
    #        _evaluate_instance, instance_pairs
    #    )

    # TODO if instance matcher already gives matching metric, adapt here!
    for metric_dict in metric_dicts:
        if decision_metric is None or (
            decision_threshold is not None
            and decision_metric.score_beats_threshold(
                metric_dict[decision_metric], decision_threshold
            )
        ):
            for k, v in metric_dict.items():
                score_dict[k].append(v)

    # Create and return the PanopticaResult object with computed metrics
    return EvaluateInstancePair(
        reference_arr=matched_instance_pair.reference_arr,
        prediction_arr=matched_instance_pair.prediction_arr,
        num_pred_instances=matched_instance_pair.n_prediction_instance,
        num_ref_instances=matched_instance_pair.n_reference_instance,
        tp=tp,
        list_metrics=score_dict,
    )


def _evaluate_instance(
    reference_arr: np.ndarray,
    prediction_arr: np.ndarray,
    ref_idx: int,
    eval_metrics: list[Metric],
    voxelspacing: tuple[float, ...] | None = None,
    processing_pair_orig_shape: tuple[int, ...] | None = None,
    num_ref_labels: int | None = None,
) -> dict[Metric, float]:
    """
    Evaluate a single instance.

    Args:
        ref_labels (np.ndarray): Reference instance segmentation mask.
        pred_labels (np.ndarray): Predicted instance segmentation mask.
        ref_idx (int): The label of the current instance.
        iou_threshold (float): The IoU threshold for considering a match.

    Returns:
        Tuple[int, float, float]: Tuple containing True Positives (int), Dice coefficient (float), and IoU (float).
    """
    ref_arr = reference_arr == ref_idx
    pred_arr = prediction_arr == ref_idx

    # Detect if we have flattened one-hot arrays that need reshaping for spatial metrics
    is_flattened_onehot = (
        reference_arr.ndim == 1
        and processing_pair_orig_shape is not None
        and num_ref_labels is not None
    )

    # Set default voxelspacing based on original or current array dimensions
    if voxelspacing is None:
        if is_flattened_onehot:
            voxelspacing = (1.0,) * len(processing_pair_orig_shape)
        else:
            voxelspacing = (1.0,) * reference_arr.ndim

    if ref_arr.sum() == 0 or pred_arr.sum() == 0:
        return {}

    # Crop down for speedup
    crop = _get_paired_crop(
        pred_arr,
        ref_arr,
    )

    ref_arr = ref_arr[crop]
    pred_arr = pred_arr[crop]

    result: dict[Metric, float] = {}

    # Cache spatial structures if any metric requires them and is_flattened_onehot is True
    needs_spatial = any(
        metric.requires_spatial and is_flattened_onehot for metric in eval_metrics
    )
    ref_spatial = pred_spatial = None
    if needs_spatial:
        # Reshape full arrays back to (num_classes, *spatial_shape) only once
        ref_spatial = _get_orig_onehotcc_structure(
            reference_arr, num_ref_labels, processing_pair_orig_shape
        )
        pred_spatial = _get_orig_onehotcc_structure(
            prediction_arr, num_ref_labels, processing_pair_orig_shape
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

    return result
