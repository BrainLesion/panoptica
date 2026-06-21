import numpy as np
from scipy.optimize import linear_sum_assignment


def _positive_labels_and_areas(
    arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return positive instance labels and their pixel/voxel counts."""
    positive = arr[arr > 0]
    if positive.size == 0:
        return (
            np.asarray([], dtype=np.int64),
            np.asarray([], dtype=np.float64),
        )

    labels, counts = np.unique(positive, return_counts=True)
    return labels.astype(np.int64, copy=False), counts.astype(
        np.float64, copy=False
    )


def _label_overlap_statistics(
    reference_arr: np.ndarray,
    prediction_arr: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Compute pairwise intersection, union, and IoU matrices.

    This is for labeled instance masks.

    Assumes:
    - 0 is background
    - positive integer values are instance IDs
    - reference_arr and prediction_arr have the same shape
    """
    reference_arr = np.asarray(reference_arr)
    prediction_arr = np.asarray(prediction_arr)

    if reference_arr.shape != prediction_arr.shape:
        raise ValueError(
            "reference_arr and prediction_arr must have the same shape"
        )

    if reference_arr.ndim < 2:
        raise ValueError(
            "reference_arr and prediction_arr must be at least 2D label maps"
        )

    if np.any(reference_arr < 0) or np.any(prediction_arr < 0):
        raise ValueError(
            "AJI expects non-negative label maps with 0 as background"
        )

    ref_labels, ref_areas = _positive_labels_and_areas(reference_arr)
    pred_labels, pred_areas = _positive_labels_and_areas(prediction_arr)

    intersections = np.zeros(
        (len(ref_labels), len(pred_labels)), dtype=np.float64
    )

    if len(ref_labels) == 0 or len(pred_labels) == 0:
        unions = ref_areas[:, None] + pred_areas[None, :]
        ious = np.zeros_like(unions, dtype=np.float64)
        return (
            ref_labels,
            pred_labels,
            ref_areas,
            pred_areas,
            intersections,
            unions,
            ious,
        )

    overlap_mask = (reference_arr > 0) & (prediction_arr > 0)

    if np.any(overlap_mask):
        overlap_pairs, overlap_counts = np.unique(
            np.stack(
                (
                    reference_arr[overlap_mask].astype(np.int64, copy=False),
                    prediction_arr[overlap_mask].astype(np.int64, copy=False),
                ),
                axis=1,
            ),
            axis=0,
            return_counts=True,
        )

        ref_index = {int(label): idx for idx, label in enumerate(ref_labels)}
        pred_index = {int(label): idx for idx, label in enumerate(pred_labels)}

        for (ref_label, pred_label), count in zip(
            overlap_pairs, overlap_counts
        ):
            intersections[
                ref_index[int(ref_label)],
                pred_index[int(pred_label)],
            ] = float(count)

    unions = ref_areas[:, None] + pred_areas[None, :] - intersections
    ious = np.divide(
        intersections,
        unions,
        out=np.zeros_like(intersections, dtype=np.float64),
        where=unions > 0,
    )

    return (
        ref_labels,
        pred_labels,
        ref_areas,
        pred_areas,
        intersections,
        unions,
        ious,
    )


def _compute_aggregated_jaccard_index(
    reference_arr: np.ndarray,
    prediction_arr: np.ndarray,
    *args,
    **kwargs,
) -> float:
    """
    Compute classical Aggregated Jaccard Index.

    For every reference instance, AJI pairs it with the prediction instance
    that has the highest IoU. This classical formulation may select the same
    prediction for multiple reference instances. Unpaired reference and
    prediction instances are added to the denominator.
    """
    (
        ref_labels,
        pred_labels,
        ref_areas,
        pred_areas,
        intersections,
        unions,
        ious,
    ) = _label_overlap_statistics(reference_arr, prediction_arr)

    if len(ref_labels) == 0 and len(pred_labels) == 0:
        return 1.0

    if len(ref_labels) == 0 or len(pred_labels) == 0:
        return 0.0

    best_pred_for_ref = np.argmax(ious, axis=1)
    best_iou_for_ref = ious[np.arange(len(ref_labels)), best_pred_for_ref]

    paired_ref = np.where(best_iou_for_ref > 0)[0]
    paired_pred = best_pred_for_ref[paired_ref]

    intersection_sum = intersections[paired_ref, paired_pred].sum(
        dtype=np.float64
    )
    union_sum = unions[paired_ref, paired_pred].sum(dtype=np.float64)

    unpaired_ref = np.setdiff1d(
        np.arange(len(ref_labels)),
        paired_ref,
        assume_unique=True,
    )

    used_pred = np.unique(paired_pred)
    unpaired_pred = np.setdiff1d(
        np.arange(len(pred_labels)),
        used_pred,
        assume_unique=True,
    )

    union_sum += ref_areas[unpaired_ref].sum(dtype=np.float64)
    union_sum += pred_areas[unpaired_pred].sum(dtype=np.float64)

    if union_sum == 0:
        return 1.0

    return float(intersection_sum / union_sum)


def _compute_aggregated_jaccard_index_plus(
    reference_arr: np.ndarray,
    prediction_arr: np.ndarray,
    *args,
    **kwargs,
) -> float:
    """
    Compute AJI+.

    AJI+ uses one-to-one optimal matching between reference and prediction
    instances before aggregating intersections and unions. Unpaired reference
    and prediction instances are added to the denominator.
    """
    (
        ref_labels,
        pred_labels,
        ref_areas,
        pred_areas,
        intersections,
        unions,
        ious,
    ) = _label_overlap_statistics(reference_arr, prediction_arr)

    if len(ref_labels) == 0 and len(pred_labels) == 0:
        return 1.0

    if len(ref_labels) == 0 or len(pred_labels) == 0:
        return 0.0

    if np.any(ious > 0):
        paired_ref, paired_pred = linear_sum_assignment(-ious)

        keep = ious[paired_ref, paired_pred] > 0
        paired_ref = paired_ref[keep]
        paired_pred = paired_pred[keep]
    else:
        paired_ref = np.asarray([], dtype=np.int64)
        paired_pred = np.asarray([], dtype=np.int64)

    intersection_sum = intersections[paired_ref, paired_pred].sum(
        dtype=np.float64
    )
    union_sum = unions[paired_ref, paired_pred].sum(dtype=np.float64)

    unpaired_ref = np.setdiff1d(
        np.arange(len(ref_labels)),
        paired_ref,
        assume_unique=True,
    )
    unpaired_pred = np.setdiff1d(
        np.arange(len(pred_labels)),
        paired_pred,
        assume_unique=True,
    )

    union_sum += ref_areas[unpaired_ref].sum(dtype=np.float64)
    union_sum += pred_areas[unpaired_pred].sum(dtype=np.float64)

    if union_sum == 0:
        return 1.0

    return float(intersection_sum / union_sum)
