"""Global (whole-image binary) metric computation.

Extracted from ``PanopticaResult._calc_global_bin_metric`` (#248) so the metric math
lives in the computation package and the result object stays focused on holding and
looking up results. The result object is passed in for its global-metric set, edge-case
handler and (for ``LabelPartGroup``) multi-channel data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from panoptica.metrics import Metric, MetricCouldNotBeComputedException

if TYPE_CHECKING:
    from panoptica.panoptica_result import PanopticaResult


def calc_global_bin_metric(
    result: "PanopticaResult",
    metric: Metric,
    prediction_arr,
    reference_arr,
    do_binarize: bool = True,
    params: dict | None = None,
):
    """
    Calculates a global binary metric based on predictions and references.
    For multi-channel data (LabelPartGroup), computes metrics per channel and averages.

    Args:
        result (PanopticaResult): The result object (global-metric set, edge-case
            handler, optional multi-channel data, channel-metric store).
        metric (Metric): The metric to compute.
        prediction_arr: The predicted values.
        reference_arr: The ground truth values.
        do_binarize (bool): Whether to binarize the input arrays. Defaults to True.
        params (dict | None): Fixed metric parameters (e.g. NSD ``threshold``) forwarded
            to every metric call. Defaults to no extra parameters.

    Returns:
        The calculated metric value or mean of channel metrics for multi-channel data.

    Raises:
        MetricCouldNotBeComputedException: If the specified metric is not set.
    """
    if metric not in result._global_metrics:
        raise MetricCouldNotBeComputedException(f"Global Metric {metric} not set")

    params = params or {}

    # Set THING_CHANNEL so it can be avoided during the part calculation
    #! Skipping channel 1 because that is not the right part + thing. That is only thing. We want part + thing evaluated and then the parts.
    THING_CHANNEL = 1

    # Handle multi-channel data from LabelPartGroup
    if hasattr(result, "_multi_channel_data"):
        channel_metrics = []
        channel_results = {}

        for i in range(result._multi_channel_data["n_channels"]):
            if i == THING_CHANNEL:
                continue
            ref_channel = result._multi_channel_data["ref_channels"][i]
            pred_channel = result._multi_channel_data["pred_channels"][i]

            # Skip empty channels (where both reference and prediction are empty)
            if ref_channel.sum() == 0 and pred_channel.sum() == 0:
                continue

            # Binarize each channel to ensure binary input
            pred_channel = (pred_channel != 0).astype(np.uint8)
            ref_channel = (ref_channel != 0).astype(np.uint8)
            # Handle edge cases for empty reference or prediction
            prediction_empty = pred_channel.sum() == 0
            reference_empty = ref_channel.sum() == 0

            if prediction_empty or reference_empty:
                is_edgecase, edge_result = result._edge_case_handler.handle_zero_tp(
                    metric, 0, int(prediction_empty), int(reference_empty)
                )
                if is_edgecase:
                    channel_result = edge_result
                else:
                    channel_result = metric(
                        reference_arr=ref_channel,
                        prediction_arr=pred_channel,
                        **params,
                    )
            else:
                channel_result = metric(
                    reference_arr=ref_channel,
                    prediction_arr=pred_channel,
                    **params,
                )

            channel_metrics.append(channel_result)
            channel_results[i] = channel_result

        # Store individual channel metrics for reference
        metric_name = metric.name.lower()
        if not hasattr(result, "_channel_metrics"):
            result._channel_metrics = {}
        result._channel_metrics[metric_name] = channel_results

        # Return mean of channel metrics
        if channel_metrics:
            return float(np.mean(channel_metrics))  # type: ignore[arg-type]
        else:
            # Handle case where no valid metrics could be computed
            is_edgecase, edge_result = result._edge_case_handler.handle_zero_tp(
                metric, 0, 1, 1
            )
            return edge_result

    # Original single-channel logic
    if do_binarize:
        pred_binary = prediction_arr.copy()
        ref_binary = reference_arr.copy()
        pred_binary[pred_binary != 0] = 1
        ref_binary[ref_binary != 0] = 1
    else:
        pred_binary = prediction_arr
        ref_binary = reference_arr

    prediction_empty = pred_binary.sum() == 0
    reference_empty = ref_binary.sum() == 0
    if prediction_empty or reference_empty:
        is_edgecase, edge_result = result._edge_case_handler.handle_zero_tp(
            metric, 0, int(prediction_empty), int(reference_empty)
        )
        if is_edgecase:
            return edge_result

    return metric(
        reference_arr=ref_binary,
        prediction_arr=pred_binary,
        **params,
    )
