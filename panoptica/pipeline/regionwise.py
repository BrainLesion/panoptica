"""Per-region (Voronoi) evaluation orchestration.

Every voxel is assigned to its nearest reference instance (a Voronoi partition
of the volume seeded by the approximated ``ref`` instances). Each region holds
exactly one reference instance, and prediction voxels are attributed to the
region they geometrically fall in, so a false-positive prediction only counts
against the region it belongs to.

This mirrors v1's ``_panoptic_evaluate_region_wise``: the partition is computed
once, then **within each region** the masked pair is matched and evaluated
independently (matching per region, NOT globally-then-restricted). Because a
region-masked pair is just a normal unmatched pair, each region's result equals
a plain :func:`evaluate` of that masked volume -- which is what the parity test
checks against v1's normal pipeline.
"""

from __future__ import annotations

from panoptica.core.pairs import UnmatchedInstancePair
from panoptica.core.protocols import Array, Xp
from panoptica.kernels.voronoi import voronoi_regions
from panoptica.pipeline.evaluate import evaluate as _evaluate
from panoptica.pipeline.match import _nonzero_labels, match


def _region_unmatched_pair(
    pair: UnmatchedInstancePair, xp: Xp, region_map: Array, region_id: int, spacing
) -> UnmatchedInstancePair:
    """Restrict `pair` to the voxels of Voronoi region `region_id`."""
    in_region = region_map == region_id
    ref_r = xp.where(in_region, pair.ref, 0)
    pred_r = xp.where(in_region, pair.pred, 0)
    return UnmatchedInstancePair(
        ref=ref_r,
        pred=pred_r,
        n_ref=len(_nonzero_labels(ref_r, xp)),
        n_pred=len(_nonzero_labels(pred_r, xp)),
        spacing=spacing,
    )


def evaluate_regionwise(
    pair: UnmatchedInstancePair,
    metrics: list[str],
    xp: Xp,
    *,
    matcher: str = "naive",
    matching_metric: str = "IOU",
    matching_threshold: float = 0.5,
    global_metrics: list[str] | None = None,
    spacing: tuple[float, ...] | None = None,
    n_jobs: int = 1,
    strict_threshold: bool = False,
    **cfg: object,
) -> dict:
    """Evaluate `pair` per Voronoi region seeded by the reference instances.

    Returns ``{"regions": {ref_id: result-dict}}`` plus ``region_avg_global_bin_
    <id>`` keys (the mean of each region's global binarized metric), matching
    v1's region-wise combining.
    """
    spacing = spacing if spacing is not None else pair.spacing
    region_map, n_regions = voronoi_regions(pair.ref, xp, spacing=spacing)

    regions: dict[int, dict] = {}
    for region_id in range(1, n_regions + 1):
        region_pair = _region_unmatched_pair(pair, xp, region_map, region_id, spacing)
        matched = match(
            region_pair,
            xp,
            algorithm=matcher,
            matching_metric=matching_metric,
            matching_threshold=matching_threshold,
            strict=strict_threshold,
        )
        regions[region_id] = _evaluate(
            matched,
            metrics,
            xp,
            spacing=spacing,
            global_metrics=global_metrics,
            n_jobs=n_jobs,
        )

    out: dict = {"regions": regions}
    for gid in global_metrics or []:
        key = f"global_bin_{gid.lower()}"
        vals = [regions[i][key] for i in regions] if regions else []
        out[f"region_avg_{key}"] = (sum(vals) / len(vals)) if vals else 0.0
    return out


__all__ = ["evaluate_regionwise"]
