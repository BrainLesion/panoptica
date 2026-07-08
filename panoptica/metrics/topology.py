"""Centerline Dice (clDSC) via skeletonization -- CPU-only.

Skeletonization has no GPU implementation, so this metric always runs on
host: if `ref`/`pred` are cupy arrays they are transferred to numpy first via
duck-typing (checking for a `.get()` method), never by unconditionally
importing cupy at module level.
"""

from __future__ import annotations

from panoptica.core.edge_cases import EdgeCaseResult
from panoptica.core.enums import Direction, MetricType
from panoptica.core.errors import InputValidationError, MetricComputeError
from panoptica.core.protocols import Array, Xp
from panoptica.metrics.registry import ZeroTPPolicy, register


def _to_host(arr: Array) -> Array:
    """Transfer a (possibly cupy) array to a numpy array, without importing cupy.

    Duck-typed: any array exposing a no-arg `.get()` (cupy's device->host
    transfer) is assumed to need it; plain numpy arrays have no such method.
    """
    getter = getattr(arr, "get", None)
    if callable(getter):
        try:
            return getter()
        except Exception as e:  # pragma: no cover - depends on GPU availability
            raise MetricComputeError(
                "Failed to transfer array to host for CPU-only clDSC skeletonization"
            ) from e
    return arr


def _get_skeleton(mask):
    try:
        from skimage.morphology import skeletonize
    except ImportError as e:  # pragma: no cover - depends on optional dependency
        raise MetricComputeError(
            "scikit-image is required to compute clDSC (skeletonize)"
        ) from e
    try:
        from skimage.morphology import skeletonize_3d  # type: ignore[attr-defined]
    except ImportError:
        skeletonize_3d = None

    if mask.ndim == 2:
        return skeletonize(mask)
    if mask.ndim == 3:
        return skeletonize_3d(mask) if skeletonize_3d is not None else skeletonize(mask)
    raise InputValidationError(f"Unsupported array dimension for clDSC: {mask.ndim}")


def _cl_score(volume, skeleton):
    import numpy as np

    # numpy float division: an empty skeleton (sum == 0) yields nan via 0/0
    # rather than a ZeroDivisionError.
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.float64(np.sum(volume * skeleton)) / np.float64(np.sum(skeleton))


def _host_union_box(r, p):
    """Padded (1 voxel) bounding box of (r | p) in host numpy, or None if empty."""
    import numpy as np

    union = r | p
    if not union.any():
        return None
    box = []
    for axis, c in enumerate(np.where(union)):
        lo = int(c.min()) - 1
        hi = int(c.max()) + 2
        box.append(slice(max(lo, 0), min(hi, r.shape[axis])))
    return tuple(box)


@register(
    id="clDSC",
    type=MetricType.INSTANCE,
    direction=Direction.INCREASING,
    long_name="Centerline Dice",
    cpu_only=True,
    zero_tp=ZeroTPPolicy(default=EdgeCaseResult.ZERO, no_instances=EdgeCaseResult.NAN),
)
def cldice_batched(
    ref: Array,
    pred: Array,
    ref_ids: Array,
    pred_ids: Array,
    xp: Xp,
    *,
    spacing: tuple[float, ...] | None = None,
    **params: object,
) -> Array:
    """clDSC = 2*tprec*tsens / (tprec + tsens), CPU-only.

    `xp` is accepted for signature uniformity, but the compute path is always
    host numpy: inputs are transferred down (via `_to_host`) before skeletonizing.
    """
    import numpy as np

    if len(ref_ids) != len(pred_ids):
        raise InputValidationError(
            f"ref_ids and pred_ids must be positionally aligned "
            f"(got {len(ref_ids)} vs {len(pred_ids)})"
        )
    ref_host = _to_host(ref)
    pred_host = _to_host(pred)
    ref_ids_host = _to_host(ref_ids)
    pred_ids_host = _to_host(pred_ids)

    k = len(ref_ids_host)
    crops = params.get("_crops")
    # On the GPU path the shared `_crops` are cupy arrays; pulling them down one
    # instance at a time is thousands of tiny D->H syncs. Since clDSC is host-only
    # anyway, derive the crops once on the host (one `find_objects` pass over the
    # already-transferred ref/pred) instead — no per-instance transfer.
    if crops is not None and k > 0 and hasattr(crops[0][0], "get"):  # pyrefly: ignore
        import numpy as np

        from panoptica.metrics._masks import precompute_crops

        crops = precompute_crops(ref_host, pred_host, ref_ids_host, pred_ids_host, np)

    def _one(i: int) -> float:
        if crops is not None:
            r = crops[i][0]  # pyrefly: ignore
            p = crops[i][1]  # pyrefly: ignore
        else:
            r = ref_host == ref_ids_host[i]
            p = pred_host == pred_ids_host[i]
            box = _host_union_box(r, p)
            if box is not None:
                r = r[box]
                p = p[box]
        tprec = _cl_score(p, _get_skeleton(r))
        tsens = _cl_score(r, _get_skeleton(p))
        # 0/0 yields nan for degenerate instances with vanished skeletons.
        with np.errstate(divide="ignore", invalid="ignore"):
            return 2.0 * tprec * tsens / (tprec + tsens)

    from panoptica.metrics._parallel import parallel_list

    n_jobs = int(params.get("_n_jobs", 1))  # pyrefly: ignore
    out = np.asarray(parallel_list(_one, range(k), n_jobs), dtype=np.float64)
    return xp.asarray(out, dtype=xp.float64)
