"""Declarative metric registry: METRIC_REGISTRY, Metric enum, aggregation, PQ/SQ/RQ.

Every batched metric registers itself here via ``@register(...)``; the ``Metric``
enum is derived from the registry so callers can do ``Metric.DSC`` etc.

This module is the single place that imports the concrete ``*_batched`` metric
modules (volumetric/surface/topology/center) in order to trigger their
``@register`` calls before deriving the ``Metric`` enum. Those modules import
``register``/``MetricSpec`` back from here; this is a standard "registration on
import" pattern, not a layering violation (metrics/ only imports metrics/).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # The concrete ``Metric`` enum is built lazily via module ``__getattr__``
    # (see below), so it has no static binding; alias it for type-checkers/linters.
    Metric = Enum

from panoptica.core.edge_cases import EdgeCaseResult
from panoptica.core.enums import Direction, MetricMode, MetricType
from panoptica.core.errors import InputValidationError
from panoptica.core.protocols import Array, Xp

#: Maps the core EdgeCaseResult sentinel to its concrete float value.
_EDGE_VALUE: dict[EdgeCaseResult, float] = {
    EdgeCaseResult.ZERO: 0.0,
    EdgeCaseResult.ONE: 1.0,
    EdgeCaseResult.NAN: float("nan"),
    EdgeCaseResult.INF: float("inf"),
}


def _edge_value(result: EdgeCaseResult | None) -> float:
    if result is None:
        return float("nan")
    if result not in _EDGE_VALUE:
        raise InputValidationError(f"Unsupported EdgeCaseResult: {result!r}")
    return _EDGE_VALUE[result]


@dataclass(frozen=True)
class ZeroTPPolicy:
    """Resolves what an instance metric's SQ should be when the matched-pairs
    list is empty (``tp == 0``), split by *why* it's empty. ``default`` is used
    for any scenario whose specific field is left ``None``.
    """

    default: EdgeCaseResult = EdgeCaseResult.NAN
    no_instances: EdgeCaseResult | None = None  # both n_ref == 0 and n_pred == 0
    empty_ref: EdgeCaseResult | None = None  # n_ref == 0, n_pred > 0
    empty_pred: EdgeCaseResult | None = None  # n_pred == 0, n_ref > 0
    normal: EdgeCaseResult | None = None  # n_ref > 0 and n_pred > 0 but tp == 0

    def resolve(self, n_ref: int, n_pred: int) -> float:
        if n_ref + n_pred == 0:
            return _edge_value(self.no_instances or self.default)
        if n_ref == 0:
            return _edge_value(self.empty_ref or self.default)
        if n_pred == 0:
            return _edge_value(self.empty_pred or self.default)
        return _edge_value(self.normal or self.default)


@dataclass(frozen=True)
class MetricSpec:
    """One row of the metric registry.

    Attributes: a short id, a long/display name, the direction
    (higher-is-better vs lower-is-better), the batched compute function,
    whether it is CPU-pinned, and the zero-TP edge-case policy used to derive
    SQ when no pairs matched.
    """

    id: str
    type: MetricType
    direction: Direction
    fn: Callable[..., Array]
    cpu_only: bool = False
    short_name: str | None = None
    long_name: str = ""
    zero_tp: ZeroTPPolicy = field(default_factory=ZeroTPPolicy)

    def __post_init__(self) -> None:
        if not self.short_name:
            object.__setattr__(self, "short_name", self.id)
        if not self.long_name:
            object.__setattr__(self, "long_name", self.id)

    @property
    def increasing(self) -> bool:
        return self.direction is Direction.INCREASING


#: The live registry, populated by @register calls in volumetric/surface/topology/center.
METRIC_REGISTRY: dict[str, MetricSpec] = {}


def register(
    *,
    id: str,
    type: MetricType,
    direction: Direction,
    cpu_only: bool = False,
    short_name: str | None = None,
    long_name: str = "",
    zero_tp: ZeroTPPolicy | None = None,
) -> Callable[[Callable[..., Array]], Callable[..., Array]]:
    """Decorator: register a ``*_batched`` function as metric ``id`` in METRIC_REGISTRY.

    Usage::

        @register(id="DSC", type=MetricType.INSTANCE, direction=Direction.INCREASING)
        def dice_batched(ref, pred, ref_ids, pred_ids, xp, *, spacing=None, **params): ...
    """

    def _decorate(fn: Callable[..., Array]) -> Callable[..., Array]:
        if id in METRIC_REGISTRY:
            raise InputValidationError(f"Metric id {id!r} already registered")
        METRIC_REGISTRY[id] = MetricSpec(
            id=id,
            type=type,
            direction=direction,
            fn=fn,
            cpu_only=cpu_only,
            short_name=short_name,
            long_name=long_name,
            zero_tp=zero_tp or ZeroTPPolicy(),
        )
        return fn

    return _decorate


# Metric enum is built lazily to avoid an import cycle during registration.
_METRIC_ENUM: type | None = None


def _import_all_metric_modules() -> None:
    from panoptica.metrics import center, surface, topology, volumetric  # noqa: F401


def _metric_enum() -> type:
    global _METRIC_ENUM
    if _METRIC_ENUM is None:
        _import_all_metric_modules()
        # Dynamic functional Enum (members come from the runtime registry) can't
        # be statically validated.
        _METRIC_ENUM = Enum(  # pyrefly: ignore[invalid-argument]
            "Metric", {mid: mid for mid in METRIC_REGISTRY}
        )
    return _METRIC_ENUM


def __getattr__(name: str):  # PEP 562 module-level lazy attribute
    if name == "Metric":
        return _metric_enum()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_spec(metric: Metric | str) -> MetricSpec:
    """Resolve a `Metric` enum member (or raw id string) to its `MetricSpec`."""
    mid = metric.value if isinstance(metric, Enum) else metric
    if mid not in METRIC_REGISTRY:
        raise InputValidationError(f"Unknown metric id: {mid!r}")
    return METRIC_REGISTRY[mid]


def aggregate(
    values: Array,
    mode: MetricMode,
    xp: Xp,
    *,
    empty_list_std: float = float("nan"),
) -> Array | float | list[float]:
    """Reduce a per-instance metric array under `MetricMode`.

    `values` holds only the TP-matched-pair values (list metrics are computed
    over TPs only). Reductions run in float64 regardless of the input dtype.
    """
    n = int(values.shape[0]) if hasattr(values, "shape") else len(values)
    if mode is MetricMode.ALL:
        return values
    values64 = xp.asarray(values, dtype=xp.float64)
    if mode is MetricMode.AVG:
        return float(xp.mean(values64)) if n > 0 else float("nan")
    if mode is MetricMode.SUM:
        return float(xp.sum(values64)) if n > 0 else float("nan")
    if mode is MetricMode.MIN:
        return float(xp.min(values64)) if n > 0 else float("nan")
    if mode is MetricMode.MAX:
        return float(xp.max(values64)) if n > 0 else float("nan")
    if mode is MetricMode.STD:
        return float(xp.std(values64)) if n > 0 else empty_list_std
    raise InputValidationError(f"Unsupported MetricMode: {mode!r}")


# Detection-quality derivation: TP/FP/FN -> prec/rec/RQ/SQ/PQ.
@dataclass(frozen=True)
class DetectionQuality:
    """precision / recall / RQ derived from TP/FP/FN."""

    prec: float
    rec: float
    rq: float


def derive_detection_quality(tp: int, fp: int, fn: int) -> DetectionQuality:
    """``prec = tp/(tp+fp)``, ``rec = tp/(tp+fn)``, returning NaN when the
    denominator is zero rather than raising ZeroDivisionError.

    ``RQ = tp/(tp + 0.5*fp + 0.5*fn)`` (Recognition Quality); when ``tp == 0``,
    RQ is ``0.0`` if any instances exist at all, else ``NaN``.
    """
    n_pred = tp + fp
    n_ref = tp + fn
    prec = float(tp) / n_pred if n_pred > 0 else float("nan")
    rec = float(tp) / n_ref if n_ref > 0 else float("nan")
    if tp == 0:
        rq = 0.0 if (n_pred + n_ref) > 0 else float("nan")
    else:
        rq = tp / (tp + 0.5 * fp + 0.5 * fn)
    return DetectionQuality(prec=prec, rec=rec, rq=rq)


def derive_sq(
    metric: Metric | str,
    tp_values: Array,
    xp: Xp,
    *,
    tp: int,
    n_ref: int,
    n_pred: int,
) -> float:
    """Segmentation Quality: mean of a metric's values over the TP-matched pairs.

    When ``tp == 0`` there are no values to average, so the metric's
    registered `ZeroTPPolicy` supplies the sentinel instead.
    """
    if tp == 0:
        spec = get_spec(metric)
        return spec.zero_tp.resolve(n_ref=n_ref, n_pred=n_pred)
    return float(aggregate(tp_values, MetricMode.AVG, xp))  # pyrefly: ignore


def derive_pq(sq: float, rq: float) -> float:
    """Panoptic Quality = SQ * RQ.

    No additional zero-TP override here: when tp==0, `sq` already carries the
    metric's edge-case sentinel and `rq` is 0.0 or NaN from
    `derive_detection_quality` (e.g. NaN * anything = NaN).
    """
    return sq * rq
