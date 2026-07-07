"""Public entry point: Evaluator orchestrates convert -> route -> evaluate.

Routes an input pair by its InputType through the pipeline stages and returns
an EvalResult.
"""

from __future__ import annotations

import os
from typing import Any

from panoptica.api.result import EvalResult
from panoptica.backends.device import to_device
from panoptica.backends.namespace import resolve
from panoptica.core.enums import InputType
from panoptica.core.errors import InputValidationError
from panoptica.core.pairs import (
    MatchedInstancePair,
    SemanticPair,
    UnmatchedInstancePair,
)
from panoptica.core.protocols import Array, Xp
from panoptica.io.sanity import sanity_check
from panoptica.metrics import Metric
from panoptica.pipeline.approximate import approximate
from panoptica.pipeline.evaluate import evaluate
from panoptica.pipeline.match import match

_DEFAULT_INSTANCE = ("DSC", "IOU", "ASSD", "RVD")
_DEFAULT_GLOBAL = ("DSC",)


def _ids(labels: list[int] | tuple[int, ...]) -> set[int]:
    return {int(x) for x in labels if int(x) != 0}


def _unique_nonzero(arr: Array, xp: Xp) -> list[int]:
    u = xp.unique(arr)
    u = u[u != 0]
    host = u.get() if hasattr(u, "get") else u
    return [int(x) for x in host]


def _as_metric_ids(metrics) -> list[str]:
    out = []
    for m in metrics:
        out.append(m.name if isinstance(m, Metric) else str(m))
    return out


class Evaluator:
    """Compute instance-wise segmentation metrics for a prediction/reference pair.

    Args:
        expected_input: which pipeline phase the input enters at.
        instance_metrics / global_metrics: metric ids or Metric enum members.
        matcher / matching_metric / matching_threshold: matching config for
            SEMANTIC/UNMATCHED input.
        device: "auto" | "cpu" | "cuda[:i]".
    """

    def __init__(
        self,
        expected_input: InputType = InputType.MATCHED_INSTANCE,
        *,
        instance_metrics=_DEFAULT_INSTANCE,
        global_metrics=_DEFAULT_GLOBAL,
        matcher: str = "naive",
        matching_metric: str = "IOU",
        matching_threshold: float = 0.5,
        strict_threshold: bool = False,
        connectivity: int | None = None,
        device: str = "auto",
        n_jobs: int | None = None,
        segmentation_class_groups=None,
        per_region_evaluation: bool = False,
        **cfg: Any,
    ) -> None:
        self.expected_input = expected_input
        self.instance_metrics = _as_metric_ids(instance_metrics)
        self.global_metrics = _as_metric_ids(global_metrics)
        self.matcher = matcher
        self.matching_metric = matching_metric
        self.matching_threshold = matching_threshold
        # Strict (``>``/``<``) matching threshold: threshold=0 rejects zero overlap.
        self.strict_threshold = strict_threshold
        self.connectivity = connectivity
        self.device = device
        # When set, evaluate() returns a per-group {name: EvalResult} dict via
        # the vectorized multi-group path (pipeline/grouped.py).
        self.segmentation_class_groups = segmentation_class_groups
        # When True, evaluate() partitions the volume into Voronoi regions seeded
        # by the reference instances and evaluates each region independently.
        self.per_region_evaluation = per_region_evaluation
        # Threads for the per-instance CPU loops (surface EDT, clDSC). Default:
        # all cores. -1 also means all cores; 1 disables threading.
        self.n_jobs = (os.cpu_count() or 1) if n_jobs in (None, -1) else n_jobs
        self.cfg = cfg

    @staticmethod
    def from_config(path) -> Evaluator:
        """Build an Evaluator from a native or v1-preset YAML config."""
        from panoptica.api.config import load_config

        return load_config(path)

    def to_config(self, path) -> None:
        """Write this Evaluator's config to a native YAML that round-trips."""
        from panoptica.api.config import save_config

        save_config(self, path)

    def evaluate(
        self,
        prediction: Array,
        reference: Array,
        *,
        spacing: tuple[float, ...] | None = None,
        skip_groups: list[str] | None = None,
    ) -> EvalResult | dict[str, EvalResult]:
        """Evaluate a prediction/reference pair.

        Returns a single :class:`EvalResult`, or -- when
        ``segmentation_class_groups`` is set -- a ``{group_name: EvalResult}``
        dict, one entry per class group. ``skip_groups`` names class groups to
        omit (not evaluated, not returned); unknown names are ignored with a
        warning. It has no effect unless ``segmentation_class_groups`` is set.
        """
        sanity_check(reference, prediction)
        xp, dev = resolve(self.device)
        ref = to_device(reference, dev)
        pred = to_device(prediction, dev)

        if self.segmentation_class_groups is not None:
            from panoptica.pipeline.grouped import evaluate_grouped

            grouped = evaluate_grouped(
                ref,
                pred,
                xp,
                input_type=self.expected_input,
                groups=self.segmentation_class_groups,
                instance_metrics=self.instance_metrics,
                global_metrics=self.global_metrics,
                matcher=self.matcher,
                matching_metric=self.matching_metric,
                matching_threshold=self.matching_threshold,
                connectivity=self.connectivity,
                spacing=spacing,
                n_jobs=self.n_jobs,
                strict_threshold=self.strict_threshold,
                skip_groups=self._resolve_skip_groups(skip_groups),
            )
            return {name: EvalResult(v, device=dev) for name, v in grouped.items()}

        if self.per_region_evaluation:
            from panoptica.pipeline.regionwise import evaluate_regionwise

            unmatched = self._to_unmatched(ref, pred, xp, spacing)
            out = evaluate_regionwise(
                unmatched,
                self.instance_metrics,
                xp,
                matcher=self.matcher,
                matching_metric=self.matching_metric,
                matching_threshold=self.matching_threshold,
                global_metrics=self.global_metrics,
                spacing=spacing,
                n_jobs=self.n_jobs,
                strict_threshold=self.strict_threshold,
            )
            result: dict[str, Any] = {
                "regions": {
                    rid: EvalResult(v, device=dev) for rid, v in out["regions"].items()
                }
            }
            result.update({k: v for k, v in out.items() if k != "regions"})
            return result

        matched = self._to_matched(ref, pred, xp, spacing)
        values = evaluate(
            matched,
            self.instance_metrics,
            xp,
            spacing=spacing,
            global_metrics=self.global_metrics,
            n_jobs=self.n_jobs,
        )
        return EvalResult(values, device=dev)

    def _to_matched(
        self, ref: Array, pred: Array, xp: Xp, spacing
    ) -> MatchedInstancePair:
        it = self.expected_input
        if it is InputType.SEMANTIC:
            unmatched = approximate(
                SemanticPair(ref=ref, pred=pred, spacing=spacing),
                xp,
                connectivity=self.connectivity,
            )
            return self._match(unmatched, xp)
        if it is InputType.UNMATCHED_INSTANCE:
            n_ref = len(_unique_nonzero(ref, xp))
            n_pred = len(_unique_nonzero(pred, xp))
            unmatched = UnmatchedInstancePair(
                ref=ref, pred=pred, n_ref=n_ref, n_pred=n_pred, spacing=spacing
            )
            return self._match(unmatched, xp)
        if it is InputType.MATCHED_INSTANCE:
            return self._matched_from_correspondence(ref, pred, xp, spacing)
        raise InputValidationError(f"Unknown InputType: {it!r}")

    def _to_unmatched(
        self, ref: Array, pred: Array, xp: Xp, spacing
    ) -> UnmatchedInstancePair:
        """Instance-labelled pair before matching (for region-wise partitioning)."""
        if self.expected_input is InputType.SEMANTIC:
            return approximate(
                SemanticPair(ref=ref, pred=pred, spacing=spacing),
                xp,
                connectivity=self.connectivity,
            )
        return UnmatchedInstancePair(
            ref=ref,
            pred=pred,
            n_ref=len(_unique_nonzero(ref, xp)),
            n_pred=len(_unique_nonzero(pred, xp)),
            spacing=spacing,
        )

    def _match(self, unmatched: UnmatchedInstancePair, xp: Xp) -> MatchedInstancePair:
        return match(
            unmatched,
            xp,
            algorithm=self.matcher,
            matching_metric=self.matching_metric,
            matching_threshold=self.matching_threshold,
            strict=self.strict_threshold,
        )

    def _resolve_skip_groups(self, skip_groups) -> set[str]:
        """Lowercased group names to skip; unknown names warned and dropped.

        A stale skip list never silently turns into "skip everything" or an
        error — unknown names are ignored so the remaining groups still run.
        """
        if not skip_groups:
            return set()
        sg = self.segmentation_class_groups
        known = {str(n).lower() for n in sg.groups} if sg is not None else set()
        requested = {str(s).lower() for s in skip_groups}
        unknown = sorted(requested - known)
        if unknown:
            import warnings

            warnings.warn(
                f"skip_groups contains unknown group name(s) {unknown}; "
                f"known groups are {sorted(known)}. They will be ignored."  # pyrefly: ignore
            )
        return requested & known

    def _matched_from_correspondence(
        self, ref: Array, pred: Array, xp: Xp, spacing
    ) -> MatchedInstancePair:
        """For MATCHED input, labels already correspond across ref/pred."""
        rset = _ids(_unique_nonzero(ref, xp))
        pset = _ids(_unique_nonzero(pred, xp))
        both = sorted(rset & pset)
        matched_ids = (
            xp.asarray([[i, i] for i in both], dtype="int64")
            if both
            else xp.zeros((0, 2), dtype="int64")
        )
        return MatchedInstancePair(
            ref=ref,
            pred=pred,
            matched_ids=matched_ids,
            unmatched_ref=tuple(sorted(rset - pset)),
            unmatched_pred=tuple(sorted(pset - rset)),
            spacing=spacing,
        )
