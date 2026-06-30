"""Configured metric: a :class:`Metric` bound to an evaluation mode and fixed params.

``ConfiguredMetric`` is the unit accepted by ``Panoptica_Evaluator(metrics=[...])``
introduced in the metric-system overhaul (#181). It binds a catalog ``Metric`` to a
single evaluation mode ("instance" or "global") and an optional set of fixed
parameters, e.g. ``Metric.NSD.instance(threshold=4)``.

A bare ``Metric`` passed in the ``metrics`` list is expanded to every mode the metric
supports; a ``ConfiguredMetric`` contributes only to its own mode.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from panoptica.metrics.metrics import Metric
from panoptica.utils.config import SupportsConfig

VALID_MODES: frozenset[str] = frozenset({"instance", "global"})


class ConfiguredMetric(SupportsConfig):
    """A :class:`Metric` bound to a mode and (optionally) fixed parameters.

    Args:
        metric: The catalog metric to configure.
        mode: Either ``"instance"`` or ``"global"``.
        params: Fixed parameters forwarded to the metric function on every call
            (e.g. ``{"threshold": 4}`` for NSD). Parameter names are validated
            against the metric's ``param_spec``.
    """

    def __init__(
        self,
        metric: Metric,
        mode: str = "instance",
        params: Mapping[str, Any] | tuple | None = None,
    ) -> None:
        if not isinstance(metric, Metric):
            raise TypeError(f"metric must be a Metric, got {type(metric)}")
        if mode not in VALID_MODES:
            raise ValueError(
                f"mode must be one of {sorted(VALID_MODES)}, got {mode!r}"
            )
        if mode not in metric.modes:
            raise ValueError(
                f"Metric {metric.name} does not support mode {mode!r} "
                f"(supports {sorted(metric.modes)})"
            )
        # Normalize params to a sorted tuple of (name, value) pairs so the object
        # is hashable and two configs with the same params compare equal.
        if params is None:
            items: tuple[tuple[str, Any], ...] = ()
        elif isinstance(params, Mapping):
            items = tuple(sorted(params.items()))
        else:
            items = tuple(sorted(dict(params).items()))
        for name, _ in items:
            if name not in metric.param_spec:
                raise ValueError(
                    f"Metric {metric.name} does not accept parameter {name!r}; "
                    f"accepted parameters: {metric.param_spec}"
                )
        self._metric = metric
        self._mode = mode
        self._params = items

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #
    @property
    def metric(self) -> Metric:
        return self._metric

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def params(self) -> tuple[tuple[str, Any], ...]:
        return self._params

    @property
    def param_dict(self) -> dict[str, Any]:
        return dict(self._params)

    @property
    def name(self) -> str:
        return self._metric.name

    @property
    def is_instance(self) -> bool:
        return self._mode == "instance"

    @property
    def is_global(self) -> bool:
        return self._mode == "global"

    def get_result_key(self, prefix: str, is_std: bool = False) -> str:
        """Result-dict key for this configured metric.

        Identical to the underlying metric's key while no parameters are set; the
        parameter suffix scheme is introduced in PR3.
        """
        return self._metric.get_result_key(prefix, is_std)

    def __call__(
        self,
        reference_arr,
        prediction_arr,
        ref_instance_idx=None,
        pred_instance_idx=None,
        *args,
        **kwargs,
    ):
        merged = {**self.param_dict, **kwargs}
        return self._metric(
            reference_arr,
            prediction_arr,
            ref_instance_idx,
            pred_instance_idx,
            *args,
            **merged,
        )

    # ------------------------------------------------------------------ #
    # Identity
    # ------------------------------------------------------------------ #
    def __eq__(self, other: object) -> bool:
        if isinstance(other, ConfiguredMetric):
            return (
                self._metric == other._metric
                and self._mode == other._mode
                and self._params == other._params
            )
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self._metric.name, self._mode, self._params))

    def __repr__(self) -> str:
        if self._params:
            return (
                f"ConfiguredMetric({self._metric.name}, {self._mode}, "
                f"{self.param_dict})"
            )
        return f"ConfiguredMetric({self._metric.name}, {self._mode})"

    # ------------------------------------------------------------------ #
    # YAML (inherited from_yaml does ``cls(**data)``)
    # ------------------------------------------------------------------ #
    @classmethod
    def _yaml_repr(cls, node) -> dict:
        d: dict[str, Any] = {"metric": node._metric, "mode": node._mode}
        if node._params:
            d["params"] = node.param_dict
        return d


def InstanceMetric(metric: Metric, **params: Any) -> ConfiguredMetric:
    """Factory: configure ``metric`` for instance-wise evaluation."""
    return ConfiguredMetric(metric, "instance", params)


def GlobalMetric(metric: Metric, **params: Any) -> ConfiguredMetric:
    """Factory: configure ``metric`` for global (whole-image binary) evaluation."""
    return ConfiguredMetric(metric, "global", params)
