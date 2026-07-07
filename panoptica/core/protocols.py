"""Structural typing seams shared by every layer.

`Array` and `Xp` are duck-typed: an array is any numpy/cupy ndarray, and `Xp`
is the numpy-or-cupy module resolved by `backends.namespace.resolve`. We annotate
as `Any` in code (numpy and cupy are not a common static type) but name them here
so signatures read clearly and reviewers know the intent.

Compute code receives `xp` and uses it; it must NOT import numpy or cupy
directly. Only backends/ and CPU-pinned kernels import concrete libs.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

# An ndarray from the active backend (numpy.ndarray | cupy.ndarray).
Array = Any

# The resolved array namespace module (numpy | cupy).
Xp = Any


@runtime_checkable
class Serializable(Protocol):
    """Anything the io/file_backend layer can persist without importing it."""

    def to_dict(self) -> dict[str, Any]: ...


class MetricFn(Protocol):
    """The batched-metric call signature. One value per instance."""

    def __call__(
        self,
        ref: Array,
        pred: Array,
        ref_ids: Array,
        pred_ids: Array,
        xp: Xp,
        *,
        spacing: tuple[float, ...] | None = None,
        **params: Any,
    ) -> Array: ...
