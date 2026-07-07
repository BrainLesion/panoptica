"""Thin result container wrapping the pipeline's flat metric dict.

Implements the ``Serializable`` protocol so io/file_backend can persist it
without importing this module (avoids a circular import).
"""

from __future__ import annotations

from typing import Any


class EvalResult:
    """Immutable-ish view over the flat metric dict produced by pipeline.evaluate."""

    def __init__(self, values: dict[str, Any], *, device: str = "cpu") -> None:
        self._values = dict(values)
        self.device = device

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._values.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        """Serializable protocol hook."""
        return dict(self._values)

    def keys(self):
        return self._values.keys()

    def to_frame(self):
        """Single-row pandas DataFrame of the scalar metrics (lists dropped)."""
        import pandas as pd

        scalars = {
            k: v for k, v in self._values.items() if not isinstance(v, (list, tuple))
        }
        return pd.DataFrame([scalars])

    def __repr__(self) -> str:
        scalar = {
            k: v for k, v in self._values.items() if not isinstance(v, (list, tuple))
        }
        return f"EvalResult(device={self.device!r}, {scalar})"
