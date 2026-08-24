"""Per-phase wall-clock timing for the panoptic evaluation pipeline."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter


class PhaseTimer:
    """Accumulating wall-clock timer for named phases.

    Entering the ``time(name)`` context measures elapsed ``perf_counter`` seconds and
    adds them to ``times[name]``. Re-entering the same name accumulates rather than
    overwriting, so callers can time the same phase across multiple invocations
    (e.g. region-wise evaluation runs matching/evaluation once per region).
    """

    def __init__(self) -> None:
        self.times: dict[str, float] = {}

    @contextmanager
    def time(self, name: str) -> Iterator[None]:
        start = perf_counter()
        try:
            yield
        finally:
            self.times[name] = self.times.get(name, 0.0) + (perf_counter() - start)

    def record(self, name: str, seconds: float) -> None:
        """Add a pre-measured duration under ``name`` (accumulates)."""
        self.times[name] = self.times.get(name, 0.0) + float(seconds)
