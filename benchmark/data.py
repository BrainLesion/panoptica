"""Deterministic synthetic multi-instance volumes for benchmarks and unit tests.

The generators here are seeded (via ``np.random.default_rng``) so that a given
:class:`SyntheticCase` produces the same ``(prediction, reference)`` pair every run —
comparing timings before/after a change is only meaningful when the input is fixed.

To point the benchmark at real data later, implement a :class:`DataProvider` that
yields ``(name, prediction, reference)`` tuples (any dtype, any dimensionality) and
pass it wherever the benchmark accepts a provider.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator, Protocol, Tuple

import numpy as np


def _ball(radius: int, ndim: int) -> np.ndarray:
    """Boolean ``(2*radius+1)**ndim`` ball mask."""
    line = np.arange(-radius, radius + 1)
    grids = np.meshgrid(*([line] * ndim), indexing="ij")
    dist2 = sum(g.astype(np.int64) ** 2 for g in grids)
    return dist2 <= radius * radius


def make_reference(
    shape: tuple[int, ...],
    n_instances: int,
    rng: np.random.Generator,
    rmin: int = 3,
    rmax: int = 7,
) -> tuple[np.ndarray, int]:
    """Place up to ``n_instances`` non-overlapping balls with distinct labels."""
    arr = np.zeros(shape, dtype=np.int32)
    ndim = len(shape)
    label = 0
    attempts = 0
    while label < n_instances and attempts < n_instances * 25:
        attempts += 1
        r = int(rng.integers(rmin, rmax + 1))
        center = [int(rng.integers(r + 1, s - r - 1)) for s in shape]
        ball = _ball(r, ndim)
        slices = tuple(slice(c - r, c + r + 1) for c in center)
        region = arr[slices]
        if np.any(region[ball] != 0):
            continue  # keep instances separate
        label += 1
        region[ball] = label
    return arr, label


def make_prediction(
    ref: np.ndarray,
    rng: np.random.Generator,
    shift: int = 1,
    drop_frac: float = 0.1,
) -> np.ndarray:
    """Perturb the reference into a plausible prediction (shift + dropped instances)."""
    pred = ref.copy()
    for ax in range(ref.ndim):
        pred = np.roll(pred, int(rng.integers(-shift, shift + 1)), axis=ax)
    labels = [int(x) for x in np.unique(pred) if x > 0]
    n_drop = int(len(labels) * drop_frac)
    if n_drop:
        for label in rng.choice(labels, size=n_drop, replace=False):
            pred[pred == label] = 0
    return pred


@dataclass(frozen=True)
class SyntheticCase:
    """A single seeded synthetic benchmark configuration."""

    name: str
    shape: tuple[int, ...]
    n_instances: int
    seed: int = 0

    def build(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(prediction, reference)`` as ``int32`` label arrays."""
        rng = np.random.default_rng(self.seed)
        ref, _ = make_reference(self.shape, self.n_instances, rng)
        pred = make_prediction(ref, rng)
        return pred, ref


class DataProvider(Protocol):
    """Yields ``(name, prediction, reference)`` triples for the benchmark harness.

    Any object with a ``cases()`` generator matching this signature is a valid
    provider — swap :class:`SyntheticDataProvider` for a real-data implementation
    without touching the benchmark or the unit test.
    """

    def cases(self) -> Iterator[Tuple[str, np.ndarray, np.ndarray]]: ...


@dataclass
class SyntheticDataProvider:
    """Yields deterministic synthetic cases built from :class:`SyntheticCase`."""

    configs: Iterable[SyntheticCase] = field(default_factory=tuple)

    def cases(self) -> Iterator[Tuple[str, np.ndarray, np.ndarray]]:
        for cfg in self.configs:
            pred, ref = cfg.build()
            yield cfg.name, pred, ref


def default_benchmark_cases(quick: bool = False) -> tuple[SyntheticCase, ...]:
    """The benchmark's canonical configs. ``quick=True`` swaps in smaller sizes."""
    if quick:
        return (
            SyntheticCase("2D quick", (256, 256), 40),
            SyntheticCase("3D quick", (96, 96, 96), 30),
        )
    return (
        SyntheticCase("2D many-instance", (512, 512), 200),
        SyntheticCase("3D medium", (160, 160, 160), 120),
    )


def default_unit_test_cases() -> tuple[SyntheticCase, ...]:
    """Small configs suitable for CI unit tests (few seconds total runtime)."""
    return (
        SyntheticCase("2D small", (64, 64), 5),
        SyntheticCase("2D larger", (128, 128), 30),
        SyntheticCase("3D small", (32, 32, 32), 5),
        SyntheticCase("3D larger", (48, 48, 48), 15),
    )
