"""Shared test fixtures: device parametrization + seeded RNG.

Every kernel/metric test parametrizes over `device` so the same assertions run on
CPU (numpy) and, when a GPU is present, CUDA (cupy). Tolerances come from the
measured spike (SPEC §8.2); until then use the starting values below.
"""

from __future__ import annotations

import numpy as np
import pytest

from panoptica.backends.namespace import _cupy_available, resolve

_DEVICES = ["cpu"] + (["cuda"] if _cupy_available() else [])

# Starting tolerances (replaced by bench/parity/tolerances.py once measured).
TOL_VOLUMETRIC = dict(rtol=1e-9, atol=0.0)
TOL_SURFACE = dict(rtol=0.0, atol=1e-4)  # voxels; mixed precision keeps ~double


@pytest.fixture(params=_DEVICES)
def device(request) -> str:
    """Yields 'cpu' then (if available) 'cuda'."""
    return request.param


@pytest.fixture
def xp(device):
    """The resolved array namespace for the current device."""
    return resolve(device)[0]


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded host RNG — deterministic across runs."""
    return np.random.default_rng(0)
