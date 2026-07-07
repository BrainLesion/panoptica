"""The batched GPU metric paths must equal the per-instance reducers.

`surface_batched.batched_surface_scalars` and `center.batched_center_distance`
are what run on the cupy path; in production they're gated to cupy, but the
functions themselves are `xp`-generic, so we exercise them here with the `xp`
fixture (numpy on CPU, cupy when a GPU is present) and assert they match the
per-instance reducers value-for-value. This gives the batched algorithm real
coverage even in a GPU-less CI, and doubles as a cpu/cuda parity check.
"""

from __future__ import annotations

import numpy as np
import pytest

from panoptica.metrics.center import batched_center_distance, center_distance_batched
from panoptica.metrics.surface import (
    assd_batched,
    hd95_batched,
    hd_batched,
    nsd_batched,
)
from panoptica.metrics.surface_batched import batched_surface_scalars


def _blobs(seed: int) -> np.ndarray:
    a = np.zeros((40, 40, 40), np.int64)
    r = np.random.default_rng(seed)
    zz, yy, xx = np.ogrid[:40, :40, :40]
    lab = 1
    for _ in range(8):
        c = r.integers(4, 36, 3)
        rad = int(r.integers(3, 6))
        m = (zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2 <= rad**2
        sel = m & (a == 0)
        if sel.any():
            a[sel] = lab
            lab += 1
    return a


def _pair(xp):
    ref = _blobs(1)
    pred = np.roll(_blobs(1), 1, axis=0)  # small shift -> nonzero distances
    ids = [i for i in range(1, ref.max() + 1) if (ref == i).any() and (pred == i).any()]
    idar = np.asarray(ids, dtype=np.int64)
    return xp.asarray(ref), xp.asarray(pred), xp.asarray(idar)


def _host(a):
    return np.asarray(a.get() if hasattr(a, "get") else a, dtype=np.float64)


@pytest.mark.parametrize("spacing", [None, (2.0, 1.0, 0.5)])
def test_batched_surface_equals_per_instance(xp, spacing):
    ref, pred, ids = _pair(xp)
    want = ["ASSD", "HD", "HD95", "NSD"]
    batched = batched_surface_scalars(ref, pred, ids, ids, xp, spacing, want)
    per = {
        "ASSD": assd_batched(ref, pred, ids, ids, xp, spacing=spacing),
        "HD": hd_batched(ref, pred, ids, ids, xp, spacing=spacing),
        "HD95": hd95_batched(ref, pred, ids, ids, xp, spacing=spacing),
        "NSD": nsd_batched(ref, pred, ids, ids, xp, spacing=spacing),
    }
    for k in want:
        b, p = _host(batched[k]), _host(per[k])
        # isotropic is bit-identical; anisotropic differs only by fp32 EDT vs
        # the matmul-NN (well under the surface tolerance).
        assert np.allclose(b, p, atol=1e-4, equal_nan=True), f"{k} spacing={spacing}"


@pytest.mark.parametrize("spacing", [None, (2.0, 1.0, 0.5)])
def test_batched_cedi_equals_per_instance(xp, spacing):
    ref, pred, ids = _pair(xp)
    b = _host(batched_center_distance(ref, pred, ids, ids, xp, spacing))
    p = _host(center_distance_batched(ref, pred, ids, ids, xp, spacing=spacing))
    assert np.allclose(b, p, atol=1e-6, equal_nan=True)


def test_batched_surface_empty_ids(xp):
    ref, pred, _ = _pair(xp)
    empty = xp.zeros((0,), dtype=xp.int64)
    out = batched_surface_scalars(ref, pred, empty, empty, xp, None, ["ASSD"])
    assert out["ASSD"].shape[0] == 0
