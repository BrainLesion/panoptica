"""Unit tests for surface-distance instance metrics (ASSD/HD/HD95/NSD).

`panoptica.kernels.edt`/`surface_border` are owned by a parallel stream and may not
exist yet. Rather than skip outright, we monkeypatch a scipy-based reference
implementation into `panoptica.kernels` (standard border/EDT conventions:
connectivity-1 binary erosion for the border, `distance_transform_edt` for the
field) so these tests still exercise our reducer code in `panoptica.metrics.surface`.
If `panoptica.kernels.edt`/`surface_border` are later implemented for real, this
fixture simply stops overriding anything meaningful once the real ones match.
"""

from __future__ import annotations

import numpy as np
import pytest

scipy_ndimage = pytest.importorskip("scipy.ndimage")

from panoptica.metrics.surface import (  # noqa: E402
    assd_batched,
    hd95_batched,
    hd_batched,
    nsd_batched,
)


def _np(arr):
    """Duck-typed device->host transfer for assertions (cupy has `.get()`)."""
    getter = getattr(arr, "get", None)
    return getter() if callable(getter) else np.asarray(arr)


def _fallback_surface_border(mask, xp):
    mask_np = _np(mask).astype(bool)
    footprint = scipy_ndimage.generate_binary_structure(mask_np.ndim, 1)
    eroded = scipy_ndimage.binary_erosion(mask_np, structure=footprint, iterations=1)
    return xp.asarray(mask_np ^ eroded)


def _fallback_edt(mask, xp, *, spacing=None):
    mask_np = _np(mask).astype(bool)
    return xp.asarray(scipy_ndimage.distance_transform_edt(mask_np, sampling=spacing))


@pytest.fixture(autouse=True)
def _inject_fallback_kernels(monkeypatch):
    import panoptica.kernels as kernels_mod

    monkeypatch.setattr(kernels_mod, "edt", _fallback_edt, raising=False)
    monkeypatch.setattr(
        kernels_mod, "surface_border", _fallback_surface_border, raising=False
    )
    yield


def _single_voxel_pair(xp):
    # ref: single voxel at (2, 2); pred: single voxel at (2, 4) -> Euclidean
    # distance 2.0. A lone voxel's "border" (mask XOR erosion) is the voxel
    # itself, so this is hand-computable without a real distance-transform.
    ref = np.zeros((5, 5), dtype=np.int32)
    pred = np.zeros((5, 5), dtype=np.int32)
    ref[2, 2] = 1
    pred[2, 4] = 1
    return xp.asarray(ref), xp.asarray(pred)


def test_assd_single_voxel_pair(device, xp):
    ref, pred = _single_voxel_pair(xp)
    out = assd_batched(ref, pred, xp.asarray([1]), xp.asarray([1]), xp)
    np.testing.assert_allclose(_np(out), [2.0], atol=1e-6)


def test_hd_single_voxel_pair(device, xp):
    ref, pred = _single_voxel_pair(xp)
    out = hd_batched(ref, pred, xp.asarray([1]), xp.asarray([1]), xp)
    np.testing.assert_allclose(_np(out), [2.0], atol=1e-6)


def test_hd95_single_voxel_pair(device, xp):
    ref, pred = _single_voxel_pair(xp)
    out = hd95_batched(ref, pred, xp.asarray([1]), xp.asarray([1]), xp)
    np.testing.assert_allclose(_np(out), [2.0], atol=1e-6)


def test_nsd_below_threshold_is_zero(device, xp):
    ref, pred = _single_voxel_pair(xp)
    # default threshold=0.5, distance=2.0 > 0.5 -> both sides count as fail
    out = nsd_batched(ref, pred, xp.asarray([1]), xp.asarray([1]), xp)
    np.testing.assert_allclose(_np(out), [0.0], atol=1e-6)


def test_nsd_above_threshold_is_one(device, xp):
    ref, pred = _single_voxel_pair(xp)
    out = nsd_batched(ref, pred, xp.asarray([1]), xp.asarray([1]), xp, threshold=3.0)
    np.testing.assert_allclose(_np(out), [1.0], atol=1e-6)


def test_assd_identical_masks_is_zero(device, xp):
    ref = np.zeros((5, 5), dtype=np.int32)
    ref[1:4, 1:4] = 1
    ref_x = xp.asarray(ref)
    out = assd_batched(ref_x, ref_x, xp.asarray([1]), xp.asarray([1]), xp)
    np.testing.assert_allclose(_np(out), [0.0], atol=1e-6)
