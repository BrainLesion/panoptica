"""Hand-computed unit tests for volumetric instance metrics (Dice/IoU/RVD/RVAE).

Every expected value below is derived by hand from the tiny label arrays, not
by re-deriving the v1 formula in the assertion (that would be circular).
"""

from __future__ import annotations

import numpy as np
import pytest

from panoptica.metrics.volumetric import (
    dice_batched,
    iou_batched,
    rvae_batched,
    rvd_batched,
)


def _np(arr):
    """Duck-typed device->host transfer for assertions (cupy has `.get()`)."""
    getter = getattr(arr, "get", None)
    return getter() if callable(getter) else np.asarray(arr)


# Fixture geometry: a 4x6 grid with three label pairs:
#   label 1: ref is a 2x2 block (4 voxels), pred is 3 of those 4 voxels
#            -> intersection=3, ref_sum=4, pred_sum=3
#   label 2: both empty (no voxels anywhere)                -> both zero
#   label 3: ref has 2 voxels, pred has 2 disjoint voxels    -> intersection=0


def _build_arrays():
    ref = np.zeros((4, 6), dtype=np.int32)
    pred = np.zeros((4, 6), dtype=np.int32)

    # label 1: ref = 2x2 block at rows0-1,cols0-1; pred = same minus one corner
    ref[0:2, 0:2] = 1
    pred[0, 0] = 1
    pred[0, 1] = 1
    pred[1, 0] = 1
    # pred[1, 1] intentionally left as background -> pred_sum = 3

    # label 2: no voxels assigned anywhere (both empty)

    # label 3: ref two voxels, pred two disjoint voxels
    ref[3, 0] = 3
    ref[3, 1] = 3
    pred[2, 4] = 3
    pred[2, 5] = 3

    return ref, pred


@pytest.fixture
def arrays():
    return _build_arrays()


def _ids():
    ref_ids = np.array([1, 2, 3])
    pred_ids = np.array([1, 2, 3])
    return ref_ids, pred_ids


def test_dice_batched(device, xp, arrays):
    ref, pred = arrays
    ref_ids, pred_ids = _ids()
    ref_x = xp.asarray(ref)
    pred_x = xp.asarray(pred)
    out = dice_batched(ref_x, pred_x, xp.asarray(ref_ids), xp.asarray(pred_ids), xp)
    out = _np(out)
    expected = np.array([2 * 3 / (4 + 3), 0.0, 0.0])
    np.testing.assert_allclose(out, expected, rtol=1e-9)


def test_iou_batched(device, xp, arrays):
    ref, pred = arrays
    ref_ids, pred_ids = _ids()
    ref_x = xp.asarray(ref)
    pred_x = xp.asarray(pred)
    out = iou_batched(ref_x, pred_x, xp.asarray(ref_ids), xp.asarray(pred_ids), xp)
    out = _np(out)
    # union for label1 = 4 + 3 - 3 = 4 -> iou = 3/4
    expected = np.array([3 / 4, 0.0, 0.0])
    np.testing.assert_allclose(out, expected, rtol=1e-9)


def test_rvd_batched(device, xp, arrays):
    ref, pred = arrays
    ref_ids, pred_ids = _ids()
    ref_x = xp.asarray(ref)
    pred_x = xp.asarray(pred)
    out = rvd_batched(ref_x, pred_x, xp.asarray(ref_ids), xp.asarray(pred_ids), xp)
    out = _np(out)
    # label1: (3-4)/4 = -0.25 ; label3: (2-2)/2 = 0.0 (equal volume, disjoint)
    expected = np.array([-0.25, 0.0, 0.0])
    np.testing.assert_allclose(out, expected, rtol=1e-9)


def test_rvae_batched(device, xp, arrays):
    ref, pred = arrays
    ref_ids, pred_ids = _ids()
    ref_x = xp.asarray(ref)
    pred_x = xp.asarray(pred)
    out = rvae_batched(ref_x, pred_x, xp.asarray(ref_ids), xp.asarray(pred_ids), xp)
    out = _np(out)
    expected = np.array([0.25, 0.0, 0.0])
    np.testing.assert_allclose(out, expected, rtol=1e-9)


def test_dice_perfect_overlap(device, xp):
    ref = xp.asarray(np.array([[1, 1], [0, 0]], dtype=np.int32))
    pred = xp.asarray(np.array([[1, 1], [0, 0]], dtype=np.int32))
    out = _np(dice_batched(ref, pred, xp.asarray([1]), xp.asarray([1]), xp))
    np.testing.assert_allclose(out, [1.0], rtol=1e-9)


def test_mismatched_lengths_raise(device, xp):
    from panoptica.core.errors import InputValidationError

    ref = xp.asarray(np.zeros((2, 2), dtype=np.int32))
    pred = xp.asarray(np.zeros((2, 2), dtype=np.int32))
    with pytest.raises(InputValidationError):
        dice_batched(ref, pred, xp.asarray([1, 2]), xp.asarray([1]), xp)
