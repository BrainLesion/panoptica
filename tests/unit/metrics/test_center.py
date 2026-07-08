"""Hand-computed unit tests for center_distance_batched (CEDI).

Also verifies the purity fix: v1's center_distance mutates its input array in
place (`arr[arr != 0] = 1`); this batched version must leave ref/pred untouched.
"""

from __future__ import annotations

import numpy as np

from panoptica.metrics.center import center_distance_batched


def _np(arr):
    """Duck-typed device->host transfer for assertions (cupy has `.get()`)."""
    getter = getattr(arr, "get", None)
    return getter() if callable(getter) else np.asarray(arr)


def test_center_distance_simple(device, xp):
    # label 1: ref is a single voxel at (0, 0); pred is a single voxel at (0, 3)
    # -> center of mass ref=(0,0), pred=(0,3) -> distance = 3.0
    ref = np.zeros((4, 4), dtype=np.int32)
    pred = np.zeros((4, 4), dtype=np.int32)
    ref[0, 0] = 1
    pred[0, 3] = 1

    ref_x = xp.asarray(ref)
    pred_x = xp.asarray(pred)
    out = center_distance_batched(ref_x, pred_x, xp.asarray([1]), xp.asarray([1]), xp)
    np.testing.assert_allclose(_np(out), [3.0], atol=1e-9)


def test_center_distance_rectangle_com(device, xp):
    # label 1 ref: rows 0-1, col 0 (2 voxels) -> com row=(0+1)/2=0.5, col=0
    # label 1 pred: rows 0-1, col 4 (2 voxels) -> com row=0.5, col=4
    # distance = 4.0
    ref = np.zeros((4, 6), dtype=np.int32)
    pred = np.zeros((4, 6), dtype=np.int32)
    ref[0, 0] = 1
    ref[1, 0] = 1
    pred[0, 4] = 1
    pred[1, 4] = 1

    ref_x = xp.asarray(ref)
    pred_x = xp.asarray(pred)
    out = center_distance_batched(ref_x, pred_x, xp.asarray([1]), xp.asarray([1]), xp)
    np.testing.assert_allclose(_np(out), [4.0], atol=1e-9)


def test_center_distance_with_spacing(device, xp):
    # Same as simple test but col spacing=2.0 -> distance doubles
    ref = np.zeros((4, 4), dtype=np.int32)
    pred = np.zeros((4, 4), dtype=np.int32)
    ref[0, 0] = 1
    pred[0, 3] = 1

    ref_x = xp.asarray(ref)
    pred_x = xp.asarray(pred)
    out = center_distance_batched(
        ref_x, pred_x, xp.asarray([1]), xp.asarray([1]), xp, spacing=(1.0, 2.0)
    )
    np.testing.assert_allclose(_np(out), [6.0], atol=1e-9)


def test_center_distance_both_empty_is_nan(device, xp):
    ref = xp.asarray(np.zeros((3, 3), dtype=np.int32))
    pred = xp.asarray(np.zeros((3, 3), dtype=np.int32))
    out = _np(center_distance_batched(ref, pred, xp.asarray([9]), xp.asarray([9]), xp))
    assert np.isnan(out[0])


def test_center_distance_does_not_mutate_inputs(device, xp):
    ref = np.array([[0, 2], [2, 0]], dtype=np.int32)
    pred = np.array([[3, 0], [0, 3]], dtype=np.int32)
    ref_before = ref.copy()
    pred_before = pred.copy()

    ref_x = xp.asarray(ref)
    pred_x = xp.asarray(pred)
    center_distance_batched(ref_x, pred_x, xp.asarray([2]), xp.asarray([3]), xp)

    # Original host arrays must be untouched (v1 bug: center_of_mass mutated
    # non-zero entries to 1 in place).
    np.testing.assert_array_equal(ref, ref_before)
    np.testing.assert_array_equal(pred, pred_before)
    # And the arrays actually passed to the function must be unmutated too.
    np.testing.assert_array_equal(_np(ref_x), ref_before)
    np.testing.assert_array_equal(_np(pred_x), pred_before)
