"""Unit tests for cldice_batched (centerline Dice), CPU-only via skimage.

Gracefully skips (not errors) if scikit-image is unavailable, per the parallel
build protocol (kernels/skimage may not be present in every environment).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("skimage")

from panoptica.metrics.topology import cldice_batched  # noqa: E402


def _np(arr):
    """Duck-typed device->host transfer for assertions (cupy has `.get()`)."""
    getter = getattr(arr, "get", None)
    return getter() if callable(getter) else np.asarray(arr)


def test_cldice_identical_lines(device, xp):
    # A straight horizontal line is its own skeleton; identical ref/pred ->
    # perfect overlap -> tprec = tsens = 1.0 -> clDice = 1.0.
    ref = np.zeros((5, 7), dtype=np.int32)
    pred = np.zeros((5, 7), dtype=np.int32)
    ref[2, 1:6] = 1
    pred[2, 1:6] = 1

    ref_x = xp.asarray(ref)
    pred_x = xp.asarray(pred)
    out = cldice_batched(ref_x, pred_x, xp.asarray([1]), xp.asarray([1]), xp)
    np.testing.assert_allclose(_np(out), [1.0], atol=1e-9)


def test_cldice_disjoint_lines_is_nan(device, xp):
    # Two parallel, non-overlapping horizontal lines: both skeletons are
    # non-empty but share no voxels, so tprec == tsens == 0 and clDice is
    # 2*0*0/(0+0) == 0/0. v1 (cldice._compute_centerline_dice_coefficient)
    # divides in numpy with no guard, yielding nan -- reproduced here.
    ref = np.zeros((5, 7), dtype=np.int32)
    pred = np.zeros((5, 7), dtype=np.int32)
    ref[1, 1:6] = 1
    pred[3, 1:6] = 1

    ref_x = xp.asarray(ref)
    pred_x = xp.asarray(pred)
    out = cldice_batched(ref_x, pred_x, xp.asarray([1]), xp.asarray([1]), xp)
    assert np.isnan(_np(out)[0])
