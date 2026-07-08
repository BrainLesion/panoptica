from __future__ import annotations

import pytest

from panoptica.core.errors import InputValidationError
from panoptica.kernels.crop import bounding_box


def test_bounding_box_2d(xp):
    arr = xp.zeros((5, 5), dtype=xp.int32)
    arr[1:3, 2:4] = 1
    box = bounding_box(arr, xp)
    assert box == (slice(1, 3), slice(2, 4))


def test_bounding_box_3d_single_voxel(xp):
    arr = xp.zeros((4, 4, 4), dtype=xp.int32)
    arr[2, 3, 1] = 1
    box = bounding_box(arr, xp)
    assert box == (slice(2, 3), slice(3, 4), slice(1, 2))


def test_bounding_box_full_extent(xp):
    arr = xp.zeros((3, 3), dtype=xp.int32)
    arr[0, 0] = 1
    arr[2, 2] = 1
    box = bounding_box(arr, xp)
    assert box == (slice(0, 3), slice(0, 3))


def test_bounding_box_empty_raises(xp):
    arr = xp.zeros((3, 3), dtype=xp.int32)
    with pytest.raises(InputValidationError):
        bounding_box(arr, xp)
