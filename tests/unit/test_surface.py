from __future__ import annotations

from panoptica.kernels.surface import surface_border


def test_surface_border_3x3_full_block(xp):
    mask = xp.ones((3, 3), dtype=bool)
    border = surface_border(mask, xp)
    # erosion (connectivity=1) of a full 3x3 block leaves only the center pixel;
    # border = mask XOR eroded -> everything except the center.
    expected = xp.ones((3, 3), dtype=bool)
    expected[1, 1] = False
    assert bool(xp.all(border == expected))


def test_surface_border_single_voxel(xp):
    mask = xp.zeros((3, 3), dtype=bool)
    mask[1, 1] = True
    border = surface_border(mask, xp)
    # a single voxel is entirely surface
    assert bool(xp.all(border == mask))


def test_surface_border_empty(xp):
    mask = xp.zeros((4, 4), dtype=bool)
    border = surface_border(mask, xp)
    assert not bool(xp.any(border))


def test_surface_border_dtype_is_bool(xp):
    mask = xp.ones((3, 3), dtype=bool)
    border = surface_border(mask, xp)
    assert border.dtype == xp.bool_
