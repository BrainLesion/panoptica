from __future__ import annotations

from panoptica.kernels.ccl import connected_components


def test_connected_components_2d_two_blobs(xp):
    arr = xp.zeros((5, 5), dtype=xp.uint8)
    arr[0, 0] = 1
    arr[4, 4] = 1
    labeled, n = connected_components(arr, xp)
    assert n == 2
    assert int(labeled[0, 0]) != 0
    assert int(labeled[4, 4]) != 0
    assert int(labeled[0, 0]) != int(labeled[4, 4])
    assert int(labeled[2, 2]) == 0


def test_connected_components_2d_single_blob(xp):
    arr = xp.zeros((5, 5), dtype=xp.uint8)
    arr[1:3, 1:3] = 1
    labeled, n = connected_components(arr, xp)
    assert n == 1
    assert bool(xp.all(labeled[1:3, 1:3] == 1))
    assert int(labeled[0, 0]) == 0


def test_connected_components_3d_two_separated_voxels(xp):
    arr = xp.zeros((6, 6, 6), dtype=xp.uint8)
    arr[0, 0, 0] = 1
    arr[5, 5, 5] = 1
    labeled, n = connected_components(arr, xp)
    assert n == 2
    assert int(labeled[0, 0, 0]) != int(labeled[5, 5, 5])


def test_connected_components_3d_single_blob(xp):
    arr = xp.zeros((5, 5, 5), dtype=xp.uint8)
    arr[1:3, 1:3, 1:3] = 1
    labeled, n = connected_components(arr, xp)
    assert n == 1
    assert bool(xp.all(labeled[1:3, 1:3, 1:3] == 1))


def test_connected_components_empty(xp):
    arr = xp.zeros((4, 4), dtype=xp.uint8)
    labeled, n = connected_components(arr, xp)
    assert n == 0
    assert bool(xp.all(labeled == 0))
