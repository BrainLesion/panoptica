from __future__ import annotations

from panoptica.kernels.voronoi import voronoi_regions


def test_voronoi_regions_1d_hand_computed(xp):
    # labels at index 0 (=1) and index 5 (=2); nearest-region split at the midpoint.
    arr = xp.array([[1, 0, 0, 0, 0, 2]], dtype=xp.int32)
    region_map, n = voronoi_regions(arr, xp)
    assert n == 2
    expected = xp.array([[1, 1, 1, 2, 2, 2]], dtype=xp.int32)
    assert bool(xp.all(region_map == expected))


def test_voronoi_regions_empty_mask(xp):
    arr = xp.zeros((4, 4), dtype=xp.int32)
    region_map, n = voronoi_regions(arr, xp)
    assert n == 0
    assert bool(xp.all(region_map == 0))


def test_voronoi_regions_single_label_fills_all(xp):
    arr = xp.zeros((3, 3), dtype=xp.int32)
    arr[1, 1] = 1
    region_map, n = voronoi_regions(arr, xp)
    assert n == 1
    assert bool(xp.all(region_map == 1))


def test_voronoi_regions_dtype(xp):
    arr = xp.zeros((3, 3), dtype=xp.int32)
    arr[0, 0] = 1
    region_map, _ = voronoi_regions(arr, xp)
    assert region_map.dtype == xp.int32
