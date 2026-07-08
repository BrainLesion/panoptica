from __future__ import annotations

from panoptica.kernels.relabel import map_labels


def test_map_labels_basic(xp):
    arr = xp.array([0, 1, 2, 3, 1], dtype=xp.int32)
    mapping = {1: 5, 2: 6}
    out = map_labels(arr, mapping, xp)
    expected = xp.array([0, 5, 6, 3, 5], dtype=xp.int32)
    assert bool(xp.all(out == expected))


def test_map_labels_empty_mapping_returns_copy(xp):
    arr = xp.array([1, 2, 3], dtype=xp.int32)
    out = map_labels(arr, {}, xp)
    assert bool(xp.all(out == arr))
    out[0] = 99
    assert int(arr[0]) == 1  # original untouched -> pure/no aliasing


def test_map_labels_does_not_mutate_input(xp):
    arr = xp.array([1, 2], dtype=xp.int32)
    arr_copy = xp.array(arr, copy=True)
    map_labels(arr, {1: 9}, xp)
    assert bool(xp.all(arr == arr_copy))


def test_map_labels_2d(xp):
    arr = xp.array([[0, 1], [2, 0]], dtype=xp.int32)
    out = map_labels(arr, {1: 10, 2: 20}, xp)
    expected = xp.array([[0, 10], [20, 0]], dtype=xp.int32)
    assert bool(xp.all(out == expected))
