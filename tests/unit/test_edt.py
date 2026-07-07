from __future__ import annotations

import math

from panoptica.kernels.edt import edt


def _mask_with_center_hole(xp):
    # 1 1 1
    # 1 0 1
    # 1 1 1
    mask = xp.ones((3, 3), dtype=bool)
    mask[1, 1] = False
    return mask


def test_edt_hand_computed_unit_spacing(xp):
    mask = _mask_with_center_hole(xp)
    field = edt(mask, xp)
    assert math.isclose(float(field[1, 1]), 0.0, abs_tol=1e-6)
    assert math.isclose(float(field[0, 1]), 1.0, abs_tol=1e-6)
    assert math.isclose(float(field[0, 0]), math.sqrt(2), abs_tol=1e-6)


def test_edt_dtype_matches_backend_precision(xp):
    mask = _mask_with_center_hole(xp)
    field = edt(mask, xp)
    if xp.__name__ == "cupy":
        assert field.dtype == xp.float32
    else:
        assert field.dtype == xp.float64


def test_edt_with_spacing(xp):
    mask = _mask_with_center_hole(xp)
    field = edt(mask, xp, spacing=(2.0, 1.0))
    # distance from (0,1) to (1,1) along axis0 with spacing 2.0 -> 2.0
    assert math.isclose(float(field[0, 1]), 2.0, abs_tol=1e-5)
    # distance from (1,0) to (1,1) along axis1 with spacing 1.0 -> 1.0
    assert math.isclose(float(field[1, 0]), 1.0, abs_tol=1e-5)
