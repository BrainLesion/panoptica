"""Derived count-based metrics (fp/fn/prec/rec/rq).

Extracted from ``panoptica_result.py`` (#248) so the metric computation lives in the
computation package and ``PanopticaResult`` stays focused on holding and looking up
results. Each function takes the result object and reads already-available values
(true positives and instance counts).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from panoptica.core.result import PanopticaResult


def fp(res: PanopticaResult):
    return res.n_pred_instances - res.tp


def fn(res: PanopticaResult):
    return res.n_ref_instances - res.tp


def prec(res: PanopticaResult):
    return res.tp / (res.tp + res.fp)


def rec(res: PanopticaResult):
    return res.tp / (res.tp + res.fn)


def rq(res: PanopticaResult):
    """
    Calculate the Recognition Quality (RQ) based on TP, FP, and FN.

    Returns:
        float: Recognition Quality (RQ).
    """
    if res.tp == 0:
        return 0.0 if res.n_pred_instances + res.n_ref_instances > 0 else np.nan
    return res.tp / (res.tp + 0.5 * res.fp + 0.5 * res.fn)
