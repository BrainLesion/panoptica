"""Metric computation extracted from the result object (#248).

``PanopticaResult`` holds and lazily looks up results; the actual metric math lives
here so the result object stays a (mostly) passive container.
"""

from panoptica.computation.derived_metrics import fn, fp, prec, rec, rq
from panoptica.computation.global_binary import calc_global_bin_metric

__all__ = ["fp", "fn", "prec", "rec", "rq", "calc_global_bin_metric"]
