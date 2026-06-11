# Call 'python -m unittest' on this folder
# coverage run -m unittest
import os
import unittest

import numpy as np
from scipy.ndimage import label as cc_label

from panoptica.metrics import Metric
from panoptica._functionals import (
    _calc_overlapping_labels,
    _calc_matching_metric_of_overlapping_labels,
)


def _brute_force_pairs(prediction_arr, reference_arr, ref_labels, metric):
    """Reference implementation: evaluate the metric per overlapping pair."""
    overlapping_labels = _calc_overlapping_labels(
        prediction_arr=prediction_arr,
        reference_arr=reference_arr,
        ref_labels=ref_labels,
    )
    pairs = [
        (metric.value(reference_arr, prediction_arr, r, p), (r, p))
        for r, p in overlapping_labels
    ]
    return sorted(pairs, key=lambda x: x[0], reverse=not metric.decreasing)


def _make_overlapping_case(seed, shape):
    """Reference and prediction instance maps with independent label numbering."""
    rng = np.random.default_rng(seed)
    base = np.zeros(shape, dtype=np.int32)
    for _ in range(60):
        center = tuple(int(rng.integers(3, s - 3)) for s in shape)
        sl = tuple(
            slice(c - int(rng.integers(2, 5)), c + int(rng.integers(2, 5)))
            for c in center
        )
        base[sl] = 1
    reference_arr, _ = cc_label(base)

    perturbed = base.copy() > 0
    for ax in range(perturbed.ndim):
        perturbed = np.roll(perturbed, int(rng.integers(-2, 3)), axis=ax)
    perturbed = perturbed ^ (rng.random(perturbed.shape) < 0.02)
    prediction_arr, _ = cc_label(perturbed)
    return reference_arr.astype(np.int32), prediction_arr.astype(np.int32)


class Test_VectorizedMatching(unittest.TestCase):
    """The vectorized IoU/Dice matching must match the per-pair reference exactly."""

    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"
        return super().setUp()

    def test_iou_and_dice_match_bruteforce(self):
        for seed in range(6):
            for shape in [(80, 80), (40, 40, 24)]:
                reference_arr, prediction_arr = _make_overlapping_case(seed, shape)
                ref_labels = tuple(int(x) for x in np.unique(reference_arr) if x > 0)
                if not ref_labels:
                    continue
                for metric in (Metric.IOU, Metric.DSC):
                    fast = _calc_matching_metric_of_overlapping_labels(
                        prediction_arr, reference_arr, ref_labels, metric
                    )
                    brute = _brute_force_pairs(
                        prediction_arr, reference_arr, ref_labels, metric
                    )
                    self.assertEqual(len(fast), len(brute))
                    for (s_fast, pair_fast), (s_brute, pair_brute) in zip(fast, brute):
                        self.assertEqual(pair_fast, pair_brute)
                        # bit-identical scores (same integer counts, same division)
                        self.assertEqual(s_fast, s_brute)


if __name__ == "__main__":
    unittest.main()
