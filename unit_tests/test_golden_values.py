"""Golden numerical characterization tests (see refactor plan #181 + #248).

These tests freeze the *computed metric values* of the pipeline on fixed inputs.
Across the metric-system overhaul the result-key strings, the evaluator's metric
API and the on-disk schema all change, but the underlying mathematics must not.

When a later phase deliberately changes a *name* (e.g. the unified ``metrics=``
constructor in PR2, or the grouped result keys in PR6), update the construction
helpers / expected-key strings in this file ONLY — the numeric values below must
stay byte-for-byte identical. A diff in a number here means a real regression.
"""

import math
import os
import unittest

import numpy as np

from panoptica import InputType
from panoptica.instance_approximator import ConnectedComponentsInstanceApproximator
from panoptica.instance_matcher import NaiveThresholdMatching
from panoptica.metrics import Metric
from panoptica.panoptica_evaluator import Panoptica_Evaluator

ALL_METRICS = [
    Metric.DSC,
    Metric.IOU,
    Metric.ASSD,
    Metric.clDSC,
    Metric.RVD,
    Metric.RVAE,
    Metric.CEDI,
    Metric.HD,
    Metric.HD95,
    Metric.NSD,
]


# --------------------------------------------------------------------------- #
# Fixtures (kept inline so the golden inputs travel with the expected values). #
# --------------------------------------------------------------------------- #
def _instance_case() -> tuple[np.ndarray, np.ndarray]:
    """Two matched instances (one perfect, one ~0.9 IoU), one FP and one FN."""
    ref = np.zeros((50, 50), dtype=np.uint16)
    pred = np.zeros((50, 50), dtype=np.uint16)
    ref[5:15, 5:15] = 1
    pred[5:15, 5:15] = 1
    ref[20:30, 20:30] = 2
    pred[21:30, 20:30] = 2
    ref[40:48, 5:13] = 3  # FN (no prediction)
    pred[40:48, 40:48] = 4  # FP (no reference)
    return pred, ref


def _autc_case() -> tuple[np.ndarray, np.ndarray]:
    """Single instance pair with IoU == 1/3 (the canonical AUTC drop-off case)."""
    ref = np.zeros((50, 50), dtype=np.uint16)
    pred = np.zeros((50, 50), dtype=np.uint16)
    ref[10:20, 10:30] = 1
    pred[10:20, 20:40] = 1
    return pred, ref


def _region_case() -> tuple[np.ndarray, np.ndarray]:
    gt = np.zeros((30, 30, 10), dtype=np.int32)
    pred = np.zeros((30, 30, 10), dtype=np.int32)
    gt[5:15, 5:15, 2:8] = 1
    gt[20:25, 20:25, 2:8] = 2
    pred[6:16, 6:16, 3:9] = 1
    pred[19:24, 19:24, 3:9] = 2
    return pred, gt


# --------------------------------------------------------------------------- #
# Construction helpers — the *only* things that should change on an API rename. #
# --------------------------------------------------------------------------- #
def _build_instance_evaluator() -> Panoptica_Evaluator:
    # Bare metrics expand to both instance and global modes, reproducing the old
    # instance_metrics=ALL, global_metrics=ALL configuration.
    return Panoptica_Evaluator(
        expected_input=InputType.SEMANTIC,
        instance_approximator=ConnectedComponentsInstanceApproximator(),
        instance_matcher=NaiveThresholdMatching(),
        metrics=list(ALL_METRICS),
    )


def _build_autc_evaluator() -> Panoptica_Evaluator:
    return Panoptica_Evaluator(
        expected_input=InputType.SEMANTIC,
        instance_approximator=ConnectedComponentsInstanceApproximator(),
        instance_matcher=NaiveThresholdMatching(),
    )


def _build_region_evaluator() -> Panoptica_Evaluator:
    # Bare metrics expand to both modes; region-wise output only depends on the
    # global set [DSC, IOU, ASSD].
    return Panoptica_Evaluator(
        expected_input=InputType.SEMANTIC,
        instance_approximator=ConnectedComponentsInstanceApproximator(),
        instance_matcher=NaiveThresholdMatching(),
        metrics=[Metric.DSC, Metric.IOU, Metric.ASSD],
        per_region_evaluation=True,
    )


# --------------------------------------------------------------------------- #
# Frozen expected values (harvested from the pre-refactor pipeline).            #
# --------------------------------------------------------------------------- #
GOLDEN_INSTANCE: dict[str, float] = {
    "n_ref_instances": 3,
    "n_pred_instances": 3,
    "tp": 2,
    "fp": 1,
    "fn": 1,
    "prec": 0.6666666666666666,
    "rec": 0.6666666666666666,
    "rq": 0.6666666666666666,
    "sq": 0.95,
    "sq_std": 0.04999999999999999,
    "pq": 0.6333333333333333,
    "sq_dsc": 0.9736842105263157,
    "sq_dsc_std": 0.026315789473684237,
    "pq_dsc": 0.6491228070175438,
    "sq_cldsc": 1.0,
    "sq_cldsc_std": 0.0,
    "pq_cldsc": 0.6666666666666666,
    "sq_assd": 0.12826797385620914,
    "sq_assd_std": 0.12826797385620914,
    "sq_rvd": -0.05,
    "sq_rvd_std": 0.05,
    "sq_rvae": 0.05,
    "sq_rvae_std": 0.05,
    "sq_cedi": 0.25,
    "sq_cedi_std": 0.25,
    "sq_hd": 0.5,
    "sq_hd_std": 0.5,
    "sq_hd95": 0.5,
    "sq_hd95_std": 0.5,
    "sq_nsd": 1.0,
    "sq_nsd_std": 0.0,
    "instance_voxel_count_ref": 100.0,
    "instance_volume_ref": 100.0,
    "global_bin_volume_pred": 254,
    "global_bin_volume_ref": 264,
    "global_bin_dsc": 0.7335907335907336,
    "global_bin_iou": 0.5792682926829268,
    "global_bin_cldsc": 0.6829268292682926,
    "global_bin_assd": 5.670966441085757,
    "global_bin_rvd": -0.03787878787878788,
    "global_bin_rvae": 0.03787878787878788,
    "global_bin_cedi": 8.44357212364506,
    "global_bin_hd": 25.45584412271571,
    "global_bin_hd95": 22.80350850198276,
    "global_bin_nsd": 0.626326530612245,
}

GOLDEN_AUTC: dict[str, float] = {
    "sq": 0.11666666666666667,
    "pq": 0.11666666666666667,
    "rq": 0.35,
    "sq_dsc": 0.175,
    "pq_dsc": 0.175,
    "sq_cldsc": 0.0,
    "pq_cldsc": 0.0,
}

GOLDEN_REGION_TO_DICT: dict[str, float] = {
    "n_ref_instances": float("nan"),
    "n_pred_instances": float("nan"),
    "tp": float("nan"),
    "global_bin_volume_pred": 750,
    "global_bin_volume_ref": 750,
    "global_bin_dsc": 0.6466666666666666,
    "global_bin_iou": 0.47783251231527096,
    "global_bin_assd": 0.9047506328833388,
    "instance_voxel_count_ref": float("nan"),
    "instance_volume_ref": float("nan"),
}

GOLDEN_REGION_AVG: dict[str, float] = {
    "region_avg_dsc": 0.6041666666666667,
    "region_avg_iou": 0.4365351629502573,
    "region_avg_assd": 0.895880639041989,
}


def _assert_close(test: unittest.TestCase, actual, expected, key: str) -> None:
    actual = float(actual)
    expected = float(expected)
    if math.isnan(expected):
        test.assertTrue(math.isnan(actual), f"{key}: expected nan, got {actual}")
    else:
        test.assertTrue(
            math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12),
            f"{key}: expected {expected}, got {actual}",
        )


class Test_Golden_Values(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"
        return super().setUp()

    def test_instance_values(self):
        evaluator = _build_instance_evaluator()
        result = evaluator.evaluate(*_instance_case(), voxelspacing=(1.0, 1.0))[
            "ungrouped"
        ]
        actual = result.to_dict()
        self.assertEqual(
            set(actual.keys()),
            set(GOLDEN_INSTANCE.keys()),
            "instance result keys drifted from the frozen schema",
        )
        for key, expected in GOLDEN_INSTANCE.items():
            _assert_close(self, actual[key], expected, key)

    def test_autc_values(self):
        evaluator = _build_autc_evaluator()
        autc = evaluator.evaluate_autc(*_autc_case(), threshold_step_size=0.1)[
            "ungrouped"
        ]
        self.assertEqual(
            autc.thresholds,
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        )
        self.assertEqual(autc.threshold_results[0.1].tp, 1)
        self.assertEqual(autc.threshold_results[0.4].tp, 0)
        for key, expected in GOLDEN_AUTC.items():
            _assert_close(self, autc.get_autc(key), expected, f"autc_{key}")

    def test_region_wise_values(self):
        evaluator = _build_region_evaluator()
        result = evaluator.evaluate(*_region_case())["ungrouped"]
        actual = result.to_dict()
        self.assertEqual(
            set(actual.keys()),
            set(GOLDEN_REGION_TO_DICT.keys()),
            "region-wise result keys drifted from the frozen schema",
        )
        for key, expected in GOLDEN_REGION_TO_DICT.items():
            _assert_close(self, actual[key], expected, key)
        for key, expected in GOLDEN_REGION_AVG.items():
            _assert_close(self, getattr(result, key), expected, key)


if __name__ == "__main__":
    unittest.main()
