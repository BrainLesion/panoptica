# Call 'python -m unittest' on this folder
# coverage run -m unittest
# coverage report
# coverage html
import os
import unittest

import numpy as np

from panoptica import InputType
from panoptica.instance_approximator import ConnectedComponentsInstanceApproximator
from panoptica.instance_matcher import MaximizeMergeMatching, NaiveThresholdMatching
from panoptica.metrics import Metric
from panoptica.panoptica_evaluator import Panoptica_Evaluator
from panoptica.panoptica_result import MetricCouldNotBeComputedException
from panoptica.utils.processing_pair import SemanticPair
from panoptica.utils.segmentation_class import SegmentationClassGroups
import sys
from pathlib import Path

from unittest import mock
from panoptica.utils.input_check_and_conversion.sanity_checker import (
    sanity_check_and_convert_to_array,
    INPUTDTYPE,
    _InputDataTypeChecker,
    print_available_package_to_input_handlers,
)


class Test_AUTC(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"
        return super().setUp()

    def test_perfect_match_autc(self):
        """
        Test AUTC computation on a perfect prediction.
        If IoU is 1.0 everywhere, the metric curve is flat at y=1.0.
        Integrating y=1.0 over the threshold range [0.1, 1.0] gives an area of 0.9.
        """
        ref = np.zeros([50, 50], dtype=np.uint16)
        ref[10:40, 10:40] = 1  # 30x30 square
        pred = ref.copy()  # Perfect match

        evaluator = Panoptica_Evaluator(
            expected_input=InputType.SEMANTIC,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=NaiveThresholdMatching(),
        )

        result = evaluator.evaluate_autc(pred, ref, threshold_step_size=0.1)[
            "ungrouped"
        ]

        # Check that trapezoidal integration of a flat 1.0 curve over 0.9 width = 0.9
        self.assertAlmostEqual(result.autc_pq, 0.9)
        self.assertAlmostEqual(result.autc_sq, 0.9)
        self.assertAlmostEqual(result.autc_rq, 0.9)

        # TP count should be exactly 1 at all thresholds
        for t in result.thresholds:
            self.assertEqual(result.threshold_results[t].tp, 1)

    def test_partial_overlap_autc(self):
        """
        Test the drop-off behavior. Pred and Ref overlap with exactly 0.5 IoU.
        The matching algorithm should find 1 TP for thresholds <= 0.5,
        and 0 TP for thresholds > 0.5.
        """
        ref = np.zeros([50, 50], dtype=np.uint16)
        pred = np.zeros([50, 50], dtype=np.uint16)

        # Reference: 10x20 = 200 pixels
        ref[10:20, 10:30] = 1
        # Prediction: 10x20 = 200 pixels, offset to the right by 10 pixels
        # Intersection: 10x10 = 100 pixels. Union = 300 pixels. IoU = 1/3 (0.333...)
        pred[10:20, 20:40] = 1

        evaluator = Panoptica_Evaluator(
            expected_input=InputType.SEMANTIC,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=NaiveThresholdMatching(),
        )

        result = evaluator.evaluate_autc(pred, ref, threshold_step_size=0.1)[
            "ungrouped"
        ]

        # Since IoU is ~0.333, threshold results at 0.1, 0.2, 0.3 should match
        self.assertEqual(result.threshold_results[0.1].tp, 1)
        self.assertEqual(result.threshold_results[0.3].tp, 1)

        # Thresholds from 0.4 onwards should fail to match due to threshold > IoU
        self.assertEqual(result.threshold_results[0.4].tp, 0)
        self.assertEqual(result.threshold_results[0.9].tp, 0)

        # The AUTC should be computable, greater than 0, but less than the perfect 0.9
        self.assertGreater(result.autc_pq, 0.0)
        self.assertLess(result.autc_pq, 0.9)

    def test_empty_arrays_autc(self):
        """
        Test that AUTC computation does not crash when there are no instances present
        in both the prediction and the reference. Verifies edge case safety.
        """
        ref = np.zeros([50, 50], dtype=np.uint16)
        pred = np.zeros([50, 50], dtype=np.uint16)

        evaluator = Panoptica_Evaluator(
            expected_input=InputType.SEMANTIC,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=NaiveThresholdMatching(),
        )

        # This should execute gracefully without raising an exception
        result = evaluator.evaluate_autc(pred, ref, threshold_step_size=0.1)[
            "ungrouped"
        ]

        self.assertIsNotNone(result)
        self.assertEqual(len(result.thresholds), 10)  # 0.1 through 1.0 inclusive

        # Verify that to_dict() safely extracts values without throwing errors
        # on uncomputable metrics
        res_dict = result.to_dict()
        self.assertIsInstance(res_dict, dict)
        self.assertIn("autc_pq", res_dict)
