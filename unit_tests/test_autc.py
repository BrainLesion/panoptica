# Call 'python -m unittest' on this folder
# coverage run -m unittest
# coverage report
# coverage html
import os
import unittest
import numpy as np
from panoptica import InputType
from panoptica.instance_approximator import ConnectedComponentsInstanceApproximator
from panoptica.instance_matcher import NaiveThresholdMatching
from panoptica.panoptica_evaluator import Panoptica_Evaluator


class Test_AUTC(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"
        return super().setUp()

    def test_perfect_match_autc(self):
        """
        Test AUTC computation on a perfect prediction.
        If IoU is 1.0 everywhere, the metric curve is flat at y=1.0.
        Integrating y=1.0 over the threshold range [0.1, 1.0] including padding of y(0)=1 gives an area of 1.
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

        self.assertAlmostEqual(result.autc_pq, 1.0)
        self.assertAlmostEqual(result.autc_sq, 1.0)
        self.assertAlmostEqual(result.autc_rq, 1.0)

        # TP count should be exactly 1 at all thresholds
        for t in result.thresholds:
            self.assertEqual(result.threshold_results[t].tp, 1)

    def test_partial_overlap_autc(self):
        """
        Test the drop-off behavior. Pred and Ref overlap with IoU = 1/3 (~0.333).
        """
        ref = np.zeros([50, 50], dtype=np.uint16)
        pred = np.zeros([50, 50], dtype=np.uint16)

        ref[10:20, 10:30] = 1
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
        self.assertEqual(result.threshold_results[1.0].tp, 0)

        # The AUTC should be computable, greater than 0, but less than the perfect 1.0
        self.assertGreater(result.autc_pq, 0.0)
        self.assertLess(result.autc_pq, 1.0)

    def test_partial_overlap_autc_boundary(self):
        """
        Test matching exactly at the IoU boundary using a fine step size.
        IoU = 1/3 exactly. With step_size=0.05, we get thresholds 0.30, 0.35, ...
        - threshold 0.30 < 0.333 → should match (TP=1)
        - threshold 0.35 > 0.333 → should NOT match (TP=0)
        This pins the boundary behaviour precisely.
        """
        ref = np.zeros([50, 50], dtype=np.uint16)
        pred = np.zeros([50, 50], dtype=np.uint16)
        ref[10:20, 10:30] = 1
        pred[10:20, 20:40] = 1

        evaluator = Panoptica_Evaluator(
            expected_input=InputType.SEMANTIC,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=NaiveThresholdMatching(),
        )

        result = evaluator.evaluate_autc(pred, ref, threshold_step_size=0.05)[
            "ungrouped"
        ]

        self.assertEqual(result.threshold_results[0.30].tp, 1)  # 0.30 < 1/3
        self.assertEqual(result.threshold_results[0.35].tp, 0)  # 0.35 > 1/3

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

    def test_autc_metric_keys_match_to_dict(self):
        """get_autc_metric_keys and to_dict must produce identical key sets."""
        evaluator = Panoptica_Evaluator(
            expected_input=InputType.SEMANTIC,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=NaiveThresholdMatching(),
        )
        ref = np.zeros([50, 50], dtype=np.uint16)
        ref[10:40, 10:40] = 1
        pred = ref.copy()

        step = 0.1
        header_keys = set(evaluator.get_autc_metric_keys(step))
        result_keys = set(
            evaluator.evaluate_autc(pred, ref, threshold_step_size=step)["ungrouped"]
            .to_dict()
            .keys()
        )

        self.assertEqual(header_keys, result_keys)
