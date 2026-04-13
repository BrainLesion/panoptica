# Call 'python -m unittest' on this folder
# coverage run -m unittest
# coverage report
# coverage html
import os
import unittest

import numpy as np
from panoptica.panoptica_pipeline import (
    _panoptic_evaluate_region_wise,
    _panoptic_evaluate,
)
from panoptica.instance_matcher import NaiveThresholdMatching
from panoptica.instance_approximator import ConnectedComponentsInstanceApproximator
from panoptica.metrics import Metric
from panoptica.utils.processing_pair import UnmatchedInstancePair


def test_scenario_1_basic():
    """Test basic case with non-overlapping regions"""
    print("Test 1: Basic non-overlapping regions")

    gt = np.zeros((30, 30, 10), dtype=np.int32)
    pred = np.zeros((30, 30, 10), dtype=np.int32)

    # GT regions
    gt[5:15, 5:15, 2:8] = 1
    gt[20:25, 20:25, 2:8] = 2

    # Pred regions - slightly offset
    pred[6:16, 6:16, 3:9] = 1
    pred[19:24, 19:24, 3:9] = 2

    return gt, pred


def test_scenario_2_overlapping():
    """Test case with overlapping predictions"""
    print("Test 2: Overlapping predictions")

    gt = np.zeros((30, 30, 10), dtype=np.int32)
    pred = np.zeros((30, 30, 10), dtype=np.int32)

    # GT regions
    gt[5:15, 5:15, 2:8] = 1
    gt[20:25, 20:25, 2:8] = 2

    # Overlapping predictions
    pred[8:18, 8:18, 3:9] = 1  # Overlaps with both GT regions
    pred[21:26, 21:26, 3:9] = 2

    return gt, pred


def test_scenario_3_empty_prediction():
    """Test case with no predictions"""
    print("Test 3: Empty predictions")

    gt = np.zeros((30, 30, 10), dtype=np.int32)
    pred = np.zeros((30, 30, 10), dtype=np.int32)

    # Only GT regions, no predictions
    gt[5:15, 5:15, 2:8] = 1
    gt[20:25, 20:25, 2:8] = 2

    return gt, pred


def test_scenario_4_extra_predictions():
    """Test case with more predictions than GT regions"""
    print("Test 4: Extra predictions")

    gt = np.zeros((40, 40, 10), dtype=np.int32)
    pred = np.zeros((40, 40, 10), dtype=np.int32)

    # GT regions
    gt[5:15, 5:15, 2:8] = 1

    # Multiple predictions
    pred[6:16, 6:16, 3:9] = 1  # Close to GT
    pred[20:25, 20:25, 3:9] = 2  # Far from GT
    pred[30:35, 30:35, 3:9] = 3  # Even farther

    return gt, pred


class Test_RegionMatching_Comprehensive(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"
        return super().setUp()

    def run_test_scenario(self, gt, pred, scenario_name):
        """Run a test scenario and return results"""
        print(f"\n{scenario_name}")
        print(f"GT unique values: {np.unique(gt)}")
        print(f"Pred unique values: {np.unique(pred)}")

        try:
            # Create components
            matcher = NaiveThresholdMatching(
                matching_metric=Metric.DSC, matching_threshold=0.5
            )
            semantic_pair = UnmatchedInstancePair(prediction_arr=pred, reference_arr=gt)

            # Run evaluation
            result1 = _panoptic_evaluate(
                input_pair=semantic_pair,
                instance_matcher=matcher,
                instance_metrics=[Metric.DSC, Metric.IOU],
                global_metrics=[Metric.DSC],
                verbose=False,
            )
            result2 = _panoptic_evaluate_region_wise(
                input_pair=semantic_pair,
                instance_matcher=matcher,
                instance_metrics=[Metric.DSC, Metric.IOU],
                global_metrics=[Metric.DSC],
                verbose=False,
            )

            print(f"✅ {scenario_name} successful!")

            # Check individual metrics if available
            print("RESULT 1")
            print(result1)
            print("RESULT 2 (region-wise)")
            print(result2)

            semantic_pair2 = UnmatchedInstancePair(
                prediction_arr=pred, reference_arr=np.asarray(gt == 1, dtype=pred.dtype)
            )

            # Run evaluation
            result3 = _panoptic_evaluate(
                input_pair=semantic_pair2,
                instance_matcher=matcher,
                instance_metrics=[Metric.DSC, Metric.IOU],
                global_metrics=[Metric.DSC],
                verbose=False,
            )
            result4 = _panoptic_evaluate_region_wise(
                input_pair=semantic_pair2,
                instance_matcher=matcher,
                instance_metrics=[Metric.DSC, Metric.IOU],
                global_metrics=[Metric.DSC],
                verbose=False,
            )

            print("RESULT 3 (one gt ref)")
            print(result3)
            print("RESULT 4 (one gt ref, region-wise)")
            print(result4)

        except Exception as e:
            print(f"❌ {scenario_name} failed: {e}")
            import traceback

            traceback.print_exc()
            return False

        self.assertEqual(result1.n_ref_instances, result2.n_ref_instances)
        self.assertEqual(result3.n_ref_instances, result4.n_ref_instances)
        self.assertEqual(result1.n_pred_instances, result3.n_pred_instances)
        self.assertEqual(result1.n_pred_instances, result4.n_pred_instances)
        #
        self.assertEqual(result3.global_bin_dsc, result4.global_bin_dsc)

        if result1.global_bin_dsc != 0.0 and semantic_pair.n_ref_instances > 1:
            self.assertNotEqual(result1.global_bin_dsc, result2.global_bin_dsc)
            self.assertNotEqual(result2.global_bin_dsc, result3.global_bin_dsc)
            self.assertNotEqual(result1.global_bin_dsc, result3.global_bin_dsc)

        self.assertTrue(np.isnan(result2.tp))
        self.assertTrue(np.isnan(result4.tp))

        return True

    def test_scenario_1_basic(self):
        gt, pred = test_scenario_1_basic()
        self.assertTrue(
            self.run_test_scenario(gt, pred, "Test 1: Basic non-overlapping")
        )

    def test_scenario_2_overlapping(self):
        gt, pred = test_scenario_2_overlapping()
        self.assertTrue(
            self.run_test_scenario(gt, pred, "Test 2: Overlapping predictions")
        )

    def test_scenario_3_empty_prediction(self):
        gt, pred = test_scenario_3_empty_prediction()
        self.assertTrue(self.run_test_scenario(gt, pred, "Test 3: Empty predictions"))

    def test_scenario_4_extra_predictions(self):
        gt, pred = test_scenario_4_extra_predictions()
        self.assertTrue(self.run_test_scenario(gt, pred, "Test 4: Extra predictions"))
