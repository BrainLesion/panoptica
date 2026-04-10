# Call 'python -m unittest' on this folder
# coverage run -m unittest
# coverage report
# coverage html
import os
import unittest

import numpy as np
from panoptica.panoptica_pipeline import _panoptic_evaluate_region_wise, _panoptic_evaluate
from panoptica.instance_matcher import NaiveThresholdMatching
from panoptica.instance_approximator import ConnectedComponentsInstanceApproximator
from panoptica.metrics import Metric
from panoptica.utils.processing_pair import SemanticPair


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


def run_test_scenario(gt, pred, scenario_name):
    """Run a test scenario and return results"""
    print(f"\n{scenario_name}")
    print(f"GT unique values: {np.unique(gt)}")
    print(f"Pred unique values: {np.unique(pred)}")

    try:
        # Create components
        matcher = NaiveThresholdMatching(matching_metric=Metric.DSC, matching_threshold=0.5)
        approximator = ConnectedComponentsInstanceApproximator()
        semantic_pair = SemanticPair(prediction_arr=pred, reference_arr=gt)

        # Run evaluation
        result = _panoptic_evaluate_region_wise(
            input_pair=semantic_pair,
            instance_approximator=approximator,
            instance_matcher=matcher,
            instance_metrics=[Metric.DSC, Metric.IOU],
            global_metrics=[Metric.DSC],
            verbose=False,
        )

        result2 = _panoptic_evaluate(
            input_pair=semantic_pair,
            instance_approximator=approximator,
            instance_matcher=matcher,
            instance_metrics=[Metric.DSC, Metric.IOU],
            global_metrics=[Metric.DSC],
            verbose=False,
        )

        print(f"✅ {scenario_name} successful!")
        print(f"  Pred instances: {result.n_pred_instances}, Ref instances: {result.n_ref_instances}")
        print(f"  TP: {result.tp}, FP: {result.fp}, FN: {result.fn}")

        # Check individual metrics if available
        print(result)
        print()
        print(result2)

        return True

    except Exception as e:
        print(f"❌ {scenario_name} failed: {e}")
        import traceback

        traceback.print_exc()
        return False


class Test_RegionMatching_Comprehensive(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"
        return super().setUp()

    def test_scenario_1_basic(self):
        gt, pred = test_scenario_1_basic()
        self.assertTrue(run_test_scenario(gt, pred, "Test 1: Basic non-overlapping"))

    def test_scenario_2_overlapping(self):
        gt, pred = test_scenario_2_overlapping()
        self.assertTrue(run_test_scenario(gt, pred, "Test 2: Overlapping predictions"))

    def test_scenario_3_empty_prediction(self):
        gt, pred = test_scenario_3_empty_prediction()
        self.assertTrue(run_test_scenario(gt, pred, "Test 3: Empty predictions"))

    def test_scenario_4_extra_predictions(self):
        gt, pred = test_scenario_4_extra_predictions()
        self.assertTrue(run_test_scenario(gt, pred, "Test 4: Extra predictions"))
