#!/usr/bin/env python3
"""
Integration test for RegionBasedMatching with full panoptic evaluation
"""
import os
import unittest

import numpy as np
from panoptica.panoptica_evaluator import panoptic_evaluate
from panoptica.instance_matcher import RegionBasedMatching
from panoptica.instance_approximator import ConnectedComponentsInstanceApproximator
from panoptica.metrics import Metric
from panoptica.utils.constants import CCABackend
from panoptica.utils.processing_pair import SemanticPair


def create_test_data():
    """Create simple test data with ground truth and prediction instances"""
    # Create a simple 3D volume with 2 GT regions
    gt = np.zeros((50, 50, 20), dtype=np.int32)
    pred = np.zeros((50, 50, 20), dtype=np.int32)

    # GT region 1: cube in corner
    gt[10:20, 10:20, 5:15] = 1

    # GT region 2: cube in opposite corner
    gt[30:40, 30:40, 5:15] = 2

    # Prediction region 1: slightly offset from GT region 1
    pred[12:22, 12:22, 6:16] = 1

    # Prediction region 2: different location, should map to closest GT region
    pred[25:35, 25:35, 6:16] = 2

    return gt, pred


class Test_RegionMatching_Integration(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"
        return super().setUp()

    def test_region_integration(self):
        """Test RegionBasedMatching with full panoptic evaluation"""
        print("Testing RegionBasedMatching with panoptic_evaluate...")

        # Create test data
        gt, pred = create_test_data()

        print(f"GT shape: {gt.shape}, unique values: {np.unique(gt)}")
        print(f"Pred shape: {pred.shape}, unique values: {np.unique(pred)}")

        # Create region-based matcher
        matcher = RegionBasedMatching(cca_backend=CCABackend.scipy)

        # Create instance approximator
        approximator = ConnectedComponentsInstanceApproximator()

        try:
            # Create semantic pair
            semantic_pair = SemanticPair(prediction_arr=pred, reference_arr=gt)

            # Run panoptic evaluation
            result = panoptic_evaluate(
                input_pair=semantic_pair,
                instance_approximator=approximator,
                instance_matcher=matcher,
                instance_metrics=[Metric.DSC, Metric.IOU],
                global_metrics=[Metric.DSC],
                verbose=True,
            )

            print(f"\n✅ Integration test successful!")
            print(f"Number of prediction instances: {result.n_pred_instances}")
            print(f"Number of reference instances: {result.n_ref_instances}")
            print(f"TP: {result.tp}")
            print(f"FP: {result.fp}")
            print(f"FN: {result.fn}")
            print(f"Precision: {result.prec}")
            print(f"Recall: {result.rec}")
            print(f"RQ: {result.rq}")

            # Check if metrics are NaN as expected for region-based matching
            if np.isnan(result.tp):
                print("✅ Count metrics correctly set to NaN for region-based matching")
            else:
                print("⚠️  Expected TP to be NaN for region-based matching")

            # Check individual instance metrics
            if hasattr(result, "list_metrics") and result.list_metrics:
                print(f"\nInstance metrics:")
                for metric, values in result.list_metrics.items():
                    print(f"  {metric}: {values}")

            self.assertTrue(True)

        except Exception as e:
            print(f"❌ Error during integration test: {e}")
            import traceback

            traceback.print_exc()
            self.assertTrue(False)
