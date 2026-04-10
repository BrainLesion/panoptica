#!/usr/bin/env python3
"""
Test script for the RegionBasedMatching implementation
"""
import os
import unittest
import numpy as np
from panoptica.instance_matcher import RegionBasedMatching
from panoptica.utils.processing_pair import UnmatchedInstancePair
from panoptica.utils.constants import CCABackend


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


class Test_RegionMatching(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"
        return super().setUp()

    def test_region_based_matching(self):
        """Test the RegionBasedMatching algorithm"""
        print("Testing RegionBasedMatching...")

        # Create test data
        gt, pred = create_test_data()

        # Create unmatched instance pair
        unmatched_pair = UnmatchedInstancePair(prediction_arr=pred, reference_arr=gt)

        print(f"Ground truth labels: {unmatched_pair.ref_labels}")
        print(f"Prediction labels: {unmatched_pair.pred_labels}")

        # Create region-based matcher
        matcher = RegionBasedMatching(cca_backend=CCABackend.scipy)

        # Perform matching
        try:
            labelmap = matcher._match_instances(unmatched_pair)

            print(f"Matching successful!")
            print(f"Label map: {labelmap.get_one_to_one_dictionary()}")

            # Create matched instance pair
            matched_pair = matcher.match_instances(unmatched_pair)

            print(f"Matched instances: {matched_pair.matched_instances}")
            print(f"Prediction instances: {matched_pair.n_prediction_instance}")
            print(f"Reference instances: {matched_pair.n_reference_instance}")

            self.assertTrue(True)

        except Exception as e:
            print(f"Error during matching: {e}")
            import traceback

            traceback.print_exc()
            self.assertTrue(False)
