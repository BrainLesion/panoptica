"""
Test for spatial structure caching optimization in instance_evaluator.

This test ensures that the spatial caching optimization produces identical results
whether spatial metrics are evaluated individually or together as a batch.
"""

import unittest
import numpy as np
from panoptica.instance_evaluator import _evaluate_instance
from panoptica.metrics import Metric


class Test_Spatial_Caching_Optimization(unittest.TestCase):

    def setUp(self):
        """Set up test data for spatial caching tests."""
        # Create flattened one-hot test data that will trigger spatial reshaping
        np.random.seed(42)  # For reproducible tests

        # Create a small test case that works with reshaping
        self.spatial_shape = (10, 8)  # 2D spatial dimensions
        self.num_ref_labels = 2

        # Create flattened arrays (total size = num_labels * spatial_size)
        total_size = (self.num_ref_labels + 1) * np.prod(
            self.spatial_shape
        )  # +1 for background

        # Reference array with instances 1 and 2
        self.reference_arr = np.zeros(total_size, dtype=np.uint8)
        self.reference_arr[20:60] = 1  # Instance 1
        self.reference_arr[100:120] = 2  # Instance 2

        # Prediction array with slight differences
        self.prediction_arr = self.reference_arr.copy()
        self.prediction_arr[50:70] = 1  # Slightly different instance 1

        # Test parameters
        self.voxelspacing = (1.0, 1.0)
        self.processing_pair_orig_shape = self.spatial_shape

        # Define spatial and non-spatial metrics for testing
        self.spatial_metrics = [Metric.ASSD, Metric.HD, Metric.CEDI]
        self.non_spatial_metrics = [Metric.DSC, Metric.IOU]
        self.mixed_metrics = [Metric.DSC, Metric.ASSD, Metric.HD]

    def test_spatial_cache_consistency(self):
        """Test that cached spatial evaluation gives same results as individual evaluation."""

        # Test with instance 1
        ref_idx = 1

        # Evaluate all spatial metrics together (uses caching)
        batched_result = _evaluate_instance(
            prediction_arr=self.prediction_arr,
            reference_arr=self.reference_arr,
            ref_idx=ref_idx,
            eval_metrics=self.spatial_metrics,
            voxelspacing=self.voxelspacing,
            processing_pair_orig_shape=self.processing_pair_orig_shape,
            num_ref_labels=self.num_ref_labels,
        )

        # Evaluate each spatial metric individually (no caching benefit)
        individual_results = {}
        for metric in self.spatial_metrics:
            result = _evaluate_instance(
                prediction_arr=self.prediction_arr,
                reference_arr=self.reference_arr,
                ref_idx=ref_idx,
                eval_metrics=[metric],  # Single metric
                voxelspacing=self.voxelspacing,
                processing_pair_orig_shape=self.processing_pair_orig_shape,
                num_ref_labels=self.num_ref_labels,
            )
            individual_results[metric] = result[metric]

        # Compare results - they should be identical
        for metric in self.spatial_metrics:
            with self.subTest(metric=metric.name):
                self.assertAlmostEqual(
                    batched_result[metric],
                    individual_results[metric],
                    places=10,
                    msg=f"Cached result differs from individual result for {metric.name}",
                )

    def test_mixed_metrics_cache_behavior(self):
        """Test that mixed spatial/non-spatial metrics work correctly with caching."""

        ref_idx = 1

        # Evaluate mixed metrics (should use caching for spatial ones)
        mixed_result = _evaluate_instance(
            prediction_arr=self.prediction_arr,
            reference_arr=self.reference_arr,
            ref_idx=ref_idx,
            eval_metrics=self.mixed_metrics,
            voxelspacing=self.voxelspacing,
            processing_pair_orig_shape=self.processing_pair_orig_shape,
            num_ref_labels=self.num_ref_labels,
        )

        # Verify all metrics computed successfully
        self.assertEqual(len(mixed_result), len(self.mixed_metrics))
        for metric in self.mixed_metrics:
            self.assertIn(metric, mixed_result)
            self.assertIsInstance(mixed_result[metric], (int, float))
            # Metrics should return reasonable values (not NaN or infinite)
            self.assertFalse(
                np.isnan(mixed_result[metric]), f"{metric.name} returned NaN"
            )
            self.assertTrue(
                np.isfinite(mixed_result[metric]),
                f"{metric.name} returned infinite value",
            )

    def test_no_spatial_metrics_no_cache(self):
        """Test that non-spatial metrics work correctly when no caching is needed."""

        ref_idx = 1

        # Evaluate only non-spatial metrics (no caching should occur)
        non_spatial_result = _evaluate_instance(
            prediction_arr=self.prediction_arr,
            reference_arr=self.reference_arr,
            ref_idx=ref_idx,
            eval_metrics=self.non_spatial_metrics,
            voxelspacing=self.voxelspacing,
            processing_pair_orig_shape=self.processing_pair_orig_shape,
            num_ref_labels=self.num_ref_labels,
        )

        # Verify results are computed and reasonable
        self.assertEqual(len(non_spatial_result), len(self.non_spatial_metrics))
        for metric in self.non_spatial_metrics:
            self.assertIn(metric, non_spatial_result)
            # DSC and IOU should be in [0, 1] range
            self.assertGreaterEqual(non_spatial_result[metric], 0.0)
            self.assertLessEqual(non_spatial_result[metric], 1.0)

    def test_empty_eval_metrics_list(self):
        """Test edge case with empty metrics list."""

        ref_idx = 1

        # Evaluate with empty metrics list
        empty_result = _evaluate_instance(
            prediction_arr=self.prediction_arr,
            reference_arr=self.reference_arr,
            ref_idx=ref_idx,
            eval_metrics=[],  # Empty list
            voxelspacing=self.voxelspacing,
            processing_pair_orig_shape=self.processing_pair_orig_shape,
            num_ref_labels=self.num_ref_labels,
        )

        # Should return empty dict
        self.assertEqual(len(empty_result), 0)
        self.assertIsInstance(empty_result, dict)


if __name__ == "__main__":
    unittest.main()
