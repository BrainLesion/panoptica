# Call 'python -m unittest' on this folder
# coverage run -m unittest
# coverage report
# coverage html
import os
import unittest
import numpy as np

from panoptica import InputType
from panoptica.panoptica_evaluator import Panoptica_Evaluator
from panoptica.instance_approximator import ConnectedComponentsInstanceApproximator
from panoptica.instance_matcher import NaiveThresholdMatching, MaxBipartiteMatching
from panoptica.metrics import Metric
from panoptica.utils.segmentation_class import SegmentationClassGroups
from panoptica.utils.label_group import LabelMergeGroup, LabelPartGroup


class Test_Part_Metrics_Global_MultiChannel(unittest.TestCase):
    """Test multi-channel global metrics for part-aware evaluation."""

    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"
        return super().setUp()

    def create_basic_masks(self, size=(28, 28)):
        """Create basic prediction and reference masks for testing."""
        pred_masks = np.zeros(size, dtype=np.int32)
        ref_masks = np.zeros(size, dtype=np.int32)
        return pred_masks, ref_masks

    def get_part_groups(self):
        """Get standard part groups configuration."""
        return {
            "class_1": (1, False),
            "class_2": (2, False),
            "class_3": (3, False),
            "class_4": (4, True),
            "Part": LabelPartGroup([1], [2], False),
        }

    def get_multipart_groups(self):
        """Get multi-part groups configuration."""
        return {
            "class_1": (1, False),
            "class_2": (2, False),
            "extra": (5, False),
            "class_3": (3, False),
            "class_4": (4, True),
            "Part": LabelPartGroup([1], [2, 5], False),
        }

    def test_perfect_overlap_multi_channel_metrics(self):
        """Test perfect overlap case for multi-channel global metrics."""
        pred_masks, ref_masks = self.create_basic_masks()

        # Create perfect overlap scenario
        # Stuff Classes first (lowest hierarchy)
        pred_masks[1:30, 1:30] = 4
        ref_masks[1:30, 1:30] = 4

        # Thing Classes next (medium hierarchy)
        pred_masks[7:20, 7:20] = 1
        ref_masks[7:20, 7:20] = 1
        pred_masks[22:27, 22:27] = 3
        ref_masks[22:27, 22:27] = 3

        # Part Classes last (highest hierarchy)
        pred_masks[14:19, 14:19] = 2
        ref_masks[14:19, 14:19] = 2

        class_groups = SegmentationClassGroups(self.get_part_groups())
        evaluator = Panoptica_Evaluator(
            expected_input=InputType.SEMANTIC,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=MaxBipartiteMatching(),
            segmentation_class_groups=class_groups,
            global_metrics=[Metric.DSC, Metric.IOU, Metric.ASSD],
        )

        results = evaluator.evaluate(pred_masks, ref_masks, verbose=False)
        part_result = results["part"]

        # Test that multi-channel metrics are calculated
        self.assertEqual(part_result.tp, 1)
        self.assertEqual(part_result.fp, 0)
        self.assertEqual(part_result.fn, 0)
        self.assertEqual(part_result.sq, 1.0)
        self.assertEqual(part_result.rq, 1.0)
        self.assertEqual(part_result.pq, 1.0)
        self.assertEqual(part_result.sq_dsc, 1.0)
        self.assertEqual(part_result.global_bin_dsc, 1.0)
        self.assertEqual(part_result.global_bin_iou, 1.0)

        # Test channel metrics retrieval
        channel_dsc = part_result.get_channel_metrics("dsc")
        self.assertIsNotNone(channel_dsc)
        self.assertIsInstance(channel_dsc, dict)

        # Test that we have metrics for the appropriate channels
        # Channel 0: combined thing+part, Channel 2: just parts
        expected_channels = [0, 2]  # Skip channel 1 (thing only)
        for channel in expected_channels:
            if channel in channel_dsc:
                self.assertAlmostEqual(channel_dsc[channel], 1.0, places=4)

    def test_partial_overlap_multi_channel_metrics(self):
        """Test partial overlap case for multi-channel global metrics."""
        pred_masks, ref_masks = self.create_basic_masks()

        # Create partial overlap scenario
        # Stuff Classes first
        pred_masks[1:30, 1:30] = 4
        ref_masks[1:30, 1:30] = 4

        # Thing Classes next
        pred_masks[7:20, 7:20] = 1
        ref_masks[7:20, 7:20] = 1
        pred_masks[22:27, 22:27] = 3
        ref_masks[22:27, 22:27] = 3

        # Part Classes last - different sizes
        pred_masks[14:19, 14:19] = 2  # Larger part in prediction
        ref_masks[14:15, 14:19] = 2  # Smaller part in reference

        class_groups = SegmentationClassGroups(self.get_part_groups())
        evaluator = Panoptica_Evaluator(
            expected_input=InputType.SEMANTIC,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=MaxBipartiteMatching(),
            segmentation_class_groups=class_groups,
            global_metrics=[Metric.DSC, Metric.IOU],
        )

        results = evaluator.evaluate(pred_masks, ref_masks, verbose=False)
        part_result = results["part"]

        # Test basic metrics
        self.assertEqual(part_result.tp, 1)
        self.assertEqual(part_result.fp, 0)
        self.assertEqual(part_result.fn, 0)

        # Test that global metrics are averages across channels
        self.assertGreater(part_result.global_bin_dsc, 0.6)
        self.assertLess(part_result.global_bin_dsc, 1.0)

        # Test channel metrics
        channel_dsc = part_result.get_channel_metrics("dsc")
        self.assertIsNotNone(channel_dsc)
        # Should have different values for different channels
        if len(channel_dsc) > 1:
            values = list(channel_dsc.values())
            self.assertFalse(all(abs(v - values[0]) < 1e-6 for v in values))

    def test_multipart_channel_metrics(self):
        """Test multi-part groups with multiple part types."""
        pred_masks, ref_masks = self.create_basic_masks()

        # Create scenario with multiple part types
        # Stuff Classes first
        pred_masks[1:30, 1:30] = 4
        ref_masks[1:30, 1:30] = 4

        # Thing Classes next
        pred_masks[7:20, 7:20] = 1
        ref_masks[7:20, 7:20] = 1
        pred_masks[22:27, 22:27] = 3
        ref_masks[22:27, 22:27] = 3

        # Multiple Part Classes
        pred_masks[8:13, 8:11] = 2
        ref_masks[8:13, 8:10] = 2
        pred_masks[14:19, 14:18] = 5
        ref_masks[14:19, 14:19] = 5

        class_groups = SegmentationClassGroups(self.get_multipart_groups())
        evaluator = Panoptica_Evaluator(
            expected_input=InputType.SEMANTIC,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=MaxBipartiteMatching(),
            segmentation_class_groups=class_groups,
            global_metrics=[Metric.DSC, Metric.IOU, Metric.ASSD],
        )

        results = evaluator.evaluate(pred_masks, ref_masks, verbose=False)
        part_result = results["part"]

        # Test basic metrics
        self.assertEqual(part_result.tp, 1)
        self.assertEqual(part_result.fp, 0)
        self.assertEqual(part_result.fn, 0)

        # Test that global metrics account for multiple part types
        self.assertGreater(part_result.global_bin_dsc, 0.8)
        self.assertLess(part_result.global_bin_dsc, 1.0)

        # Test channel metrics for multiple part types
        channel_dsc = part_result.get_channel_metrics("dsc")
        self.assertIsNotNone(channel_dsc)

    def test_empty_channel_handling(self):
        """Test handling of empty channels in multi-channel metrics."""
        pred_masks, ref_masks = self.create_basic_masks()

        # Create scenario where some channels are empty
        # Only thing classes, no parts
        pred_masks[1:30, 1:30] = 4
        ref_masks[1:30, 1:30] = 4
        pred_masks[7:20, 7:10] = 1  # Different sizes
        ref_masks[7:20, 7:20] = 1
        pred_masks[22:27, 22:27] = 3
        ref_masks[22:27, 22:27] = 3

        class_groups = SegmentationClassGroups(self.get_multipart_groups())
        evaluator = Panoptica_Evaluator(
            expected_input=InputType.SEMANTIC,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=MaxBipartiteMatching(),
            segmentation_class_groups=class_groups,
            global_metrics=[Metric.DSC],
        )

        results = evaluator.evaluate(pred_masks, ref_masks, verbose=False)
        part_result = results["part"]

        # Should handle empty part channels gracefully
        # When no parts are present in either pred or ref, tp should be 0
        self.assertEqual(part_result.tp, 0)
        self.assertIsNotNone(part_result.global_bin_dsc)

        # Test that channel metrics don't include empty channels
        channel_dsc = part_result.get_channel_metrics("dsc")
        if channel_dsc:
            # Empty part channels should be skipped
            for channel_id, value in channel_dsc.items():
                self.assertFalse(np.isnan(value))

    def test_edge_case_metric_handling(self):
        """Test edge case handling in multi-channel metrics."""
        pred_masks, ref_masks = self.create_basic_masks()

        # Create edge case: parts in prediction but not in reference
        pred_masks[1:30, 1:30] = 4
        ref_masks[1:30, 1:30] = 4
        pred_masks[1:4, 1:4] = 1
        ref_masks[1:3, 1:3] = 1
        pred_masks[7:20, 7:20] = 1
        ref_masks[7:20, 7:20] = 1
        pred_masks[22:27, 22:27] = 3
        ref_masks[22:27, 22:27] = 3

        # Parts only in prediction
        pred_masks[8:13, 8:13] = 2
        pred_masks[14:19, 14:19] = 2
        pred_masks[10:15, 22:25] = 2

        class_groups = SegmentationClassGroups(self.get_part_groups())
        evaluator = Panoptica_Evaluator(
            expected_input=InputType.SEMANTIC,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=MaxBipartiteMatching(),
            segmentation_class_groups=class_groups,
            global_metrics=[Metric.DSC, Metric.IOU],
        )

        results = evaluator.evaluate(pred_masks, ref_masks, verbose=False)
        part_result = results["part"]

        # Should handle the edge case without crashing
        self.assertIsNotNone(part_result.global_bin_dsc)
        self.assertFalse(np.isnan(part_result.global_bin_dsc))

        # Channel metrics should be available
        channel_dsc = part_result.get_channel_metrics("dsc")
        self.assertIsNotNone(channel_dsc)

    def test_no_global_metrics_set(self):
        """Test behavior when no global metrics are requested."""
        pred_masks, ref_masks = self.create_basic_masks()

        # Simple perfect overlap
        pred_masks[1:30, 1:30] = 4
        ref_masks[1:30, 1:30] = 4
        pred_masks[7:20, 7:20] = 1
        ref_masks[7:20, 7:20] = 1
        pred_masks[14:19, 14:19] = 2
        ref_masks[14:19, 14:19] = 2

        class_groups = SegmentationClassGroups(self.get_part_groups())
        evaluator = Panoptica_Evaluator(
            expected_input=InputType.SEMANTIC,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=MaxBipartiteMatching(),
            segmentation_class_groups=class_groups,
            global_metrics=[],  # No global metrics
        )

        results = evaluator.evaluate(pred_masks, ref_masks, verbose=False)
        part_result = results["part"]

        # Basic metrics should still work
        self.assertEqual(part_result.tp, 1)

        # Channel metrics should be None when no global metrics set
        channel_dsc = part_result.get_channel_metrics("dsc")
        self.assertIsNone(channel_dsc)

    def test_channel_metrics_consistency(self):
        """Test that channel metrics are consistent with overall global metrics."""
        pred_masks, ref_masks = self.create_basic_masks()

        # Create a specific scenario for testing consistency
        pred_masks[1:30, 1:30] = 4
        ref_masks[1:30, 1:30] = 4
        pred_masks[7:20, 7:20] = 1
        ref_masks[7:20, 7:20] = 1
        pred_masks[22:27, 22:27] = 3
        ref_masks[22:27, 22:27] = 3
        pred_masks[14:19, 14:19] = 2
        ref_masks[14:17, 14:17] = 2  # Partial overlap in parts

        class_groups = SegmentationClassGroups(self.get_part_groups())
        evaluator = Panoptica_Evaluator(
            expected_input=InputType.SEMANTIC,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=MaxBipartiteMatching(),
            segmentation_class_groups=class_groups,
            global_metrics=[Metric.DSC],
        )

        results = evaluator.evaluate(pred_masks, ref_masks, verbose=False)
        part_result = results["part"]

        # Get channel metrics
        channel_dsc = part_result.get_channel_metrics("dsc")

        if channel_dsc and len(channel_dsc) > 0:
            # Global metric should be the mean of channel metrics
            channel_values = list(channel_dsc.values())
            expected_global = np.mean(channel_values)

            # Allow for small numerical differences
            self.assertAlmostEqual(
                part_result.global_bin_dsc,
                expected_global,
                places=4,
                msg=f"Global DSC {part_result.global_bin_dsc} should equal mean of channel DSCs {expected_global}",
            )


class Test_Part_Metrics_Integration(unittest.TestCase):
    """Integration tests for part metrics with different scenarios."""

    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"
        return super().setUp()

    def test_comparison_merge_vs_part_groups(self):
        """Test that LabelMergeGroup and LabelPartGroup give different results."""
        pred_masks = np.zeros((28, 28), dtype=np.int32)
        ref_masks = np.zeros((28, 28), dtype=np.int32)

        # Create scenario
        pred_masks[1:30, 1:30] = 4
        ref_masks[1:30, 1:30] = 4
        pred_masks[7:20, 7:20] = 1
        ref_masks[7:20, 7:20] = 1
        pred_masks[14:19, 14:19] = 2
        ref_masks[14:15, 14:19] = 2  # Partial overlap

        # Test with LabelMergeGroup
        merge_groups = SegmentationClassGroups(
            {
                "class_1": (1, False),
                "class_2": (2, False),
                "class_4": (4, True),
                "Merge": LabelMergeGroup([1, 2], False),
            }
        )

        # Test with LabelPartGroup
        part_groups = SegmentationClassGroups(
            {
                "class_1": (1, False),
                "class_2": (2, False),
                "class_4": (4, True),
                "Part": LabelPartGroup([1], [2], False),
            }
        )

        evaluator_merge = Panoptica_Evaluator(
            expected_input=InputType.SEMANTIC,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=MaxBipartiteMatching(),
            segmentation_class_groups=merge_groups,
            global_metrics=[Metric.DSC],
        )

        evaluator_part = Panoptica_Evaluator(
            expected_input=InputType.SEMANTIC,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=MaxBipartiteMatching(),
            segmentation_class_groups=part_groups,
            global_metrics=[Metric.DSC],
        )

        merge_results = evaluator_merge.evaluate(pred_masks, ref_masks, verbose=False)
        part_results = evaluator_part.evaluate(pred_masks, ref_masks, verbose=False)

        merge_result = merge_results["merge"]
        part_result = part_results["part"]

        # Results should be different due to different evaluation logic
        # LabelMergeGroup treats everything as binary, LabelPartGroup is multi-channel
        self.assertNotEqual(merge_result.global_bin_dsc, part_result.global_bin_dsc)

        # Part group should have channel metrics, merge group should not
        self.assertIsNone(merge_result.get_channel_metrics("dsc"))
        self.assertIsNotNone(part_result.get_channel_metrics("dsc"))
