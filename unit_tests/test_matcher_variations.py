# Call 'python -m unittest' on this folder
# coverage run -m unittest
# coverage report
# coverage html
import os
import unittest
import numpy as np
import warnings
import sys
from pathlib import Path

# Add the parent directory to the sys.path to import panoptica
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Suppress SWIG-related deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*SwigPy.*")
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, message=".*swigvarlink.*"
)

from panoptica import Panoptica_Evaluator, InputType
from panoptica.utils.segmentation_class import SegmentationClassGroups
from panoptica.instance_approximator import ConnectedComponentsInstanceApproximator
from panoptica.instance_matcher import NaiveThresholdMatching, MaxBipartiteMatching
from panoptica.utils.label_group import LabelPartGroup


class BaseMatcherTest(unittest.TestCase):
    """Base class for matcher tests with common functionality."""

    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"
        return super().setUp()

    def create_evaluator(self, matcher, groups):
        """Create a Panoptica evaluator with given matcher and groups."""
        class_groups = SegmentationClassGroups(groups)
        return Panoptica_Evaluator(
            expected_input=InputType.SEMANTIC,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=matcher,
            segmentation_class_groups=class_groups,
        )

    def create_masks(self, size=(28, 28)):
        """Create empty prediction and reference masks."""
        return np.zeros(size, dtype=np.int32), np.zeros(size, dtype=np.int32)

    def assert_metrics_match(
        self, actual_results, expected_results, class_name="class_1"
    ):
        """Assert that all important metrics match expected values."""
        class_result = actual_results[class_name].to_dict()
        important_metrics = [
            "tp",
            "fp",
            "fn",
            "sq",
            "rq",
            "pq",
            "sq_dsc",
            "global_bin_dsc",
        ]

        for metric in important_metrics:
            with self.subTest(metric=metric):
                actual_value = class_result[metric]
                expected_value = expected_results[metric]

                if isinstance(actual_value, float):
                    self.assertAlmostEqual(
                        actual_value,
                        expected_value,
                        places=4,
                        msg=f"Metric {metric}: expected {expected_value}, got {actual_value}",
                    )
                else:
                    self.assertEqual(
                        actual_value,
                        expected_value,
                        msg=f"Metric {metric}: expected {expected_value}, got {actual_value}",
                    )

    def run_overlap_scenario(
        self,
        matcher,
        pred_setup_func,
        ref_setup_func,
        expected_results,
        groups=None,
        class_name="class_1",
    ):
        """Generic helper method for different overlap scenarios."""
        if groups is None:
            groups = {"class_1": (1, False)}

        evaluator = self.create_evaluator(matcher, groups)
        pred_masks, ref_masks = self.create_masks()

        # Apply setup functions to create the specific test scenario
        pred_setup_func(pred_masks)
        ref_setup_func(ref_masks)

        # Run evaluation and assert results
        results = evaluator.evaluate(pred_masks, ref_masks, verbose=False)
        self.assert_metrics_match(results, expected_results, class_name)


class Test_Naive_Matcher_Variations(BaseMatcherTest):

    def test_perfect_overlap_masks(self):
        """Test case with perfect overlap between prediction and reference masks."""

        def setup_perfect_overlap(masks):
            masks[1:28, 1:28] = 1

        expected_results = {
            "tp": 1,
            "fp": 0,
            "fn": 0,
            "sq": 1.0,
            "rq": 1.0,
            "pq": 1.0,
            "sq_dsc": 1.0,
            "global_bin_dsc": 1.0,
        }

        self.run_overlap_scenario(
            NaiveThresholdMatching(matching_threshold=0.0),
            setup_perfect_overlap,
            setup_perfect_overlap,
            expected_results,
        )

    def test_no_overlap_masks(self):
        """Test case with no overlap between prediction and reference masks."""

        def setup_pred_no_overlap(masks):
            masks[1:14, 1:14] = 1

        def setup_ref_no_overlap(masks):
            masks[15:28, 15:28] = 1

        expected_results = {
            "tp": 0,
            "fp": 1,
            "fn": 1,
            "sq": 0.0,
            "rq": 0.0,
            "pq": 0.0,
            "sq_dsc": 0.0,
            "global_bin_dsc": 0.0,
        }

        self.run_overlap_scenario(
            NaiveThresholdMatching(matching_threshold=0.0),
            setup_pred_no_overlap,
            setup_ref_no_overlap,
            expected_results,
        )


class Test_Bipartite_Matcher_Variations(BaseMatcherTest):

    def test_perfect_overlap_masks(self):
        """Test case with perfect overlap between prediction and reference masks."""

        def setup_perfect_overlap(masks):
            masks[1:28, 1:28] = 1

        expected_results = {
            "tp": 1,
            "fp": 0,
            "fn": 0,
            "sq": 1.0,
            "rq": 1.0,
            "pq": 1.0,
            "sq_dsc": 1.0,
            "global_bin_dsc": 1.0,
        }

        self.run_overlap_scenario(
            MaxBipartiteMatching(matching_threshold=0.0),
            setup_perfect_overlap,
            setup_perfect_overlap,
            expected_results,
        )

    def test_no_overlap_masks(self):
        """Test case with no overlap between prediction and reference masks."""

        def setup_pred_no_overlap(masks):
            masks[1:14, 1:14] = 1

        def setup_ref_no_overlap(masks):
            masks[15:28, 15:28] = 1

        expected_results = {
            "tp": 0,
            "fp": 1,
            "fn": 1,
            "sq": 0.0,
            "rq": 0.0,
            "pq": 0.0,
            "sq_dsc": 0.0,
            "global_bin_dsc": 0.0,
        }

        # Note: Original code used NaiveThresholdMatching here, which seems like a bug
        # Changed to MaxBipartiteMatching to match the class name
        self.run_overlap_scenario(
            MaxBipartiteMatching(matching_threshold=0.0),
            setup_pred_no_overlap,
            setup_ref_no_overlap,
            expected_results,
        )


class Test_Part_Matcher_Variations(BaseMatcherTest):

    def get_part_groups(self):
        """Get the complex groups configuration for part tests."""
        return {
            "class_1": (1, False),
            "class_2": (2, False),
            "class_3": (3, False),
            "class_4": (4, True),
            "Part": LabelPartGroup([1], [2], False),
        }

    def test_perfect_overlap_masks(self):
        """Test case with perfect overlap between prediction and reference masks."""

        def setup_perfect(masks):
            # Stuff Classes first (lowest hierarchy)
            masks[1:30, 1:30] = 4
            # Thing Classes next (medium hierarchy)
            masks[7:20, 7:20] = 1
            masks[22:27, 22:27] = 3
            # Part Classes last (highest hierarchy)
            masks[14:19, 14:19] = 2

        expected_results = {
            "tp": 1,
            "fp": 0,
            "fn": 0,
            "sq": 1.0,
            "rq": 1.0,
            "pq": 1.0,
            "sq_dsc": 1.0,
            "global_bin_dsc": 1.0,
        }

        self.run_overlap_scenario(
            MaxBipartiteMatching(),
            setup_perfect,
            setup_perfect,
            expected_results,
            self.get_part_groups(),
            "part",
        )

    def test_partial_overlap_masks(self):
        """Test case with partial overlap between prediction and reference masks."""

        def setup_pred_partial(masks):
            # Stuff Classes first (lowest hierarchy)
            masks[1:30, 1:30] = 4
            # Thing Classes next (medium hierarchy)
            masks[7:20, 7:20] = 1
            masks[22:27, 22:27] = 3
            # Part Classes last (highest hierarchy)
            masks[14:19, 14:19] = 2

        def setup_ref_partial(masks):
            # Stuff Classes first (lowest hierarchy)
            masks[1:30, 1:30] = 4
            # Thing Classes next (medium hierarchy)
            masks[7:20, 7:20] = 1
            masks[22:27, 22:27] = 3
            # Part Classes last (highest hierarchy) - smaller area
            masks[14:15, 14:19] = 2

        expected_results = {
            "tp": 1,
            "fp": 0,
            "fn": 0,
            "sq": 0.8883,
            "rq": 1.0,
            "pq": 0.8883,
            "sq_dsc": 0.9408,
            "global_bin_dsc": 0.6667,
        }

        self.run_overlap_scenario(
            MaxBipartiteMatching(matching_threshold=0.0),
            setup_pred_partial,
            setup_ref_partial,
            expected_results,
            self.get_part_groups(),
            "part",
        )

    def test_isolation_masks(self):
        """Test case with isolation - multiple part instances in prediction, single in reference."""

        def setup_pred_isolation(masks):
            # Stuff Classes first (lowest hierarchy)
            masks[1:30, 1:30] = 4
            # Thing Classes next (medium hierarchy)
            masks[7:20, 7:20] = 1
            masks[22:27, 22:27] = 3
            # Part Classes last (highest hierarchy) - multiple instances
            masks[8:13, 8:13] = 2
            masks[14:19, 14:19] = 2
            masks[10:15, 22:25] = 2

        def setup_ref_isolation(masks):
            # Stuff Classes first (lowest hierarchy)
            masks[1:30, 1:30] = 4
            # Thing Classes next (medium hierarchy)
            masks[7:20, 7:20] = 1
            masks[22:27, 22:27] = 3
            # Part Classes last (highest hierarchy) - single instance
            masks[14:19, 14:19] = 2

        expected_results = {
            "tp": 1,
            "fp": 1,
            "fn": 0,
            "sq": 0.7934,
            "rq": 0.6667,
            "pq": 0.5289,
            "sq_dsc": 0.8848,
            "global_bin_dsc": 0.7565,
        }

        self.run_overlap_scenario(
            MaxBipartiteMatching(matching_threshold=0.0),
            setup_pred_isolation,
            setup_ref_isolation,
            expected_results,
            self.get_part_groups(),
            "part",
        )

    def test_uneven_pairs_masks(self):
        """Test case with uneven pairs - unequal numbers of thing class instances between prediction and reference."""

        def setup_pred_uneven(masks):
            # Stuff Classes first (lowest hirearchy)
            masks[1:30, 1:30] = 4

            # Thing Classes next (medium hirearchy)
            masks[1:4, 1:4] = 1
            masks[7:20, 7:20] = 1
            masks[22:27, 22:27] = 3

            # Part Classes last (highest hirearchy)
            masks[8:13, 8:13] = 2
            masks[14:19, 14:19] = 2

        def setup_ref_uneven(masks):
            masks[1:30, 1:30] = 4

            # Thing Classes next
            masks[1:3, 1:3] = 1
            masks[7:20, 7:20] = 1
            masks[22:27, 22:27] = 3

            # Part Classes last
            masks[14:19, 14:19] = 2

        expected_results = {
            "tp": 2,
            "fp": 0,
            "fn": 0,
            "sq": 0.4849,
            "rq": 1.0,
            "pq": 0.4849,
            "sq_dsc": 0.5653,
            "global_bin_dsc": 0.8262,
        }

        self.run_overlap_scenario(
            MaxBipartiteMatching(matching_threshold=0.0),
            setup_pred_uneven,
            setup_ref_uneven,
            expected_results,
            self.get_part_groups(),
            "part",
        )

    def test_uneven_class_masks(self):
        """Test case with uneven class - part instances in prediction but none in reference."""

        def setup_pred_uneven_class(masks):
            # Stuff Classes first (lowest hierarchy)
            masks[1:30, 1:30] = 4
            # Thing Classes next (medium hierarchy)
            masks[1:4, 1:4] = 1
            masks[7:20, 7:20] = 1
            masks[22:27, 22:27] = 3
            # Part Classes last (highest hierarchy) - multiple instances
            masks[8:13, 8:13] = 2
            masks[14:19, 14:19] = 2
            masks[10:15, 22:25] = 2

        def setup_ref_uneven_class(masks):
            # Stuff Classes first (lowest hierarchy)
            masks[1:30, 1:30] = 4
            # Thing Classes next (medium hierarchy)
            masks[1:3, 1:3] = 1
            masks[7:20, 7:20] = 1
            masks[22:27, 22:27] = 3
            # Part Classes last (highest hierarchy) - no instances
            # ref_masks[14:19, 14:19] = 2  # commented out - no parts in reference

        expected_results = {
            "tp": 2,
            "fp": 1,
            "fn": 0,
            "sq": 0.501,
            "rq": 0.8,
            "pq": 0.4008,
            "sq_dsc": 0.6062,
            "global_bin_dsc": 0.4727,
        }

        self.run_overlap_scenario(
            MaxBipartiteMatching(matching_threshold=0.0),
            setup_pred_uneven_class,
            setup_ref_uneven_class,
            expected_results,
            self.get_part_groups(),
            "part",
        )


class Test_MultiPart_Matcher_Variations(BaseMatcherTest):

    def get_part_groups(self):
        """Get the complex groups configuration for part tests."""
        return {
            "class_1": (1, False),
            "class_2": (2, False),
            "extra": (5, False),  # Extra class for multi-part tests
            "class_3": (3, False),
            "class_4": (4, True),
            "Part": LabelPartGroup([1], [2, 5], False),
        }

    def test_perfect_overlap_masks(self):
        """Test case with perfect overlap between prediction and reference masks."""

        def setup_perfect_overlap(masks):
            # Stuff Classes first (lowest hierarchy)
            masks[1:30, 1:30] = 4
            # Thing Classes next (medium hierarchy)
            masks[7:20, 7:20] = 1
            masks[22:27, 22:27] = 3
            # Part Classes last (highest hierarchy)
            masks[8:13, 8:10] = 2
            masks[14:19, 14:18] = 5

        expected_results = {
            "tp": 1,
            "fp": 0,
            "fn": 0,
            "sq": 1.0,
            "rq": 1.0,
            "pq": 1.0,
            "sq_dsc": 1.0,
            "global_bin_dsc": 1.0,
        }

        self.run_overlap_scenario(
            MaxBipartiteMatching(matching_threshold=0.0),
            setup_perfect_overlap,
            setup_perfect_overlap,
            expected_results,
            self.get_part_groups(),
            "part",
        )

    def test_partial_overlap_masks(self):
        """Test case with partial overlap between prediction and reference masks."""

        def setup_pred_partial(masks):
            # Stuff Classes first (lowest hierarchy)
            masks[1:30, 1:30] = 4
            # Thing Classes next (medium hierarchy)
            masks[7:20, 7:20] = 1
            masks[22:27, 22:27] = 3
            # Part Classes last (highest hierarchy)
            masks[8:13, 8:11] = 2
            masks[14:19, 14:18] = 5

        def setup_ref_partial(masks):
            # Stuff Classes first (lowest hierarchy)
            masks[1:30, 1:30] = 4
            # Thing Classes next (medium hierarchy)
            masks[7:20, 7:20] = 1
            masks[22:27, 22:27] = 3
            # Part Classes last (highest hierarchy)
            masks[8:13, 8:10] = 2
            masks[14:19, 14:19] = 5

        expected_results = {
            "tp": 1,
            "fp": 0,
            "fn": 0,
            "sq": 0.9425,
            "rq": 1.0,
            "pq": 0.9425,
            "sq_dsc": 0.9704,
            "global_bin_dsc": 0.8963,
        }

        self.run_overlap_scenario(
            MaxBipartiteMatching(matching_threshold=0.0),
            setup_pred_partial,
            setup_ref_partial,
            expected_results,
            self.get_part_groups(),
            "part",
        )

    def test_uneven_classes_masks(self):
        """Test case with uneven classes - different part class types between prediction and reference."""

        def setup_pred_uneven_classes(masks):
            # Stuff Classes first (lowest hierarchy)
            masks[1:30, 1:30] = 4
            # Thing Classes next (medium hierarchy)
            masks[7:20, 7:20] = 1
            masks[22:27, 22:27] = 3
            # Part Classes last (highest hierarchy)
            masks[8:13, 8:11] = 2
            masks[14:19, 14:18] = 5

        def setup_ref_uneven_classes(masks):
            # Stuff Classes first (lowest hierarchy)
            masks[1:30, 1:30] = 4
            # Thing Classes next (medium hierarchy)
            masks[7:20, 7:20] = 1
            masks[22:27, 22:27] = 3
            # Part Classes last (highest hierarchy) - only class 5, no class 2
            masks[14:19, 14:19] = 5

        expected_results = {
            "tp": 1,
            "fp": 0,
            "fn": 0,
            "sq": 0.8883,
            "rq": 1.0,
            "pq": 0.8883,
            "sq_dsc": 0.9408,
            "global_bin_dsc": 0.6296,
        }

        self.run_overlap_scenario(
            MaxBipartiteMatching(matching_threshold=0.0),
            setup_pred_uneven_classes,
            setup_ref_uneven_classes,
            expected_results,
            self.get_part_groups(),
            "part",
        )

    def test_no_parts_masks(self):
        """Test case with no part classes - only thing classes present."""

        def setup_pred_no_parts(masks):
            # Stuff Classes first (lowest hierarchy)
            masks[1:30, 1:30] = 4
            # Thing Classes next (medium hierarchy) - different size than reference
            masks[7:20, 7:10] = 1
            masks[22:27, 22:27] = 3
            # Part Classes last (highest hierarchy) - none present

        def setup_ref_no_parts(masks):
            # Stuff Classes first (lowest hierarchy)
            masks[1:30, 1:30] = 4
            # Thing Classes next (medium hierarchy)
            masks[7:20, 7:20] = 1
            masks[22:27, 22:27] = 3
            # Part Classes last (highest hierarchy) - none present

        expected_results = {
            "tp": 1,
            "fp": 0,
            "fn": 0,
            "sq": 0.2308,
            "rq": 1.0,
            "pq": 0.2308,
            "sq_dsc": 0.375,
            "global_bin_dsc": 0.375,
        }

        self.run_overlap_scenario(
            MaxBipartiteMatching(matching_threshold=0.0),
            setup_pred_no_parts,
            setup_ref_no_parts,
            expected_results,
            self.get_part_groups(),
            "part",
        )
