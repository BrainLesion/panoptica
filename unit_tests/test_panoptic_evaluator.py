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


class Test_Example_Scripts(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"
        return super().setUp()

    def test_example_scripts(self):
        directory = Path(__file__).parent.parent.joinpath("examples")

        print(directory)
        if not directory.exists():
            self.skipTest(f"directory {directory} does not exist")

        sys.path.append(str(directory))

        from examples.example_spine_semantic import main

        main()

        from examples.example_spine_instance import main

        main()

        from examples.example_spine_instance_config import main

        main()


class Test_Panoptica_Evaluator(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"
        return super().setUp()

    def test_simple_evaluation(self):
        a = np.zeros([50, 50], dtype=np.uint16)
        b = a.copy().astype(a.dtype)
        a[20:40, 10:20] = 1
        b[20:35, 10:20] = 2

        evaluator = Panoptica_Evaluator(
            expected_input=InputType.SEMANTIC,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=NaiveThresholdMatching(),
        )

        result, debug_data = evaluator.evaluate(b, a)["ungrouped"]
        print(result)
        self.assertEqual(result.tp, 1)
        self.assertEqual(result.fp, 0)
        self.assertEqual(result.sq, 0.75)
        self.assertEqual(result.pq, 0.75)
        self.assertAlmostEqual(result.global_bin_dsc, 0.8571428571428571)

    def test_simple_evaluation_instance_multiclass(self):
        a = np.zeros([50, 50], dtype=np.uint16)
        b = a.copy().astype(a.dtype)
        a[20:30, 10:20] = 1
        a[30:40, 10:20] = 3
        b[20:35, 10:20] = 2

        evaluator = Panoptica_Evaluator(
            expected_input=InputType.UNMATCHED_INSTANCE,
            instance_matcher=NaiveThresholdMatching(),
        )

        result, debug_data = evaluator.evaluate(b, a)["ungrouped"]
        print(result)
        self.assertAlmostEqual(result.global_bin_dsc, 0.8571428571428571)
        self.assertEqual(result.tp, 1)
        self.assertEqual(result.fp, 0)
        self.assertEqual(result.fn, 1)
        self.assertAlmostEqual(result.sq, 0.6666666666666666)
        self.assertAlmostEqual(result.pq, 0.4444444444444444)

    def test_simple_evaluation_DSC(self):
        a = np.zeros([50, 50], dtype=np.uint16)
        b = a.copy().astype(a.dtype)
        a[20:40, 10:20] = 1
        b[20:35, 10:20] = 2

        evaluator = Panoptica_Evaluator(
            expected_input=InputType.SEMANTIC,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=NaiveThresholdMatching(),
        )

        result, debug_data = evaluator.evaluate(b, a)["ungrouped"]
        print(result)
        self.assertEqual(result.tp, 1)
        self.assertEqual(result.fp, 0)
        self.assertEqual(result.sq, 0.75)
        self.assertEqual(result.pq, 0.75)

    def test_simple_evaluation_DSC_partial(self):
        a = np.zeros([50, 50], dtype=np.uint16)
        b = a.copy().astype(a.dtype)
        a[20:40, 10:20] = 1
        b[20:35, 10:20] = 2

        evaluator = Panoptica_Evaluator(
            expected_input=InputType.SEMANTIC,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=NaiveThresholdMatching(matching_metric=Metric.DSC),
            instance_metrics=[Metric.DSC],
        )

        result, debug_data = evaluator.evaluate(b, a)["ungrouped"]
        print(result)
        self.assertEqual(result.tp, 1)
        self.assertEqual(result.fp, 0)
        with self.assertRaises(MetricCouldNotBeComputedException):
            result.sq
        # must be none because no IOU has been calculated
        with self.assertRaises(MetricCouldNotBeComputedException):
            result.pq
        self.assertEqual(result.rq, 1.0)

    def test_simple_evaluation_ASSD(self):
        a = np.zeros([50, 50], dtype=np.uint16)
        b = a.copy().astype(a.dtype)
        a[20:40, 10:20] = 1
        b[20:35, 10:20] = 2

        evaluator = Panoptica_Evaluator(
            expected_input=InputType.SEMANTIC,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=NaiveThresholdMatching(
                matching_metric=Metric.ASSD,
                matching_threshold=1.0,
            ),
        )

        result, debug_data = evaluator.evaluate(b, a)["ungrouped"]
        print(result)
        self.assertEqual(result.tp, 1)
        self.assertEqual(result.fp, 0)
        self.assertEqual(result.sq, 0.75)
        self.assertEqual(result.pq, 0.75)

    def test_simple_evaluation_ASSD_negative(self):
        a = np.zeros([50, 50], dtype=np.uint16)
        b = a.copy().astype(a.dtype)
        a[20:40, 10:20] = 1
        b[20:35, 10:20] = 2

        evaluator = Panoptica_Evaluator(
            expected_input=InputType.SEMANTIC,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=NaiveThresholdMatching(
                matching_metric=Metric.ASSD,
                matching_threshold=0.5,
            ),
        )

        result, debug_data = evaluator.evaluate(b, a)["ungrouped"]
        print(result)
        self.assertEqual(result.tp, 0)
        self.assertEqual(result.fp, 1)
        self.assertEqual(result.sq, 0.0)
        self.assertEqual(result.pq, 0.0)
        self.assertEqual(result.sq_assd, np.inf)

    def test_pred_empty(self):
        a = np.zeros([50, 50], np.uint16)
        b = a.copy()
        a[20:40, 10:20] = 1
        # b[20:35, 10:20] = 2

        evaluator = Panoptica_Evaluator(
            expected_input=InputType.SEMANTIC,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=NaiveThresholdMatching(),
        )

        result, debug_data = evaluator.evaluate(b, a)["ungrouped"]
        print(result)
        self.assertEqual(result.tp, 0)
        self.assertEqual(result.fp, 0)
        self.assertEqual(result.fn, 1)
        self.assertEqual(result.sq, 0.0)
        self.assertEqual(result.pq, 0.0)
        self.assertEqual(result.global_bin_dsc, 0.0)
        self.assertEqual(result.sq_assd, np.inf)

    def test_no_TP_but_overlap(self):
        a = np.zeros([50, 50], np.uint16)
        b = a.copy()
        a[20:40, 10:20] = 1
        b[20:25, 10:15] = 2

        evaluator = Panoptica_Evaluator(
            expected_input=InputType.SEMANTIC,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=NaiveThresholdMatching(),
        )

        result, debug_data = evaluator.evaluate(b, a)["ungrouped"]
        print(result)
        self.assertEqual(result.tp, 0)
        self.assertEqual(result.fp, 1)
        self.assertEqual(result.fn, 1)
        self.assertEqual(result.sq, 0.0)
        self.assertEqual(result.pq, 0.0)
        self.assertAlmostEqual(result.global_bin_dsc, 0.22222222222)
        self.assertEqual(result.sq_assd, np.inf)
        self.assertTrue(np.isnan(result.sq_rvd))

    def test_ref_empty(self):
        a = np.zeros([50, 50], np.uint16)
        b = a.copy()
        # a[20:40, 10:20] = 1
        b[20:35, 10:20] = 2

        evaluator = Panoptica_Evaluator(
            expected_input=InputType.SEMANTIC,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=NaiveThresholdMatching(),
        )

        result, debug_data = evaluator.evaluate(b, a)["ungrouped"]
        print(result)
        self.assertEqual(result.tp, 0)
        self.assertEqual(result.fp, 1)
        self.assertEqual(result.fn, 0)
        self.assertEqual(result.sq, 0.0)
        self.assertEqual(result.pq, 0.0)
        self.assertEqual(result.sq_assd, np.inf)

    def test_both_empty(self):
        a = np.zeros([50, 50], np.uint16)
        b = a.copy()
        # a[20:40, 10:20] = 1
        # b[20:35, 10:20] = 2

        evaluator = Panoptica_Evaluator(
            expected_input=InputType.SEMANTIC,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=NaiveThresholdMatching(),
        )

        result, debug_data = evaluator.evaluate(b, a)["ungrouped"]
        print(result)
        self.assertEqual(result.tp, 0)
        self.assertEqual(result.fp, 0)
        self.assertEqual(result.fn, 0)
        self.assertTrue(np.isnan(result.sq))
        self.assertTrue(np.isnan(result.pq))
        self.assertTrue(np.isnan(result.sq_assd))

    def test_dtype_evaluation(self):
        ddtypes = [
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        ]
        dtype_combinations = [(a, b) for a in ddtypes for b in ddtypes]
        for da, db in dtype_combinations:
            a = np.zeros([50, 50], dtype=da)
            b = a.copy().astype(db)
            a[20:40, 10:20] = 1
            b[20:35, 10:20] = 2

            evaluator = Panoptica_Evaluator(
                expected_input=InputType.SEMANTIC,
                instance_approximator=ConnectedComponentsInstanceApproximator(),
                instance_matcher=NaiveThresholdMatching(),
            )

            result, debug_data = evaluator.evaluate(b, a)["ungrouped"]
            print(result)
            self.assertEqual(result.tp, 1)
            self.assertEqual(result.fp, 0)
            self.assertEqual(result.sq, 0.75)
            self.assertEqual(result.pq, 0.75)

    def test_simple_evaluation_maximize_matcher(self):
        a = np.zeros([50, 50], dtype=np.uint16)
        b = a.copy().astype(a.dtype)
        a[20:40, 10:20] = 1
        b[20:35, 10:20] = 2

        evaluator = Panoptica_Evaluator(
            expected_input=InputType.SEMANTIC,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=MaximizeMergeMatching(),
        )

        result, debug_data = evaluator.evaluate(b, a)["ungrouped"]
        print(result)
        self.assertEqual(result.tp, 1)
        self.assertEqual(result.fp, 0)
        self.assertEqual(result.sq, 0.75)
        self.assertEqual(result.pq, 0.75)

    def test_simple_evaluation_maximize_matcher_overlaptwo(self):
        a = np.zeros([50, 50], dtype=np.uint16)
        b = a.copy().astype(a.dtype)
        a[20:40, 10:20] = 1
        b[20:35, 10:20] = 2
        b[36:38, 10:20] = 3

        evaluator = Panoptica_Evaluator(
            expected_input=InputType.SEMANTIC,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=MaximizeMergeMatching(),
        )

        result, debug_data = evaluator.evaluate(b, a)["ungrouped"]
        print(result)
        self.assertEqual(result.tp, 1)
        self.assertEqual(result.fp, 0)
        self.assertEqual(result.sq, 0.85)
        self.assertEqual(result.pq, 0.85)

    def test_simple_evaluation_maximize_matcher_overlap(self):
        a = np.zeros([50, 50], dtype=np.uint16)
        b = a.copy().astype(a.dtype)
        a[20:40, 10:20] = 1
        b[20:35, 10:20] = 2
        b[36:38, 10:20] = 3
        # match the two above to 1 and the 4 to nothing (FP)
        b[39:47, 10:20] = 4

        evaluator = Panoptica_Evaluator(
            expected_input=InputType.SEMANTIC,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=MaximizeMergeMatching(),
        )

        result, debug_data = evaluator.evaluate(b, a)["ungrouped"]
        print(result)
        self.assertEqual(result.tp, 1)
        self.assertEqual(result.fp, 1)
        self.assertEqual(result.sq, 0.85)
        self.assertAlmostEqual(result.pq, 0.56666666)
        self.assertAlmostEqual(result.rq, 0.66666666)
        self.assertAlmostEqual(result.sq_dsc, 0.9189189189189)

    def test_single_instance_mode(self):
        a = np.zeros([50, 50], dtype=np.uint16)
        b = a.copy().astype(a.dtype)
        a[20:40, 10:20] = 5
        b[20:35, 10:20] = 5

        evaluator = Panoptica_Evaluator(
            expected_input=InputType.SEMANTIC,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=NaiveThresholdMatching(),
            segmentation_class_groups=SegmentationClassGroups({"organ": (5, True)}),
        )

        result, debug_data = evaluator.evaluate(b, a)["organ"]
        print(result)
        self.assertEqual(result.tp, 1)
        self.assertEqual(result.fp, 0)
        self.assertEqual(result.sq, 0.75)
        self.assertEqual(result.pq, 0.75)

    def test_single_instance_mode_nooverlap(self):
        a = np.zeros([50, 50], dtype=np.uint16)
        b = a.copy().astype(a.dtype)
        a[20:40, 10:20] = 5
        b[5:15, 30:50] = 5

        evaluator = Panoptica_Evaluator(
            expected_input=InputType.SEMANTIC,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=NaiveThresholdMatching(),
            segmentation_class_groups=SegmentationClassGroups({"organ": (5, True)}),
        )

        result, debug_data = evaluator.evaluate(b, a)["organ"]
        print(result)
        self.assertEqual(result.tp, 1)
        self.assertEqual(result.fp, 0)
        self.assertEqual(result.sq, 0.0)
        self.assertEqual(result.pq, 0.0)
        self.assertEqual(result.global_bin_dsc, 0.0)
