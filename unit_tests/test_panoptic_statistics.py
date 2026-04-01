# Call 'python -m unittest' on this folder
# coverage run -m unittest
# coverage report
# coverage html
import os
import unittest

import numpy as np

from panoptica import InputType, Panoptica_Aggregator, Panoptica_Statistic, ValueSummary
from panoptica.instance_approximator import ConnectedComponentsInstanceApproximator
from panoptica.instance_matcher import MaximizeMergeMatching, NaiveThresholdMatching
from panoptica.metrics import Metric
from panoptica.panoptica_evaluator import Panoptica_Evaluator
from panoptica.panoptica_result import MetricCouldNotBeComputedException
from panoptica.utils.processing_pair import SemanticPair
from panoptica.utils.segmentation_class import SegmentationClassGroups
import sys
from pathlib import Path


output_test_dir = Path(__file__).parent.joinpath("unittest_tmp_file.tsv")

input_test_file = Path(__file__).parent.joinpath("test_unittest_file.tsv")


class Test_Panoptica_Statistics(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"
        return super().setUp()

    def test_simple_statistic(self):
        a = np.zeros([50, 50], dtype=np.uint16)
        b = a.copy().astype(a.dtype)
        a[20:40, 10:20] = 1
        b[20:35, 10:20] = 2

        evaluator = Panoptica_Evaluator(
            expected_input=InputType.SEMANTIC,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=NaiveThresholdMatching(),
        )

        aggregator = Panoptica_Aggregator(evaluator, output_file=output_test_dir)

        aggregator.evaluate(b, a, "test")

        statistic_obj = Panoptica_Statistic.from_file(output_test_dir)

        statistic_obj.print_summary()

        self.assertEqual(statistic_obj.get("ungrouped", "tp"), [1.0])
        self.assertEqual(statistic_obj.get("ungrouped", "sq"), [0.75])
        self.assertEqual(statistic_obj.get("ungrouped", "sq_rvd"), [-0.25])

        tp_values = statistic_obj.get("ungrouped", "tp")
        sq_values = statistic_obj.get("ungrouped", "sq")
        self.assertEqual(ValueSummary(tp_values).avg, 1.0)
        self.assertEqual(ValueSummary(sq_values).avg, 0.75)

        os.remove(str(output_test_dir))

    def test_multiple_samples_statistic(self):
        a = np.zeros([50, 50], dtype=np.uint16)
        b = a.copy().astype(a.dtype)
        c = a.copy().astype(a.dtype)
        a[20:40, 10:20] = 1
        b[20:35, 10:20] = 2
        c[20:40, 10:20] = 5
        c[0:10, 0:10] = 3

        evaluator = Panoptica_Evaluator(
            expected_input=InputType.SEMANTIC,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=NaiveThresholdMatching(),
        )

        aggregator = Panoptica_Aggregator(evaluator, output_file=output_test_dir)

        aggregator.evaluate(b, a, "test")
        aggregator.evaluate(a, c, "test2")

        statistic_obj = Panoptica_Statistic.from_file(output_test_dir)

        statistic_obj.print_summary()

        self.assertEqual(ValueSummary(statistic_obj.get("ungrouped", "tp")).avg, 1.0)
        self.assertEqual(ValueSummary(statistic_obj.get("ungrouped", "sq")).avg, 0.875)
        self.assertEqual(ValueSummary(statistic_obj.get("ungrouped", "fn")).avg, 0.5)
        self.assertEqual(ValueSummary(statistic_obj.get("ungrouped", "rec")).avg, 0.75)
        self.assertEqual(ValueSummary(statistic_obj.get("ungrouped", "rec")).std, 0.25)

        os.remove(str(output_test_dir))

    def test_statistics_from_file(self):
        statistic_obj = Panoptica_Statistic.from_file(input_test_file)
        #
        test2 = statistic_obj.get_one_subject("test2")  # get one subject
        print()
        print("test2", test2)
        self.assertEqual(test2["ungrouped"]["num_ref_instances"], 2)

        all_num_ref_instances = statistic_obj.get_across_groups("num_ref_instances")
        print()
        print("all_num_ref_instances", all_num_ref_instances)
        self.assertEqual(len(all_num_ref_instances), 2)
        self.assertEqual(sum(all_num_ref_instances), 3)

        groupwise_summary = statistic_obj.get_summary_across_groups()
        print()
        print(groupwise_summary)
        self.assertEqual(groupwise_summary["num_ref_instances"].avg, 1.5)
