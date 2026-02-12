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
from panoptica.panoptica_evaluator import Panoptica_Evaluator, EdgeCaseHandler
from panoptica.panoptica_result import MetricCouldNotBeComputedException
from panoptica.utils.processing_pair import SemanticPair
from panoptica.utils.segmentation_class import SegmentationClassGroups
import sys
from pathlib import Path


class Test_EdgeCase_Handling(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"
        return super().setUp()

    def test_simple_evaluation(self):
        a = np.zeros([50, 50], dtype=np.uint16)
        b = a.copy().astype(a.dtype)
        a[20:40, 10:20] = 1
        b[10:15, 10:20] = 2

        evaluator = Panoptica_Evaluator(
            expected_input=InputType.SEMANTIC,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=NaiveThresholdMatching(),
            edge_case_handler=EdgeCaseHandler(),
        )

        result = evaluator.evaluate(b, a)["ungrouped"]
        print(result)
        # self.assertEqual(result.tp, 1)
        # self.assertEqual(result.fp, 0)
        # self.assertEqual(result.sq, 0.75)
        # self.assertEqual(result.pq, 0.75)
        self.assertEqual(result.global_bin_dsc, 0.0)
        self.assertEqual(result.sq_dsc, 0.0)

    def test_no_instances(self):
        a = np.zeros([50, 50], dtype=np.uint16)
        b = a.copy().astype(a.dtype)
        # a[20:40, 10:20] = 1
        # b[10:15, 10:20] = 2

        evaluator = Panoptica_Evaluator(
            expected_input=InputType.SEMANTIC,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=NaiveThresholdMatching(),
            edge_case_handler=EdgeCaseHandler(),
        )

        result = evaluator.evaluate(b, a)["ungrouped"]
        print(result)
        # self.assertEqual(result.tp, 1)
        # self.assertEqual(result.fp, 0)
        # self.assertEqual(result.sq, 0.75)
        # self.assertEqual(result.pq, 0.75)
        self.assertTrue(np.isnan(result.global_bin_dsc))
        self.assertTrue(np.isnan(result.sq_dsc))
        self.assertTrue(np.isnan(result.rq))
        self.assertTrue(np.isnan(result.sq_assd))
