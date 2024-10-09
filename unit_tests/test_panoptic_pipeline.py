# Call 'python -m unittest' on this folder
# coverage run -m unittest
# coverage report
# coverage html
import os
import unittest

import numpy as np

from panoptica import InputType
from panoptica.instance_approximator import ConnectedComponentsInstanceApproximator
from panoptica.instance_matcher import MaximizeMergeMatching, NaiveThresholdMatching, InstanceLabelMap
from panoptica.metrics import Metric
from panoptica.instance_evaluator import (
    evaluate_matched_instance,
    _evaluate_instance,
    _get_paired_crop,
)
import sys
from pathlib import Path


class Test_Panoptica_Instance_Approximation(unittest.TestCase):
    # TODO
    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"
        return super().setUp()


class Test_Panoptica_Instance_Matching(unittest.TestCase):
    # TODO
    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"
        return super().setUp()

    def test_labelmap(self):
        labelmap = InstanceLabelMap()

        labelmap.add_labelmap_entry(1, 1)
        labelmap.add_labelmap_entry([2, 3], 2)

        with self.assertRaises(Exception):
            labelmap.add_labelmap_entry(1, 1)
            labelmap.add_labelmap_entry(1, 2)

        self.assertTrue(labelmap.contains_and(None, None))
        self.assertTrue(labelmap.contains_and(1, 1))
        self.assertTrue(not labelmap.contains_and(1, 3))
        self.assertTrue(not labelmap.contains_and(4, 1))

        print(labelmap)

        with self.assertRaises(Exception):
            labelmap.labelmap = {}


class Test_Panoptica_Instance_Evaluation(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"
        return super().setUp()

    def test_eval_empty(self):
        a = np.zeros([50, 50], dtype=np.uint16)
        b = a.copy().astype(a.dtype)
        # a[20:40, 10:20] = 1
        # b[20:40, 10:20] = 1

        result = _evaluate_instance(a, b, ref_idx=1, eval_metrics=[Metric.DSC])
        print()
        print(result)
        self.assertEqual(len(result), 0)

    def test_simple_evaluation_nooverlap(self):
        a = np.zeros([50, 50], dtype=np.uint16)
        b = a.copy().astype(a.dtype)
        a[20:40, 10:20] = 1
        b[20:35, 10:20] = 2

        result = _evaluate_instance(a, b, ref_idx=1, eval_metrics=[Metric.DSC])
        print()
        print(result)
        self.assertEqual(len(result), 0)

    def test_simple_evaluation_wrongidx(self):
        a = np.zeros([50, 50], dtype=np.uint16)
        b = a.copy().astype(a.dtype)
        a[20:40, 10:20] = 1
        b[20:35, 10:20] = 2

        result = _evaluate_instance(a, b, ref_idx=3, eval_metrics=[Metric.DSC])
        print()
        print(result)
        self.assertEqual(len(result), 0)
        # self.assertEqual(result.tp, 1)
        # self.assertEqual(result.fp, 0)
        # self.assertEqual(result.sq, 0.75)
        # self.assertEqual(result.pq, 0.75)
        # self.assertAlmostEqual(result.global_bin_dsc, 0.8571428571428571)

    def test_simple_evaluation(self):
        a = np.zeros([50, 50], dtype=np.uint16)
        b = a.copy().astype(a.dtype)
        a[20:40, 10:20] = 1
        b[20:40, 10:20] = 1

        result = _evaluate_instance(a, b, ref_idx=1, eval_metrics=[Metric.DSC])
        print()
        print(result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[Metric.DSC], 1.0)
