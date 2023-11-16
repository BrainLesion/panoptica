# Call 'python -m unittest' on this folder
# coverage run -m unittest
# coverage report
# coverage html
import unittest

from panoptica.instance_approximator import ConnectedComponentsInstanceApproximator, CCABackend
from panoptica.instance_matcher import NaiveThresholdMatching
from panoptica.instance_evaluator import evaluate_matched_instance
import numpy as np
from panoptica.result import PanopticaResult
from panoptica.utils.datatypes import SemanticPair, UnmatchedInstancePair, MatchedInstancePair, _ProcessingPair
from panoptica.evaluator import Panoptic_Evaluator


class Test_Panoptic_Evaluator(unittest.TestCase):
    def test_simple_evaluation(self):
        a = np.zeros([50, 50], dtype=np.uint16)
        b = a.copy().astype(a.dtype)
        a[20:40, 10:20] = 1
        b[20:35, 10:20] = 2

        sample = SemanticPair(b, a)

        evaluator = Panoptic_Evaluator(
            expected_input=SemanticPair,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=NaiveThresholdMatching(),
        )

        result, debug_data = evaluator.evaluate(sample)
        print(result)
        self.assertEqual(result.tp, 1)
        self.assertEqual(result.fp, 0)
        self.assertEqual(result.sq, 0.75)
        self.assertEqual(result.pq, 0.75)

    def test_pred_empty(self):
        a = np.zeros([50, 50], np.uint16)
        b = a.copy()
        a[20:40, 10:20] = 1
        # b[20:35, 10:20] = 2

        sample = SemanticPair(b, a)

        evaluator = Panoptic_Evaluator(
            expected_input=SemanticPair,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=NaiveThresholdMatching(),
        )

        result, debug_data = evaluator.evaluate(sample)
        print(result)
        self.assertEqual(result.tp, 0)
        self.assertEqual(result.fp, 0)
        self.assertEqual(result.fn, 1)
        self.assertEqual(result.sq, 0.0)
        self.assertEqual(result.pq, 0.0)
        self.assertEqual(result.instance_assd, np.inf)

    def test_ref_empty(self):
        a = np.zeros([50, 50], np.uint16)
        b = a.copy()
        # a[20:40, 10:20] = 1
        b[20:35, 10:20] = 2

        sample = SemanticPair(b, a)

        evaluator = Panoptic_Evaluator(
            expected_input=SemanticPair,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=NaiveThresholdMatching(),
        )

        result, debug_data = evaluator.evaluate(sample)
        print(result)
        self.assertEqual(result.tp, 0)
        self.assertEqual(result.fp, 1)
        self.assertEqual(result.fn, 0)
        self.assertEqual(result.sq, 0.0)
        self.assertEqual(result.pq, 0.0)
        self.assertEqual(result.instance_assd, np.inf)

    def test_both_empty(self):
        a = np.zeros([50, 50], np.uint16)
        b = a.copy()
        # a[20:40, 10:20] = 1
        # b[20:35, 10:20] = 2

        sample = SemanticPair(b, a)

        evaluator = Panoptic_Evaluator(
            expected_input=SemanticPair,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=NaiveThresholdMatching(),
        )

        result, debug_data = evaluator.evaluate(sample)
        print(result)
        self.assertEqual(result.tp, 0)
        self.assertEqual(result.fp, 0)
        self.assertEqual(result.fn, 0)
        self.assertTrue(np.isnan(result.sq))
        self.assertTrue(np.isnan(result.pq))
        self.assertTrue(np.isnan(result.instance_assd))

    def test_dtype_evaluation(self):
        ddtypes = [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]
        dtype_combinations = [(a, b) for a in ddtypes for b in ddtypes]
        for da, db in dtype_combinations:
            a = np.zeros([50, 50], dtype=da)
            b = a.copy().astype(db)
            a[20:40, 10:20] = 1
            b[20:35, 10:20] = 2

            if da != db:
                self.assertRaises(AssertionError, SemanticPair, b, a)
            else:
                sample = SemanticPair(b, a)

                evaluator = Panoptic_Evaluator(
                    expected_input=SemanticPair,
                    instance_approximator=ConnectedComponentsInstanceApproximator(),
                    instance_matcher=NaiveThresholdMatching(),
                )

                result, debug_data = evaluator.evaluate(sample)
                print(result)
                self.assertEqual(result.tp, 1)
                self.assertEqual(result.fp, 0)
                self.assertEqual(result.sq, 0.75)
                self.assertEqual(result.pq, 0.75)
