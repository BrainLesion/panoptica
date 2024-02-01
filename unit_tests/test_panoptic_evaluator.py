# Call 'python -m unittest' on this folder
# coverage run -m unittest
# coverage report
# coverage html
import os
import unittest

import numpy as np

from panoptica.instance_approximator import ConnectedComponentsInstanceApproximator
from panoptica.instance_matcher import MaximizeMergeMatching, NaiveThresholdMatching
from panoptica.metrics import Metric, _Metric
from panoptica.panoptic_evaluator import Panoptic_Evaluator
from panoptica.panoptic_result import MetricCouldNotBeComputedException, PanopticaResult
from panoptica.utils.processing_pair import SemanticPair


class Test_Panoptic_Evaluator(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"
        return super().setUp()

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

    def test_simple_evaluation_DSC(self):
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

    def test_simple_evaluation_DSC_partial(self):
        a = np.zeros([50, 50], dtype=np.uint16)
        b = a.copy().astype(a.dtype)
        a[20:40, 10:20] = 1
        b[20:35, 10:20] = 2

        sample = SemanticPair(b, a)

        evaluator = Panoptic_Evaluator(
            expected_input=SemanticPair,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=NaiveThresholdMatching(matching_metric=Metric.DSC),
            eval_metrics=[Metric.DSC],
        )

        result, debug_data = evaluator.evaluate(sample)
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

        sample = SemanticPair(b, a)

        evaluator = Panoptic_Evaluator(
            expected_input=SemanticPair,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=NaiveThresholdMatching(
                matching_metric=Metric.ASSD,
                matching_threshold=1.0,
            ),
        )

        result, debug_data = evaluator.evaluate(sample)
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

        sample = SemanticPair(b, a)

        evaluator = Panoptic_Evaluator(
            expected_input=SemanticPair,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=NaiveThresholdMatching(
                matching_metric=Metric.ASSD,
                matching_threshold=0.5,
            ),
        )

        result, debug_data = evaluator.evaluate(sample)
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
        self.assertEqual(result.sq_assd, np.inf)

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
        self.assertEqual(result.sq_assd, np.inf)

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

    def test_simple_evaluation_maximize_matcher(self):
        a = np.zeros([50, 50], dtype=np.uint16)
        b = a.copy().astype(a.dtype)
        a[20:40, 10:20] = 1
        b[20:35, 10:20] = 2

        sample = SemanticPair(b, a)

        evaluator = Panoptic_Evaluator(
            expected_input=SemanticPair,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=MaximizeMergeMatching(),
        )

        result, debug_data = evaluator.evaluate(sample)
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

        sample = SemanticPair(b, a)

        evaluator = Panoptic_Evaluator(
            expected_input=SemanticPair,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=MaximizeMergeMatching(),
        )

        result, debug_data = evaluator.evaluate(sample)
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

        sample = SemanticPair(b, a)

        evaluator = Panoptic_Evaluator(
            expected_input=SemanticPair,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=MaximizeMergeMatching(),
        )

        result, debug_data = evaluator.evaluate(sample)
        print(result)
        self.assertEqual(result.tp, 1)
        self.assertEqual(result.fp, 1)
        self.assertEqual(result.sq, 0.85)
        self.assertAlmostEqual(result.pq, 0.56666666)
        self.assertAlmostEqual(result.rq, 0.66666666)
        self.assertAlmostEqual(result.sq_dsc, 0.9189189189189)
