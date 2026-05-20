# Call 'python -m unittest' on this folder
# coverage run -m unittest
# coverage report
# coverage html
import os
import unittest

import numpy as np

from panoptica import UnmatchedInstancePair
from panoptica.metrics import (
    Metric,
)
from panoptica.instance_approximator import ConnectedComponentsInstanceApproximator
from panoptica.instance_matcher import (
    MaximizeMergeMatching,
    NaiveThresholdMatching,
    MaxBipartiteMatching,
    InstanceLabelMap,
    MatchingContext,
)
from panoptica.metrics import Metric
from panoptica.instance_evaluator import (
    evaluate_matched_instance,
    _evaluate_instance,
)
from panoptica.utils.processing_pair import MatchedInstancePair
from unit_tests.unit_test_utils import (
    case_simple_identical,
    case_simple_nooverlap,
    case_simple_overpredicted,
    case_simple_shifted,
    case_simple_underpredicted,
    case_simple_overlap_but_large_discrepancy,
    case_multiple_overlapping_instances,
)
import sys
from pathlib import Path


class Test_Panoptica_Instance_Approximation(unittest.TestCase):
    # TODO
    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"
        return super().setUp()


class Test_Panoptica_InstanceLabelMap(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"
        return super().setUp()

    def test_labelmap(self):
        labelmap = InstanceLabelMap()

        labelmap.add_labelmap_entry(1, 1)
        labelmap.add_labelmap_entry([2, 3], 2)
        print(labelmap)

        with self.assertRaises(Exception):
            labelmap.add_labelmap_entry(1, 1)
            labelmap.add_labelmap_entry(1, 2)

        self.assertTrue(labelmap.contains_and(None, None))
        self.assertTrue(labelmap.contains_and(1, 1))
        self.assertTrue(not labelmap.contains_and(1, 3))
        self.assertTrue(not labelmap.contains_and(4, 1))
        print(labelmap)

    def test_instancelabelmap_dict(self):
        for fe in range(1, 10):
            for se in range(-1, 10):
                for re in range(-1, 10):
                    a = InstanceLabelMap()
                    if fe == se:
                        continue
                    p = [fe, se] if se != 0 else [fe]
                    print(f"Testing {p} -> {re}")
                    if se == -1 or re <= 0:
                        with self.assertRaises(Exception):
                            a.add_labelmap_entry(p, re)
                    else:
                        a.add_labelmap_entry(p, re)
                        self.assertTrue(p[0] in a)
                        self.assertTrue(a.contains_pred(p[0]))
                        self.assertTrue(a.contains_ref(re))
                        self.assertTrue(a.contains_and(p[0], re))
                        self.assertTrue(a.contains_or(p[0], re))

    def test_instancelabelmap_readonly(self):
        a = InstanceLabelMap()
        a.add_labelmap_entry([1, 2], 3)
        print(a)

        with self.assertRaises(Exception):
            a.__labelmap[1] = 4
        with self.assertRaises(Exception):
            a.__labelmap[2] = 5
            print(a)
        with self.assertRaises(Exception):
            a[1] = 4
        with self.assertRaises(Exception):
            a[2] = 5

        get_one_to_one_dictionary = a.get_one_to_one_dictionary()
        get_one_to_one_dictionary[2] = 5
        print(a, a.values())
        self.assertTrue(5 not in a.values())


class Test_Panoptica_Instance_Matching(unittest.TestCase):
    def test_naive_threshold_matching_zerofive(self):
        a = NaiveThresholdMatching(matching_metric=Metric.DSC, matching_threshold=0.5)
        context = MatchingContext()

        for c in [
            case_simple_identical,
            case_simple_overpredicted,
            case_simple_shifted,
            case_simple_underpredicted,
        ]:
            print(c.__name__)
            ref, pred = c()
            b = UnmatchedInstancePair(pred, ref)

            labelmap = a._match_instances(b, context=context, matching_threshold=0.5)
            print(labelmap)
            self.assertTrue(labelmap[1] == 1)
            self.assertTrue(len(labelmap) == 1)

        for c in [
            case_simple_nooverlap,
            case_simple_overlap_but_large_discrepancy,
            case_multiple_overlapping_instances,
        ]:
            print(c.__name__)
            ref, pred = c()
            b = UnmatchedInstancePair(pred, ref)

            labelmap = a._match_instances(b, context=context, matching_threshold=0.5)
            print(labelmap)
            self.assertTrue(len(labelmap) == 0)

    def test_naive_threshold_matching_zero(self):
        a = NaiveThresholdMatching(matching_metric=Metric.DSC, matching_threshold=0.0)
        context = MatchingContext()

        for c in [
            case_simple_identical,
            case_simple_overpredicted,
            case_simple_shifted,
            case_simple_underpredicted,
            case_simple_overlap_but_large_discrepancy,
        ]:
            print(c.__name__)
            ref, pred = c()
            b = UnmatchedInstancePair(pred, ref)

            labelmap = a._match_instances(b, context=context, matching_threshold=0.0)
            print(labelmap)
            self.assertTrue(labelmap[1] == 1)
            self.assertTrue(len(labelmap) == 1)

        for c in [
            case_simple_nooverlap,
        ]:
            print(c.__name__)
            ref, pred = c()
            b = UnmatchedInstancePair(pred, ref)

            labelmap = a._match_instances(b, context=context, matching_threshold=0.0)
            print(labelmap)
            self.assertTrue(len(labelmap) == 0)

        for c in [
            case_multiple_overlapping_instances,
        ]:
            print(c.__name__)
            ref, pred = c()
            b = UnmatchedInstancePair(pred, ref)

            labelmap = a._match_instances(b, context=context, matching_threshold=0.0)
            print(labelmap)
            self.assertTrue(len(labelmap) == 1)
            self.assertTrue(labelmap[1] == 2)

    def test_MaxBipartiteMatching_zerofive(self):
        a = MaxBipartiteMatching(matching_metric=Metric.DSC, matching_threshold=0.5)
        context = MatchingContext()

        for c in [
            case_simple_identical,
            case_simple_overpredicted,
            case_simple_shifted,
            case_simple_underpredicted,
        ]:
            print(c.__name__)
            ref, pred = c()
            b = UnmatchedInstancePair(pred, ref)

            labelmap = a._match_instances(b, context=context, matching_threshold=0.5)
            print(labelmap)
            self.assertTrue(labelmap[1] == 1)
            self.assertTrue(len(labelmap) == 1)

        for c in [
            case_simple_nooverlap,
            case_simple_overlap_but_large_discrepancy,
            case_multiple_overlapping_instances,
        ]:
            print(c.__name__)
            ref, pred = c()
            b = UnmatchedInstancePair(pred, ref)

            labelmap = a._match_instances(b, context=context, matching_threshold=0.5)
            print(labelmap)
            self.assertTrue(len(labelmap) == 0)

    def test_MaxBipartiteMatching_zero(self):
        a = MaxBipartiteMatching(matching_metric=Metric.DSC, matching_threshold=0.0)
        context = MatchingContext()

        for c in [
            case_simple_identical,
            case_simple_overpredicted,
            case_simple_shifted,
            case_simple_underpredicted,
            case_simple_overlap_but_large_discrepancy,
        ]:
            print(c.__name__)
            ref, pred = c()
            b = UnmatchedInstancePair(pred, ref)

            labelmap = a._match_instances(b, context=context, matching_threshold=0.0)
            print(labelmap)
            self.assertTrue(labelmap[1] == 1)
            self.assertTrue(len(labelmap) == 1)

        for c in [
            case_simple_nooverlap,
        ]:
            print(c.__name__)
            ref, pred = c()
            b = UnmatchedInstancePair(pred, ref)

            labelmap = a._match_instances(b, context=context, matching_threshold=0.0)
            print(labelmap)
            self.assertTrue(len(labelmap) == 0)

        for c in [
            case_multiple_overlapping_instances,
        ]:
            print(c.__name__)
            ref, pred = c()
            b = UnmatchedInstancePair(pred, ref)

            labelmap = a._match_instances(b, context=context, matching_threshold=0.0)
            print(labelmap)
            self.assertTrue(labelmap[1] == 1)
            self.assertTrue(labelmap[2] == 2)
            self.assertTrue(len(labelmap) == 2)


class Test_Panoptica_Instance_Evaluation(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"
        return super().setUp()

    def test_eval_empty(self):
        a = np.zeros([50, 50], dtype=np.uint16)
        b = a.copy().astype(a.dtype)
        # a[20:40, 10:20] = 1
        # b[20:40, 10:20] = 1

        result = _evaluate_instance(a, b, ref_idx=1, eval_metrics=[Metric.DSC]).metrics
        print()
        print(result)
        self.assertEqual(len(result), 0)

    def test_simple_evaluation_nooverlap(self):
        a = np.zeros([50, 50], dtype=np.uint16)
        b = a.copy().astype(a.dtype)
        a[20:40, 10:20] = 1
        b[20:35, 10:20] = 2

        result = _evaluate_instance(a, b, ref_idx=1, eval_metrics=[Metric.DSC]).metrics
        print()
        print(result)
        self.assertEqual(len(result), 0)

    def test_simple_evaluation_wrongidx(self):
        a = np.zeros([50, 50], dtype=np.uint16)
        b = a.copy().astype(a.dtype)
        a[20:40, 10:20] = 1
        b[20:35, 10:20] = 2

        result = _evaluate_instance(a, b, ref_idx=3, eval_metrics=[Metric.DSC]).metrics
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

        eval_result = _evaluate_instance(a, b, ref_idx=1, eval_metrics=[Metric.DSC])
        print()
        print(eval_result)
        self.assertEqual(len(eval_result.metrics), 1)
        self.assertEqual(eval_result.metrics[Metric.DSC], 1.0)
        self.assertEqual(eval_result.voxel_count_ref, int(np.count_nonzero(a == 1)))
        self.assertEqual(eval_result.volume_ref, float(eval_result.voxel_count_ref))

    def test_decision_threshold_rejection_reported_as_unmatched(self):
        # Reference instance 1 and prediction instance 1 overlap at IoU ~0.33 below the 0.5 decision_threshold. 
        # The match must be reported as unmatched (FN) rather than silently dropped.
        prediction_arr = np.zeros([4, 4], dtype=np.uint16)
        reference_arr = np.zeros([4, 4], dtype=np.uint16)
        prediction_arr[:, 0:2] = 1
        reference_arr[:, 1:3] = 1
        pair = MatchedInstancePair(
            prediction_arr=prediction_arr,
            reference_arr=reference_arr,
            matched_instances=[1],
            missed_reference_labels=[],
            missed_prediction_labels=[],
        )

        result = evaluate_matched_instance(
            pair,
            eval_metrics=[Metric.IOU],
            decision_metric=Metric.IOU,
            decision_threshold=0.5,
        )

        ref_voxel_count = int(np.count_nonzero(reference_arr == 1))
        self.assertEqual(result.tp, 0)
        self.assertEqual(len(result.instance_voxel_count_matched_ref), 0)
        self.assertEqual(len(result.instance_voxel_count_unmatched_ref), 1)
        self.assertEqual(
            result.instance_voxel_count_unmatched_ref[0], ref_voxel_count
        )
        self.assertEqual(
            result.instance_volume_unmatched_ref[0], float(ref_voxel_count)
        )
        self.assertEqual(result.list_metrics[Metric.IOU], [])

    def test_no_overlap_instance_reported_as_unmatched(self):
        # Matcher paired ref label 1 with a prediction label that has no voxels: `_evaluate_instance` returns the empty-metrics default. 
        # The ref must still be reported as unmatched (FN), with its true voxel count.
        prediction_arr = np.zeros([4, 4], dtype=np.uint16)
        reference_arr = np.zeros([4, 4], dtype=np.uint16)
        reference_arr[1:3, 1:3] = 1
        pair = MatchedInstancePair(
            prediction_arr=prediction_arr,
            reference_arr=reference_arr,
            matched_instances=[1],
            missed_reference_labels=[],
            missed_prediction_labels=[],
        )

        result = evaluate_matched_instance(
            pair,
            eval_metrics=[Metric.IOU],
            decision_metric=None,
            decision_threshold=None,
        )

        ref_voxel_count = int(np.count_nonzero(reference_arr == 1))
        self.assertEqual(result.tp, 0)
        self.assertEqual(len(result.instance_voxel_count_matched_ref), 0)
        self.assertEqual(len(result.instance_voxel_count_unmatched_ref), 1)
        self.assertEqual(
            result.instance_voxel_count_unmatched_ref[0], ref_voxel_count
        )
        self.assertEqual(
            result.instance_volume_unmatched_ref[0], float(ref_voxel_count)
        )
