"""Tests for the grouped result accessors (#248): result["global"].dice etc."""

import os
import unittest

import numpy as np

from panoptica import InputType, Metric, Panoptica_Evaluator
from panoptica.instance_approximator import ConnectedComponentsInstanceApproximator
from panoptica.instance_matcher import NaiveThresholdMatching
from panoptica.panoptica_result import MetricCouldNotBeComputedException


def _instance_result():
    ref = np.zeros((50, 50), dtype=np.uint16)
    pred = np.zeros((50, 50), dtype=np.uint16)
    ref[5:15, 5:15] = 1
    pred[5:15, 5:15] = 1
    ref[20:30, 20:30] = 2
    pred[21:30, 20:30] = 2
    ev = Panoptica_Evaluator(
        expected_input=InputType.SEMANTIC,
        instance_approximator=ConnectedComponentsInstanceApproximator(),
        instance_matcher=NaiveThresholdMatching(),
        metrics=[Metric.DSC, Metric.IOU, Metric.ASSD],
    )
    return ev.evaluate(pred, ref, voxelspacing=(1.0, 1.0))["ungrouped"]


class Test_Grouped_Results(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"

    def test_instance_group_matches_flat(self):
        r = _instance_result()
        self.assertEqual(r.instance.dice.sq, r.sq_dsc)
        self.assertEqual(r.instance.dice.pq, r.pq_dsc)
        self.assertEqual(r.instance.dice.std, r.sq_dsc_std)
        # IOU uses the bare sq/pq keys.
        self.assertEqual(r.instance.iou.sq, r.sq)
        self.assertEqual(r.instance.iou.pq, r.pq)

    def test_semantic_group_matches_flat(self):
        r = _instance_result()
        self.assertEqual(r.semantic.dice, r.global_bin_dsc)
        self.assertEqual(r.semantic.iou, r.global_bin_iou)
        self.assertEqual(r.semantic.assd, r.global_bin_assd)

    def test_subscript_global_alias(self):
        r = _instance_result()
        self.assertEqual(r["global"].dice, r.global_bin_dsc)
        self.assertEqual(r["global"]["iou"], r.global_bin_iou)
        self.assertEqual(r["semantic"].dice, r.semantic.dice)
        self.assertEqual(r["instance"].dice.sq, r.instance.dice.sq)

    def test_distance_metric_has_no_pq(self):
        r = _instance_result()
        with self.assertRaises(AttributeError):
            r.instance.assd.pq

    def test_unknown_metric_and_group(self):
        r = _instance_result()
        with self.assertRaises(AttributeError):
            r.instance.bogus
        with self.assertRaises(KeyError):
            r["bogus"]

    def test_unrequested_metric_still_raises(self):
        # NSD was not requested -> grouped access raises the same exception as flat.
        r = _instance_result()
        with self.assertRaises(MetricCouldNotBeComputedException):
            r.instance.nsd.sq

    def test_region_group_matches_flat(self):
        gt = np.zeros((30, 30, 10), dtype=np.int32)
        pred = np.zeros((30, 30, 10), dtype=np.int32)
        gt[5:15, 5:15, 2:8] = 1
        gt[20:25, 20:25, 2:8] = 2
        pred[6:16, 6:16, 3:9] = 1
        pred[19:24, 19:24, 3:9] = 2
        ev = Panoptica_Evaluator(
            expected_input=InputType.SEMANTIC,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=NaiveThresholdMatching(),
            metrics=[Metric.DSC, Metric.IOU],
            per_region_evaluation=True,
        )
        r = ev.evaluate(pred, gt)["ungrouped"]
        self.assertEqual(r.region.dice, r.region_avg_dsc)
        self.assertEqual(r.region.iou, r.region_avg_iou)

    def test_autc_group_matches_flat(self):
        ref = np.zeros((50, 50), dtype=np.uint16)
        pred = np.zeros((50, 50), dtype=np.uint16)
        ref[10:20, 10:30] = 1
        pred[10:20, 20:40] = 1
        ev = Panoptica_Evaluator(
            expected_input=InputType.SEMANTIC,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=NaiveThresholdMatching(),
        )
        autc = ev.evaluate_autc(pred, ref, threshold_step_size=0.1)["ungrouped"]
        self.assertEqual(autc.autc.dice.pq, autc.autc_pq_dsc)
        self.assertEqual(autc.autc.dice.sq, autc.autc_sq_dsc)
        self.assertEqual(autc.autc.iou.sq, autc.autc_sq)
        self.assertEqual(autc.autc.iou.pq, autc.autc_pq)


if __name__ == "__main__":
    unittest.main()
