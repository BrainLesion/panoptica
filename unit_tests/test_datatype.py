# Call 'python -m unittest' on this folder
# coverage run -m unittest
# coverage report
# coverage html
import os
import unittest

import numpy as np

from panoptica.instance_approximator import ConnectedComponentsInstanceApproximator
from panoptica.instance_matcher import MaximizeMergeMatching, NaiveThresholdMatching
from panoptica.metrics import Metric, MetricMode, _Metric
from panoptica.panoptic_evaluator import Panoptic_Evaluator
from panoptica.panoptic_result import MetricCouldNotBeComputedException, PanopticaResult
from panoptica.utils.edge_case_handling import EdgeCaseHandler, EdgeCaseResult
from panoptica.utils.processing_pair import SemanticPair


class Test_Panoptic_Evaluator(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"
        return super().setUp()

    def test_metrics_enum(self):
        print(Metric.DSC)
        # print(MatchingMetric.DSC.name)

        self.assertEqual(Metric.DSC, Metric.DSC)
        self.assertEqual(Metric.DSC, "DSC")
        self.assertEqual(Metric.DSC.name, "DSC")
        #
        self.assertNotEqual(Metric.DSC, Metric.IOU)
        self.assertNotEqual(Metric.DSC, "IOU")

    def test_matching_metric(self):
        dsc_metric = Metric.DSC

        self.assertTrue(dsc_metric.score_beats_threshold(0.55, 0.5))
        self.assertFalse(dsc_metric.score_beats_threshold(0.5, 0.55))

        assd_metric = Metric.ASSD

        self.assertFalse(assd_metric.score_beats_threshold(0.55, 0.5))
        self.assertTrue(assd_metric.score_beats_threshold(0.5, 0.55))

    # TODO listmetric + Mode (STD and so on)
