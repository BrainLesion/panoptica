# Call 'python -m unittest' on this folder
# coverage run -m unittest
# coverage report
# coverage html
import os
import unittest

from panoptica.metrics import (
    Metric,
    Evaluation_List_Metric,
    MetricMode,
    MetricCouldNotBeComputedException,
)
from panoptica.utils.edge_case_handling import EdgeCaseResult


class Test_Datatypes(unittest.TestCase):
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

    def test_EdgeCaseResult_enum(self):
        for e in EdgeCaseResult:
            self.assertEqual(e, e)

    def test_matching_metric(self):
        dsc_metric = Metric.DSC

        self.assertTrue(dsc_metric.score_beats_threshold(0.55, 0.5))
        self.assertFalse(dsc_metric.score_beats_threshold(0.5, 0.55))

        assd_metric = Metric.ASSD

        self.assertFalse(assd_metric.score_beats_threshold(0.55, 0.5))
        self.assertTrue(assd_metric.score_beats_threshold(0.5, 0.55))

    def test_listmetric(self):
        lmetric = Evaluation_List_Metric(
            name_id="Test",
            empty_list_std=None,
            value_list=[1, 3, 5],
        )

        self.assertEqual(lmetric[MetricMode.ALL], [1, 3, 5])
        self.assertTrue(lmetric[MetricMode.AVG] == 3)
        self.assertTrue(lmetric[MetricMode.SUM] == 9)

    def test_listmetric_edgecase(self):
        lmetric = Evaluation_List_Metric(
            name_id="Test",
            empty_list_std=None,
            value_list=[1, 3, 5],
            is_edge_case=True,
            edge_case_result=50,
        )

        self.assertEqual(lmetric[MetricMode.ALL], [1, 3, 5])
        self.assertTrue(lmetric[MetricMode.AVG] == 50)
        self.assertTrue(lmetric[MetricMode.SUM] == 50)

    def test_listmetric_emptylist(self):
        lmetric = Evaluation_List_Metric(
            name_id="Test",
            empty_list_std=None,
            value_list=None,
            is_edge_case=True,
            edge_case_result=50,
        )

        for mode in MetricMode:
            with self.assertRaises(MetricCouldNotBeComputedException):
                lmetric[mode]
