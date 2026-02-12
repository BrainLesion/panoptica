# Call 'python -m unittest' on this folder
# coverage run -m unittest
# coverage report
# coverage html
import os
import unittest
import numpy as np

from panoptica.metrics import (
    Metric,
    Evaluation_List_Metric,
    MetricMode,
    MetricCouldNotBeComputedException,
)
from panoptica.utils.edge_case_handling import (
    EdgeCaseResult,
    EdgeCaseHandler,
    MetricZeroTPEdgeCaseHandling,
)
from panoptica import InputType


class Test_EdgeCaseHandler(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"
        return super().setUp()

    def test_edgecasehandler_simple(self):
        handler = EdgeCaseHandler()

        print()
        # print(handler.get_metric_zero_tp_handle(ListMetric.IOU))
        r = handler.handle_zero_tp(
            Metric.IOU, tp=0, num_pred_instances=1, num_ref_instances=1
        )
        print(r)

        iou_test = MetricZeroTPEdgeCaseHandling(
            no_instances_result=EdgeCaseResult.NAN,
            default_result=EdgeCaseResult.ZERO,
        )
        # print(iou_test)
        t = iou_test(tp=0, num_pred_instances=1, num_ref_instances=1)
        print(t)

        # iou_test = default_iou
        # print(iou_test)
        # t = iou_test(tp=0, num_pred_instances=1, num_ref_instances=1)
        # print(t)


class Test_Enums(unittest.TestCase):
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


class Test_ProcessingPair(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"
        return super().setUp()

    def test_semanticpair(self):
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

            for it in InputType:
                # SemanticPair accepts everything
                # For Unmatched and MatchedInstancePair, the numpys must be uints!
                it(a, b)

                c = -a
                d = -b

                if c.min() < 0:
                    with self.assertRaises(AssertionError):
                        it(c, b)
                if d.min() < 0:
                    with self.assertRaises(AssertionError):
                        it(a, d)
                if c.min() < 0 or d.min() < 0:
                    with self.assertRaises(AssertionError):
                        it(c, d)
