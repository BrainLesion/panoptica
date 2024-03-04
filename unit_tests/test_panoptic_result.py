# Call 'python -m unittest' on this folder
# coverage run -m unittest
# coverage report
# coverage html
import os
import unittest

import numpy as np

from panoptica.metrics import Metric
from panoptica.panoptic_result import MetricCouldNotBeComputedException, PanopticaResult
from panoptica.utils.edge_case_handling import EdgeCaseHandler, EdgeCaseResult


class Test_Panoptic_Evaluator(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"
        return super().setUp()

    def test_simple_evaluation(self):
        c = PanopticaResult(
            prediction_arr=None,
            reference_arr=None,
            num_ref_instances=2,
            num_pred_instances=5,
            tp=0,
            list_metrics={Metric.IOU: []},
            edge_case_handler=EdgeCaseHandler(),
        )
        c.calculate_all(print_errors=True)
        print(c)

        self.assertEqual(c.tp, 0)
        self.assertEqual(c.fp, 5)
        self.assertEqual(c.fn, 2)
        self.assertEqual(c.rq, 0.0)
        self.assertEqual(c.pq, 0.0)

    def test_simple_tp_fp(self):
        for n_ref in range(1, 10):
            for n_pred in range(1, 10):
                for tp in range(1, n_ref):
                    c = PanopticaResult(
                        prediction_arr=None,
                        reference_arr=None,
                        num_ref_instances=n_ref,
                        num_pred_instances=n_pred,
                        tp=tp,
                        list_metrics={Metric.IOU: []},
                        edge_case_handler=EdgeCaseHandler(),
                    )
                    c.calculate_all(print_errors=False)
                    print(c)

                    self.assertEqual(c.tp, tp)
                    self.assertEqual(c.fp, n_pred - tp)
                    self.assertEqual(c.fn, n_ref - tp)

    def test_std_edge_case(self):
        for ecr in EdgeCaseResult:
            c = PanopticaResult(
                prediction_arr=None,
                reference_arr=None,
                num_ref_instances=2,
                num_pred_instances=5,
                tp=0,
                list_metrics={Metric.IOU: []},
                edge_case_handler=EdgeCaseHandler(empty_list_std=ecr),
            )
            c.calculate_all(print_errors=True)
            print(c)

            if c.sq_std is None:
                self.assertTrue(ecr.value is None)
            elif np.isnan(c.sq_std):
                self.assertTrue(np.isnan(ecr.value))
            else:
                self.assertEqual(c.sq_std, ecr.value)

    def test_existing_metrics(self):
        from itertools import chain, combinations

        def powerset(iterable):
            s = list(iterable)
            return list(
                chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
            )

        power_set = powerset([Metric.DSC, Metric.IOU, Metric.ASSD])
        for m in power_set[1:]:
            list_metrics: dict = {}
            for me in m:
                list_metrics[me] = [1.0]
            print(list(list_metrics.keys()))

            c = PanopticaResult(
                prediction_arr=None,
                reference_arr=None,
                num_ref_instances=2,
                num_pred_instances=5,
                tp=1,
                list_metrics=list_metrics,
                edge_case_handler=EdgeCaseHandler(),
            )
            c.calculate_all(print_errors=True)
            print(c)

            if Metric.DSC in list_metrics:
                self.assertEqual(c.sq_dsc, 1.0)
                self.assertEqual(c.sq_dsc_std, 0.0)
            else:
                with self.assertRaises(MetricCouldNotBeComputedException):
                    c.sq_dsc
                with self.assertRaises(MetricCouldNotBeComputedException):
                    c.sq_dsc_std
            #
            if Metric.IOU in list_metrics:
                self.assertEqual(c.sq, 1.0)
                self.assertEqual(c.sq_std, 0.0)
            else:
                with self.assertRaises(MetricCouldNotBeComputedException):
                    c.sq
                with self.assertRaises(MetricCouldNotBeComputedException):
                    c.sq_std
            #
            if Metric.ASSD in list_metrics:
                self.assertEqual(c.sq_assd, 1.0)
                self.assertEqual(c.sq_assd_std, 0.0)
            else:
                with self.assertRaises(MetricCouldNotBeComputedException):
                    c.sq_assd
                with self.assertRaises(MetricCouldNotBeComputedException):
                    c.sq_assd_std
