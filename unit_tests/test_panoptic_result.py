# Call 'python -m unittest' on this folder
# coverage run -m unittest
# coverage report
# coverage html
import os
import unittest

import numpy as np

from panoptica.metrics import Metric
from panoptica.panoptica_result import (
    MetricCouldNotBeComputedException,
    PanopticaResult,
)
from panoptica.utils.edge_case_handling import EdgeCaseHandler, EdgeCaseResult


class Test_Panoptica_Results(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"
        return super().setUp()

    def test_simple_evaluation(self):
        c = PanopticaResult(
            prediction_arr=None,
            reference_arr=None,
            n_ref_instances=2,
            n_pred_instances=5,
            tp=0,
            list_metrics={Metric.IOU: []},
            global_metrics=[Metric.DSC],
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
                        n_ref_instances=n_ref,
                        n_pred_instances=n_pred,
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
                n_ref_instances=2,
                n_pred_instances=5,
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

        metrics_list = [m for m in Metric]
        power_set = powerset(metrics_list)  # [Metric.DSC, Metric.IOU, Metric.ASSD])
        for m in power_set[1:]:
            list_metrics: dict = {}
            for me in m:
                list_metrics[me] = [1.0]
            print(list(list_metrics.keys()))

            c = PanopticaResult(
                prediction_arr=None,
                reference_arr=None,
                n_ref_instances=2,
                n_pred_instances=5,
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
            #
            if Metric.RVD in list_metrics:
                self.assertEqual(c.sq_rvd, 1.0)
                self.assertEqual(c.sq_rvd_std, 0.0)
            else:
                with self.assertRaises(MetricCouldNotBeComputedException):
                    c.sq_rvd
                with self.assertRaises(MetricCouldNotBeComputedException):
                    c.sq_rvd_std
            #
            if Metric.RVAE in list_metrics:
                self.assertEqual(c.sq_rvae, 1.0)
                self.assertEqual(c.sq_rvae_std, 0.0)
            else:
                with self.assertRaises(MetricCouldNotBeComputedException):
                    c.sq_rvae
                with self.assertRaises(MetricCouldNotBeComputedException):
                    c.sq_rvae_std

    def test_to_dict_individual_instances(self):
        result = PanopticaResult(
            prediction_arr=None,
            reference_arr=None,
            n_ref_instances=2,
            n_pred_instances=2,
            tp=2,
            list_metrics={Metric.IOU: [0.8, 0.9], Metric.DSC: [0.85, 0.95]},
            edge_case_handler=EdgeCaseHandler(),
        )
        result.calculate_all(print_errors=False)

        result_dicts = result.to_dict(output_individual_instance_metrics=True)

        self.assertIsInstance(result_dicts, list)
        self.assertEqual(len(result_dicts), 3)

        master_dict = result_dicts[0]
        inst_0_dict = result_dicts[1]
        inst_1_dict = result_dicts[2]

        self.assertIn("tp", master_dict)
        self.assertIn("sq", master_dict)
        self.assertIn("sq_std", master_dict)
        self.assertEqual(master_dict["tp"], 2)

        self.assertIn("sq", inst_0_dict)
        self.assertIn("sq_dsc", inst_0_dict)
        self.assertNotIn("tp", inst_0_dict)
        self.assertNotIn("sq_std", inst_0_dict)
        self.assertNotIn("fp", inst_0_dict)

        self.assertEqual(inst_0_dict["sq"], 0.8)
        self.assertEqual(inst_1_dict["sq"], 0.9)
        self.assertEqual(inst_0_dict["sq_dsc"], 0.85)
        self.assertEqual(inst_1_dict["sq_dsc"], 0.95)

    def test_n_matched_preds_per_instance(self):
        # num_preds_per_match aligns with per-instance order in list_metrics:
        # 1 pred merged into instance 0, 3 preds merged into instance 1.
        result = PanopticaResult(
            prediction_arr=None,
            reference_arr=None,
            n_ref_instances=2,
            n_pred_instances=4,
            tp=2,
            list_metrics={Metric.IOU: [0.8, 0.9]},
            edge_case_handler=EdgeCaseHandler(),
            num_preds_per_match=[1, 3],
        )
        result.calculate_all(print_errors=False)

        result_dicts = result.to_dict(output_individual_instance_metrics=True)
        master_dict, inst_0_dict, inst_1_dict = result_dicts

        # Master row holds the average count across matched refs (mirrors
        # instance_volume_ref); per-instance rows hold individual counts.
        self.assertIn("n_matched_preds", master_dict)
        self.assertAlmostEqual(master_dict["n_matched_preds"], 2.0)

        self.assertEqual(inst_0_dict["n_matched_preds"], 1)
        self.assertEqual(inst_1_dict["n_matched_preds"], 3)

    def test_n_matched_preds_absent_when_not_provided(self):
        # If the matcher path didn't supply num_preds_per_match (e.g., direct
        # construction without pipeline), per-instance dicts should not include
        # the key.
        result = PanopticaResult(
            prediction_arr=None,
            reference_arr=None,
            n_ref_instances=1,
            n_pred_instances=1,
            tp=1,
            list_metrics={Metric.IOU: [0.5]},
            edge_case_handler=EdgeCaseHandler(),
        )
        result.calculate_all(print_errors=False)

        _, inst_0_dict = result.to_dict(output_individual_instance_metrics=True)
        self.assertNotIn("n_matched_preds", inst_0_dict)
