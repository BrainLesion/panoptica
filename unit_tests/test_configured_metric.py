"""Tests for ConfiguredMetric and the unified metrics=[...] evaluator API (#181)."""

import os
import tempfile
import unittest

import numpy as np

from panoptica import (
    ConfiguredMetric,
    GlobalMetric,
    InputType,
    InstanceMetric,
    Metric,
    Panoptica_Evaluator,
)
from panoptica.instance_approximator import ConnectedComponentsInstanceApproximator
from panoptica.instance_matcher import NaiveThresholdMatching


class Test_ConfiguredMetric(unittest.TestCase):
    def test_factories_equivalent(self):
        self.assertEqual(Metric.IOU.instance(), InstanceMetric(Metric.IOU))
        self.assertEqual(Metric.DSC.as_global(), GlobalMetric(Metric.DSC))
        self.assertEqual(
            Metric.NSD.instance(threshold=4),
            ConfiguredMetric(Metric.NSD, "instance", {"threshold": 4}),
        )

    def test_mode_and_accessors(self):
        cm = Metric.IOU.instance()
        self.assertTrue(cm.is_instance)
        self.assertFalse(cm.is_global)
        self.assertEqual(cm.mode, "instance")
        self.assertEqual(cm.metric, Metric.IOU)
        self.assertEqual(cm.name, "IOU")
        gm = GlobalMetric(Metric.DSC)
        self.assertTrue(gm.is_global)

    def test_invalid_mode_rejected(self):
        with self.assertRaises(ValueError):
            ConfiguredMetric(Metric.DSC, "regional")

    def test_unknown_param_rejected(self):
        # DSC declares no params.
        with self.assertRaises(ValueError):
            Metric.DSC.instance(threshold=4)
        # NSD accepts threshold but not foo.
        with self.assertRaises(ValueError):
            Metric.NSD.instance(foo=1)

    def test_hash_and_equality_identity(self):
        a = Metric.NSD.instance(threshold=4)
        b = Metric.NSD.instance(threshold=4)
        c = Metric.NSD.instance(threshold=2)
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))
        self.assertNotEqual(a, c)
        # usable as dict keys / set members
        self.assertEqual(len({a, b, c}), 2)

    def test_result_key_unchanged_without_params(self):
        # PR2: result keys match the underlying metric (parameter suffixes come in PR3).
        self.assertEqual(Metric.NSD.instance().get_result_key("sq"), "sq_nsd")
        self.assertEqual(Metric.IOU.instance().get_result_key("sq"), "sq")

    def test_call_forwards_params(self):
        # NSD with an explicit large threshold should differ from the default.
        ref = np.zeros((20, 20), dtype=np.uint8)
        pred = np.zeros((20, 20), dtype=np.uint8)
        ref[5:15, 5:15] = 1
        pred[6:15, 5:15] = 1  # slight surface mismatch
        default = Metric.NSD(ref, pred, voxelspacing=(1.0, 1.0))
        loose = Metric.NSD.instance(threshold=10)(ref, pred, voxelspacing=(1.0, 1.0))
        self.assertGreaterEqual(loose, default)
        self.assertAlmostEqual(loose, 1.0)


class Test_Unified_Metrics_API(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"

    def _evaluator(self, metrics):
        return Panoptica_Evaluator(
            expected_input=InputType.SEMANTIC,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=NaiveThresholdMatching(),
            metrics=metrics,
        )

    def test_bare_metric_expands_to_both_modes(self):
        ev = self._evaluator([Metric.DSC])
        self.assertEqual(ev._Panoptica_Evaluator__eval_metrics, [Metric.DSC])
        self.assertEqual(ev._Panoptica_Evaluator__global_metrics, [Metric.DSC])

    def test_configured_contributes_single_mode(self):
        ev = self._evaluator([Metric.IOU.instance(), GlobalMetric(Metric.DSC)])
        self.assertEqual(ev._Panoptica_Evaluator__eval_metrics, [Metric.IOU])
        self.assertEqual(ev._Panoptica_Evaluator__global_metrics, [Metric.DSC])

    def test_default_metrics(self):
        ev = self._evaluator(None)
        self.assertEqual(
            ev._Panoptica_Evaluator__eval_metrics,
            [Metric.DSC, Metric.IOU, Metric.ASSD, Metric.RVD],
        )
        self.assertEqual(ev._Panoptica_Evaluator__global_metrics, [Metric.DSC])

    def test_bare_single_object_metric_is_instance_only(self):
        # ASSD/CEDI/HD/HD95 have supports_semantic=False: a bare metric expands to its
        # instance variant only in a non-region evaluator (no whole-image global).
        ev = self._evaluator([Metric.ASSD])
        self.assertEqual(ev._Panoptica_Evaluator__eval_metrics, [Metric.ASSD])
        self.assertEqual(ev._Panoptica_Evaluator__global_metrics, [])

    def test_bare_single_object_metric_keeps_global_region_wise(self):
        # Under per_region_evaluation each region is one object, so the whole-image
        # ("global") variant is meaningful again and a bare single-object metric
        # expands to it (surfaced as region_avg_*).
        ev = Panoptica_Evaluator(
            expected_input=InputType.SEMANTIC,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=NaiveThresholdMatching(),
            metrics=[Metric.ASSD],
            per_region_evaluation=True,
        )
        self.assertIn(Metric.ASSD, ev._Panoptica_Evaluator__global_metrics)

    def test_invalid_entry_type_rejected(self):
        with self.assertRaises(TypeError):
            self._evaluator(["DSC"])

    def test_decision_metric_accepts_configured(self):
        ev = Panoptica_Evaluator(
            expected_input=InputType.SEMANTIC,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=NaiveThresholdMatching(),
            metrics=[Metric.DSC],
            decision_metric=Metric.DSC.instance(),
            decision_threshold=0.5,
        )
        self.assertEqual(ev._Panoptica_Evaluator__decision_metric, Metric.DSC)

    def test_instance_params_flow_through(self):
        # A large NSD threshold tolerates a small surface mismatch, so it scores higher
        # than the default threshold -> proves the parameter reaches the computation.
        ref = np.zeros((40, 40), dtype=np.uint8)
        pred = np.zeros((40, 40), dtype=np.uint8)
        ref[10:30, 10:30] = 1
        pred[10:30, 12:32] = 1  # shifted by 2 voxels

        def sq_nsd(metrics):
            ev = Panoptica_Evaluator(
                expected_input=InputType.MATCHED_INSTANCE, metrics=metrics
            )
            return ev.evaluate(pred, ref)["ungrouped"].sq_nsd

        default = sq_nsd([Metric.NSD.instance()])
        loose = sq_nsd([Metric.NSD.instance(threshold=10)])
        self.assertGreater(loose, default)
        self.assertAlmostEqual(loose, 1.0)

    def test_global_params_rejected_for_now(self):
        with self.assertRaises(NotImplementedError):
            self._evaluator([GlobalMetric(Metric.NSD, threshold=4)])

    def test_conflicting_instance_params_rejected(self):
        with self.assertRaises(ValueError):
            self._evaluator(
                [Metric.NSD.instance(threshold=2), Metric.NSD.instance(threshold=4)]
            )

    def test_yaml_round_trip(self):
        ev = self._evaluator(
            [
                Metric.DSC.instance(),
                Metric.NSD.instance(threshold=4),
                GlobalMetric(Metric.DSC),
            ]
        )
        path = tempfile.mktemp(suffix=".yaml")
        try:
            ev.save_to_config(path)
            ev2 = Panoptica_Evaluator.load_from_config(path)
        finally:
            if os.path.exists(path):
                os.remove(path)
        self.assertEqual(
            ev2._Panoptica_Evaluator__metrics,
            ev._Panoptica_Evaluator__metrics,
        )


if __name__ == "__main__":
    unittest.main()
