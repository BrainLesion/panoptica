# Call 'python -m unittest' on this folder
import math
import unittest

import numpy as np

from benchmark.data import (
    SyntheticCase,
    SyntheticDataProvider,
    default_unit_test_cases,
)

from panoptica import InputType, Panoptica_Evaluator
from panoptica.instance_approximator import ConnectedComponentsInstanceApproximator
from panoptica.instance_matcher import NaiveThresholdMatching
from panoptica.metrics import Metric
from panoptica.utils.citation_reminder import disable_citation_reminder
from panoptica.utils.speed_toggles import PanopticaSpeedToggles


EXPECTED_PHASE_KEYS = {
    "preprocess",
    "approximation",
    "matching",
    "instance_evaluation",
    "global_metrics",
}


def _build_evaluator(
    speed_toggles: PanopticaSpeedToggles | None = None,
    save_group_times: bool = False,
) -> Panoptica_Evaluator:
    return Panoptica_Evaluator(
        expected_input=InputType.SEMANTIC,
        instance_approximator=ConnectedComponentsInstanceApproximator(),
        instance_matcher=NaiveThresholdMatching(
            matching_metric=Metric.IOU, matching_threshold=0.3
        ),
        instance_metrics=[Metric.DSC, Metric.IOU],
        global_metrics=[Metric.DSC],
        verbose=False,
        save_group_times=save_group_times,
        speed_toggles=speed_toggles,
    )


def _as_binary(pred: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (pred > 0).astype(np.uint8), (ref > 0).astype(np.uint8)


class Test_PhaseTiming(unittest.TestCase):
    def setUp(self) -> None:
        disable_citation_reminder()
        # Swap this provider for a real DataProvider to run the same asserts on
        # real segmentation data — the rest of the test doesn't change.
        self.provider = SyntheticDataProvider(configs=default_unit_test_cases())
        return super().setUp()

    def test_phase_times_populated(self):
        evaluator = _build_evaluator()
        for name, pred, ref in self.provider.cases():
            with self.subTest(case=name):
                pred_bin, ref_bin = _as_binary(pred, ref)
                result_grouped = evaluator.evaluate(pred_bin, ref_bin)
                result = next(iter(result_grouped.values()))

                self.assertIsNotNone(result.phase_times)
                assert result.phase_times is not None  # mypy
                keys = set(result.phase_times.keys())

                # Every core phase must be present.
                missing = EXPECTED_PHASE_KEYS - keys
                self.assertFalse(
                    missing, f"missing phase keys for {name}: {missing}"
                )

                # At least one per-metric key was recorded during global_metrics.
                self.assertTrue(
                    any(k.startswith("metric_") for k in keys),
                    f"no metric_* entries recorded for {name}",
                )

                for phase, seconds in result.phase_times.items():
                    self.assertIsInstance(seconds, float, f"{name}::{phase}")
                    self.assertTrue(
                        math.isfinite(seconds),
                        f"{name}::{phase} not finite: {seconds}",
                    )
                    self.assertGreaterEqual(
                        seconds, 0.0, f"{name}::{phase} negative: {seconds}"
                    )

                self.assertGreater(
                    sum(result.phase_times.values()), 0.0, f"{name} sum was zero"
                )

    def test_phase_times_sum_bounded_by_computation_time(self):
        evaluator = _build_evaluator(save_group_times=True)
        for name, pred, ref in self.provider.cases():
            with self.subTest(case=name):
                pred_bin, ref_bin = _as_binary(pred, ref)
                result = next(iter(evaluator.evaluate(pred_bin, ref_bin).values()))
                self.assertIsNotNone(result.phase_times)
                assert result.phase_times is not None

                # The metric_* entries are already accounted for inside the
                # global_metrics roll-up — exclude them to avoid double counting.
                real_phases = {
                    k: v
                    for k, v in result.phase_times.items()
                    if not k.startswith("metric_") and k != "preprocess"
                }
                phases_sum = sum(real_phases.values())
                self.assertIsNotNone(result.computation_time)
                assert result.computation_time is not None
                # Loose bound: phases must sit inside the group-level window
                # allowing a generous margin for measurement noise.
                self.assertLessEqual(
                    phases_sum,
                    result.computation_time * 1.5 + 0.01,
                    f"{name}: phase-sum {phases_sum:.4f}s exceeded 1.5x "
                    f"computation_time={result.computation_time:.4f}s",
                )

    def test_speed_toggles_produce_same_result(self):
        # A single mid-size case is enough — the point is behavioral equivalence,
        # not statistical significance.
        case = SyntheticCase("toggle-equivalence", (64, 64), 8, seed=1)
        pred, ref = case.build()
        pred_bin, ref_bin = _as_binary(pred, ref)

        toggle_variants: dict[str, PanopticaSpeedToggles] = {
            "default": PanopticaSpeedToggles(),
            "no_crop": PanopticaSpeedToggles(crop_at_start=False),
            "no_bbox": PanopticaSpeedToggles(precompute_instance_bboxes=False),
            "parallel": PanopticaSpeedToggles(
                parallel_instance_eval=True, parallel_instance_eval_workers=2
            ),
        }

        results: dict[str, dict[str, float]] = {}
        for variant_name, toggles in toggle_variants.items():
            evaluator = _build_evaluator(speed_toggles=toggles)
            result = next(iter(evaluator.evaluate(pred_bin, ref_bin).values()))
            results[variant_name] = {
                "global_bin_dsc": float(result.global_bin_dsc),
                "tp": int(result.tp),
                "n_pred_instances": int(result.n_pred_instances),
                "n_ref_instances": int(result.n_ref_instances),
            }

        baseline = results["default"]
        for variant_name, values in results.items():
            if variant_name == "default":
                continue
            with self.subTest(variant=variant_name):
                self.assertTrue(
                    math.isclose(
                        values["global_bin_dsc"],
                        baseline["global_bin_dsc"],
                        rel_tol=1e-9,
                        abs_tol=1e-9,
                    ),
                    f"{variant_name}: global_bin_dsc drifted "
                    f"({values['global_bin_dsc']} vs {baseline['global_bin_dsc']})",
                )
                self.assertEqual(values["tp"], baseline["tp"], variant_name)
                self.assertEqual(
                    values["n_pred_instances"],
                    baseline["n_pred_instances"],
                    variant_name,
                )
                self.assertEqual(
                    values["n_ref_instances"],
                    baseline["n_ref_instances"],
                    variant_name,
                )


if __name__ == "__main__":
    unittest.main()
