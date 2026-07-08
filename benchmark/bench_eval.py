"""Microbenchmarks for panoptica's compute-heavy paths.

Run from the repository root::

    python benchmark/bench_eval.py            # default 3D + 2D configs
    python benchmark/bench_eval.py --quick    # smaller, faster sizes
    python benchmark/bench_eval.py --quick --json out.json
    python benchmark/bench_eval.py --quick --toggle-comparison

Each configuration times these things:

* end-to-end ``Panoptica_Evaluator.evaluate`` on a semantic input,
* instance matching (``_calc_matching_metric_of_overlapping_labels`` with IoU),
* the surface-distance metric family (ASSD / HD / HD95 / NSD) on one instance,
* Voronoi region assignment (``_get_voronoi_regions``),
* per-phase durations from :class:`~panoptica.utils.phase_timer.PhaseTimer`
  (``phase_*`` and ``metric_*`` measurements from the last evaluator run of each case).

``--json`` emits a machine-readable document for :mod:`benchmark.compare`.
``--toggle-comparison`` also runs each :class:`~panoptica.utils.speed_toggles.PanopticaSpeedToggles`
lever flipped from its default so a developer can see the per-toggle win.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any, Callable

import numpy as np

# Import the panoptica copy that lives next to this script, not whatever editable
# install happens to be on sys.path (a plain ``python benchmark/bench_eval.py`` puts
# the script dir, not the cwd, on sys.path[0]).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from panoptica import InputType, Panoptica_Evaluator
from panoptica.instance_approximator import ConnectedComponentsInstanceApproximator
from panoptica.instance_matcher import NaiveThresholdMatching
from panoptica.metrics import Metric
from panoptica._functionals import (
    _calc_matching_metric_of_overlapping_labels,
    _get_voronoi_regions,
)
from panoptica.metrics.assd import _average_symmetric_surface_distance
from panoptica.metrics.hausdorff_distance import (
    _compute_hausdorff_distance,
    _compute_hausdorff_distance95,
)
from panoptica.metrics.normalized_surface_dice import _compute_normalized_surface_dice
from panoptica.metrics._surface_distances import (
    _surface_distance_pair,
    _assd_from_pair,
    _hd_from_pair,
    _hd95_from_pair,
    _nsd_from_pair,
)
from panoptica.utils.speed_toggles import PanopticaSpeedToggles

from benchmark.data import (
    SyntheticCase,
    default_benchmark_cases,
)


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #
def timeit(fn: Callable[[], Any], repeats: int = 3) -> float:
    """Return the best (min) wall time over ``repeats`` runs, in milliseconds."""
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


def _largest_instance_masks(
    ref: np.ndarray, pred: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    labels, counts = np.unique(ref[ref > 0], return_counts=True)
    label = int(labels[int(np.argmax(counts))])
    return ref == label, pred == label


def _build_evaluator(speed_toggles: PanopticaSpeedToggles | None = None) -> Panoptica_Evaluator:
    return Panoptica_Evaluator(
        expected_input=InputType.SEMANTIC,
        instance_approximator=ConnectedComponentsInstanceApproximator(),
        instance_matcher=NaiveThresholdMatching(
            matching_metric=Metric.IOU, matching_threshold=0.3
        ),
        instance_metrics=[Metric.DSC, Metric.IOU, Metric.ASSD, Metric.HD95, Metric.NSD],
        global_metrics=[Metric.DSC],
        verbose=False,
        speed_toggles=speed_toggles,
    )


def _measure_case(
    ref: np.ndarray,
    pred: np.ndarray,
    speed_toggles: PanopticaSpeedToggles | None = None,
) -> dict[str, float]:
    """Return the ms measurements dict for a single (ref, pred) case."""
    ref_labels = tuple(int(x) for x in np.unique(ref) if x > 0)
    ref_mask, pred_mask = _largest_instance_masks(ref, pred)
    ref_bin = (ref > 0).astype(np.uint8)
    pred_bin = (pred > 0).astype(np.uint8)

    evaluator = _build_evaluator(speed_toggles)

    measurements: dict[str, float] = {}
    measurements["end_to_end"] = timeit(
        lambda: evaluator.evaluate(pred_bin, ref_bin)
    )
    measurements["matching_iou_all_pairs"] = timeit(
        lambda: _calc_matching_metric_of_overlapping_labels(
            pred, ref, ref_labels, Metric.IOU
        )
    )

    def surface_unshared():
        _average_symmetric_surface_distance(ref_mask, pred_mask)
        _compute_hausdorff_distance(ref_mask, pred_mask)
        _compute_hausdorff_distance95(ref_mask, pred_mask)
        _compute_normalized_surface_dice(ref_mask, pred_mask)

    def surface_shared():
        sd_ref, sd_pred = _surface_distance_pair(ref_mask, pred_mask)
        _assd_from_pair(sd_ref, sd_pred)
        _hd_from_pair(sd_ref, sd_pred)
        _hd95_from_pair(sd_ref, sd_pred)
        _nsd_from_pair(sd_ref, sd_pred)

    measurements["surface_unshared"] = timeit(surface_unshared)
    measurements["surface_shared"] = timeit(surface_shared)

    n_ref = int(np.max(ref)) if ref.size else 0
    measurements["voronoi_regions"] = timeit(lambda: _get_voronoi_regions(ref, n_ref))

    # Pull per-phase and per-metric times from the last evaluator run.
    last_result = evaluator.evaluate(pred_bin, ref_bin)["ungrouped"]
    if last_result.phase_times:
        for name, seconds in last_result.phase_times.items():
            if name.startswith("metric_"):
                key = name
            else:
                key = f"phase_{name}"
            measurements[key] = seconds * 1e3

    return measurements


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _print_case(name: str, shape: tuple[int, ...], n_instances: int, ms: dict[str, float]) -> None:
    print(f"\n### {name}  shape={shape}  instances={n_instances}")
    for key, val in ms.items():
        print(f"{key:32s} {val:8.1f} ms")


def _git_commit_short() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------- #
# Case runner
# --------------------------------------------------------------------------- #
def _toggle_variants() -> dict[str, PanopticaSpeedToggles]:
    """Named variants for --toggle-comparison. Each flips a single lever."""
    return {
        "no_crop": PanopticaSpeedToggles(crop_at_start=False),
        "no_bbox": PanopticaSpeedToggles(precompute_instance_bboxes=False),
        "parallel": PanopticaSpeedToggles(parallel_instance_eval=True),
    }


def run_case(
    case: SyntheticCase,
    toggle_comparison: bool = False,
) -> dict[str, Any]:
    pred, ref = case.build()
    n_actual = int(np.max(ref)) if ref.size else 0
    measurements = _measure_case(ref, pred)

    entry: dict[str, Any] = {
        "name": case.name,
        "shape": list(case.shape),
        "instances": n_actual,
        "measurements_ms": measurements,
    }
    _print_case(case.name, case.shape, n_actual, measurements)

    if toggle_comparison:
        toggles_block: dict[str, dict[str, float]] = {}
        for variant_name, toggles in _toggle_variants().items():
            toggle_measurements = _measure_case(ref, pred, speed_toggles=toggles)
            toggles_block[variant_name] = toggle_measurements
            print(f"\n--- {case.name} [{variant_name}]")
            for key, val in toggle_measurements.items():
                print(f"{key:32s} {val:8.1f} ms")
        entry["toggles"] = toggles_block

    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="smaller, faster sizes")
    parser.add_argument(
        "--json",
        dest="json_path",
        default=None,
        help="Also emit machine-readable JSON to this path (for benchmark/compare.py).",
    )
    parser.add_argument(
        "--toggle-comparison",
        action="store_true",
        help="Also run each speed toggle flipped from its default and record the timings.",
    )
    args = parser.parse_args()

    cases = default_benchmark_cases(quick=args.quick)

    doc: dict[str, Any] = {
        "python": ".".join(str(x) for x in sys.version_info[:3]),
        "commit": _git_commit_short(),
        "cases": [],
    }
    for case in cases:
        doc["cases"].append(
            run_case(case, toggle_comparison=args.toggle_comparison)
        )

    if args.json_path:
        with open(args.json_path, "w") as f:
            json.dump(doc, f, indent=2, sort_keys=True)
        print(f"\nWrote {args.json_path}")


if __name__ == "__main__":
    main()
