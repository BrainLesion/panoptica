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

Each measurement is warmed up ``--warmup`` times (default 1, discarded) and then
sampled ``--repeats`` times (default 7). We report ``{min, median, p90}`` per
measurement — the median is what :mod:`benchmark.compare` gates on and the
``(p90 - min)`` spread lets the gate ignore movement inside the baseline's own
noise band.

``--json`` emits a machine-readable document for :mod:`benchmark.compare`.
``--toggle-comparison`` also runs each :class:`~panoptica.utils.speed_toggles.PanopticaSpeedToggles`
lever flipped from its default so a developer can see the per-toggle win.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import subprocess
import sys
import time
from typing import Any
from collections.abc import Callable

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

# Older panoptica (e.g. main before this PR lands) doesn't have these — the workflow
# runs the same bench script against main's source when producing the baseline, so
# we degrade gracefully rather than crash.
try:
    from panoptica.utils.speed_toggles import (
        PanopticaSpeedToggles as _PanopticaSpeedToggles,
    )
except ImportError:
    _PanopticaSpeedToggles = None  # type: ignore[assignment,misc]

from benchmark.data import (
    SyntheticCase,
    default_benchmark_cases,
)


DEFAULT_REPEATS = 7
DEFAULT_WARMUP = 1

# We probe *both* the module presence and the evaluator signature, because a partial
# swap (module present, evaluator kwarg absent) is a real state we've hit in CI.
_EVAL_ACCEPTS_TOGGLES = (
    "speed_toggles" in inspect.signature(Panoptica_Evaluator.__init__).parameters
)


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #
def timeit(
    fn: Callable[[], Any],
    repeats: int = DEFAULT_REPEATS,
    warmup: int = DEFAULT_WARMUP,
) -> dict[str, float]:
    """Time ``fn`` and return ``{min, median, p90}`` in milliseconds.

    ``warmup`` iterations are discarded (they absorb import/first-touch/JIT costs),
    then ``repeats`` timed iterations are recorded. Median is the primary stat
    reported by :mod:`benchmark.compare`; ``p90 - min`` is used as a spread proxy
    so the gate can distinguish real regressions from noise-band motion.
    """
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    return _summarize(samples)


def _summarize(samples_ms: list[float]) -> dict[str, float]:
    ordered = sorted(samples_ms)
    n = len(ordered)
    median = ordered[n // 2] if n % 2 else 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])
    p90_idx = min(int(0.9 * (n - 1) + 0.5), n - 1)
    return {
        "min": ordered[0],
        "median": median,
        "p90": ordered[p90_idx],
    }


def _largest_instance_masks(
    ref: np.ndarray, pred: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    labels, counts = np.unique(ref[ref > 0], return_counts=True)
    label = int(labels[int(np.argmax(counts))])
    return ref == label, pred == label


def _build_evaluator(speed_toggles: Any = None) -> Panoptica_Evaluator:
    """Build a Panoptica_Evaluator, passing ``speed_toggles`` only when supported."""
    kwargs: dict[str, Any] = dict(
        expected_input=InputType.SEMANTIC,
        instance_approximator=ConnectedComponentsInstanceApproximator(),
        instance_matcher=NaiveThresholdMatching(
            matching_metric=Metric.IOU, matching_threshold=0.3
        ),
        instance_metrics=[Metric.DSC, Metric.IOU, Metric.ASSD, Metric.HD95, Metric.NSD],
        global_metrics=[Metric.DSC],
        verbose=False,
    )
    if _EVAL_ACCEPTS_TOGGLES and _PanopticaSpeedToggles is not None:
        kwargs["speed_toggles"] = speed_toggles
    return Panoptica_Evaluator(**kwargs)


def _measure_case(
    ref: np.ndarray,
    pred: np.ndarray,
    speed_toggles: Any = None,
    repeats: int = DEFAULT_REPEATS,
    warmup: int = DEFAULT_WARMUP,
) -> dict[str, dict[str, float]]:
    """Return per-measurement ``{min, median, p90}`` for a single (ref, pred) case."""
    ref_labels = tuple(int(x) for x in np.unique(ref) if x > 0)
    ref_mask, pred_mask = _largest_instance_masks(ref, pred)
    ref_bin = (ref > 0).astype(np.uint8)
    pred_bin = (pred > 0).astype(np.uint8)

    evaluator = _build_evaluator(speed_toggles)

    measurements: dict[str, dict[str, float]] = {}
    measurements["end_to_end"] = timeit(
        lambda: evaluator.evaluate(pred_bin, ref_bin), repeats=repeats, warmup=warmup
    )
    measurements["matching_iou_all_pairs"] = timeit(
        lambda: _calc_matching_metric_of_overlapping_labels(
            pred, ref, ref_labels, Metric.IOU
        ),
        repeats=repeats,
        warmup=warmup,
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

    measurements["surface_unshared"] = timeit(
        surface_unshared, repeats=repeats, warmup=warmup
    )
    measurements["surface_shared"] = timeit(
        surface_shared, repeats=repeats, warmup=warmup
    )

    n_ref = int(np.max(ref)) if ref.size else 0
    measurements["voronoi_regions"] = timeit(
        lambda: _get_voronoi_regions(ref, n_ref), repeats=repeats, warmup=warmup
    )

    # Phase / metric timings: run the evaluator warmup + repeats times and aggregate
    # per-key samples so phase_* / metric_* also get {min, median, p90}. Skipped when
    # running against a pre-phase-timer version of panoptica.
    for _ in range(warmup):
        evaluator.evaluate(pred_bin, ref_bin)
    phase_samples: dict[str, list[float]] = {}
    for _ in range(repeats):
        result = evaluator.evaluate(pred_bin, ref_bin)["ungrouped"]
        phase_times = getattr(result, "phase_times", None)
        if not phase_times:
            break
        for name, seconds in phase_times.items():
            key = name if name.startswith("metric_") else f"phase_{name}"
            phase_samples.setdefault(key, []).append(seconds * 1e3)
    for key, samples in phase_samples.items():
        if samples:
            measurements[key] = _summarize(samples)

    return measurements


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _print_case(
    name: str,
    shape: tuple[int, ...],
    n_instances: int,
    ms: dict[str, dict[str, float]],
) -> None:
    print(f"\n### {name}  shape={shape}  instances={n_instances}")
    for key, stats in ms.items():
        spread = stats["p90"] - stats["min"]
        print(
            f"{key:32s} median {stats['median']:8.1f} ms  "
            f"(min {stats['min']:6.1f}, p90 {stats['p90']:6.1f}, spread {spread:5.1f})"
        )


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
def _toggle_variants() -> dict[str, Any]:
    """Named variants for --toggle-comparison. Each flips a single lever."""
    if _PanopticaSpeedToggles is None or not _EVAL_ACCEPTS_TOGGLES:
        return {}
    return {
        "no_crop": _PanopticaSpeedToggles(crop_at_start=False),
        "no_bbox": _PanopticaSpeedToggles(precompute_instance_bboxes=False),
        "parallel": _PanopticaSpeedToggles(parallel_instance_eval=True),
    }


def run_case(
    case: SyntheticCase,
    toggle_comparison: bool = False,
    repeats: int = DEFAULT_REPEATS,
    warmup: int = DEFAULT_WARMUP,
) -> dict[str, Any]:
    pred, ref = case.build()
    n_actual = int(np.max(ref)) if ref.size else 0
    measurements = _measure_case(ref, pred, repeats=repeats, warmup=warmup)

    entry: dict[str, Any] = {
        "name": case.name,
        "shape": list(case.shape),
        "instances": n_actual,
        "measurements_ms": measurements,
    }
    _print_case(case.name, case.shape, n_actual, measurements)

    if toggle_comparison:
        variants = _toggle_variants()
        if not variants:
            print(
                "\n(--toggle-comparison skipped: PanopticaSpeedToggles not available "
                "in this panoptica install)"
            )
        else:
            toggles_block: dict[str, dict[str, dict[str, float]]] = {}
            for variant_name, toggles in variants.items():
                toggle_measurements = _measure_case(
                    ref, pred, speed_toggles=toggles, repeats=repeats, warmup=warmup
                )
                toggles_block[variant_name] = toggle_measurements
                print(f"\n--- {case.name} [{variant_name}]")
                for key, stats in toggle_measurements.items():
                    print(
                        f"{key:32s} median {stats['median']:8.1f} ms  "
                        f"(min {stats['min']:6.1f}, p90 {stats['p90']:6.1f})"
                    )
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
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help=f"Timed iterations per measurement (default {DEFAULT_REPEATS}).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help=f"Discarded warmup iterations per measurement (default {DEFAULT_WARMUP}).",
    )
    args = parser.parse_args()

    cases = default_benchmark_cases(quick=args.quick)

    doc: dict[str, Any] = {
        "python": ".".join(str(x) for x in sys.version_info[:3]),
        "commit": _git_commit_short(),
        "repeats": args.repeats,
        "warmup": args.warmup,
        "cases": [],
    }
    for case in cases:
        doc["cases"].append(
            run_case(
                case,
                toggle_comparison=args.toggle_comparison,
                repeats=args.repeats,
                warmup=args.warmup,
            )
        )

    if args.json_path:
        with open(args.json_path, "w") as f:
            json.dump(doc, f, indent=2, sort_keys=True)
        print(f"\nWrote {args.json_path}")


if __name__ == "__main__":
    main()
