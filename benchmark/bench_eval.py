"""Microbenchmarks for panoptica's compute-heavy paths.

Run from the repository root::

    python benchmark/bench_eval.py            # default 3D + 2D configs
    python benchmark/bench_eval.py --quick    # smaller, faster sizes

The harness builds deterministic synthetic multi-instance volumes (seeded, so
"before"/"after" runs are comparable) and times four things:

* end-to-end ``Panoptica_Evaluator.evaluate`` on a semantic input,
* instance matching (``_calc_matching_metric_of_overlapping_labels`` with IoU),
* the surface-distance metric family (ASSD / HD / HD95 / NSD) on one instance,
* Voronoi region assignment (``_get_voronoi_regions``).

Each is the target of a Phase 1 speed change; compare the printed timings before
and after a change to quantify the win.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

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


# --------------------------------------------------------------------------- #
# Synthetic data
# --------------------------------------------------------------------------- #
def _ball(radius: int, ndim: int) -> np.ndarray:
    """Boolean (2*radius+1)**ndim ball mask."""
    line = np.arange(-radius, radius + 1)
    grids = np.meshgrid(*([line] * ndim), indexing="ij")
    dist2 = sum(g.astype(np.int64) ** 2 for g in grids)
    return dist2 <= radius * radius


def make_reference(
    shape: tuple[int, ...],
    n_instances: int,
    rng: np.random.Generator,
    rmin: int = 3,
    rmax: int = 7,
) -> tuple[np.ndarray, int]:
    """Place up to ``n_instances`` non-overlapping balls with distinct labels."""
    arr = np.zeros(shape, dtype=np.int32)
    ndim = len(shape)
    label = 0
    attempts = 0
    while label < n_instances and attempts < n_instances * 25:
        attempts += 1
        r = int(rng.integers(rmin, rmax + 1))
        center = [int(rng.integers(r + 1, s - r - 1)) for s in shape]
        ball = _ball(r, ndim)
        slices = tuple(slice(c - r, c + r + 1) for c in center)
        region = arr[slices]
        if np.any(region[ball] != 0):
            continue  # keep instances separate
        label += 1
        region[ball] = label
    return arr, label


def make_prediction(
    ref: np.ndarray, rng: np.random.Generator, shift: int = 1, drop_frac: float = 0.1
) -> np.ndarray:
    """Perturb the reference into a plausible prediction (shift + dropped instances)."""
    pred = ref.copy()
    for ax in range(ref.ndim):
        pred = np.roll(pred, int(rng.integers(-shift, shift + 1)), axis=ax)
    labels = [int(x) for x in np.unique(pred) if x > 0]
    n_drop = int(len(labels) * drop_frac)
    if n_drop:
        for label in rng.choice(labels, size=n_drop, replace=False):
            pred[pred == label] = 0
    return pred


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #
def timeit(fn, repeats: int = 3) -> float:
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


def run_config(
    name: str, shape: tuple[int, ...], n_instances: int, seed: int = 0
) -> None:
    rng = np.random.default_rng(seed)
    ref, n = make_reference(shape, n_instances, rng)
    pred = make_prediction(ref, rng)
    ref_labels = tuple(int(x) for x in np.unique(ref) if x > 0)
    ref_mask, pred_mask = _largest_instance_masks(ref, pred)

    evaluator = Panoptica_Evaluator(
        expected_input=InputType.SEMANTIC,
        instance_approximator=ConnectedComponentsInstanceApproximator(),
        instance_matcher=NaiveThresholdMatching(
            matching_metric=Metric.IOU, matching_threshold=0.3
        ),
        metrics=[
            Metric.DSC,  # bare -> instance + global
            Metric.IOU.instance(),
            Metric.ASSD.instance(),
            Metric.HD95.instance(),
            Metric.NSD.instance(),
        ],
        verbose=False,
    )
    ref_bin = (ref > 0).astype(np.uint8)
    pred_bin = (pred > 0).astype(np.uint8)

    print(f"\n### {name}  shape={shape}  instances={n}")
    print(
        f"{'end-to-end evaluate':32s} {timeit(lambda: evaluator.evaluate(pred_bin, ref_bin)):8.1f} ms"
    )
    print(
        f"{'matching (IoU, all pairs)':32s} "
        f"{timeit(lambda: _calc_matching_metric_of_overlapping_labels(pred, ref, ref_labels, Metric.IOU)):8.1f} ms"
    )

    def surface_unshared():
        # what computing each metric standalone costs (recomputes the transforms)
        _average_symmetric_surface_distance(ref_mask, pred_mask)
        _compute_hausdorff_distance(ref_mask, pred_mask)
        _compute_hausdorff_distance95(ref_mask, pred_mask)
        _compute_normalized_surface_dice(ref_mask, pred_mask)

    def surface_shared():
        # what _evaluate_instance now does: one pair, reduced four ways
        sd_ref, sd_pred = _surface_distance_pair(ref_mask, pred_mask)
        _assd_from_pair(sd_ref, sd_pred)
        _hd_from_pair(sd_ref, sd_pred)
        _hd95_from_pair(sd_ref, sd_pred)
        _nsd_from_pair(sd_ref, sd_pred)

    print(f"{'surface family (unshared)':32s} {timeit(surface_unshared):8.1f} ms")
    print(f"{'surface family (shared)':32s} {timeit(surface_shared):8.1f} ms")
    print(
        f"{'voronoi regions':32s} {timeit(lambda: _get_voronoi_regions(ref, n)):8.1f} ms"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="smaller, faster sizes")
    args = parser.parse_args()

    if args.quick:
        run_config("2D quick", (256, 256), 40)
        run_config("3D quick", (96, 96, 96), 30)
    else:
        run_config("2D many-instance", (512, 512), 200)
        run_config("3D medium", (160, 160, 160), 120)


if __name__ == "__main__":
    main()
