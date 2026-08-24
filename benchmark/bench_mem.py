"""Peak-RSS microbenchmarks for panoptica's compute-heavy paths.

Mirrors :mod:`benchmark.bench_eval` but reports peak resident-set-size growth
(in MiB) instead of wall-clock time. Same synthetic cases, same measurement
points — so a PR's speed and memory reports line up row-for-row.

Usage::

    python benchmark/bench_mem.py            # default 3D + 2D configs
    python benchmark/bench_mem.py --quick    # smaller, faster sizes
    python benchmark/bench_mem.py --quick --json mem_head.json

How peak RSS is measured
------------------------
For each sample we ``os.fork()`` a child, take a fresh RSS baseline in the
child, run the function once, and read the child's peak RSS via a background
sampler thread on ``/proc/self/status`` (``VmRSS``) plus a final
``resource.getrusage(RUSAGE_SELF).ru_maxrss`` check. The child prints
``peak - baseline`` in MiB and exits. Because the child is a *fresh* address
space each time (COW pages from the parent don't grow the delta), every
sample is independent — Python's non-shrinking heap in the parent can't
squash iteration 2..N to zero.

On non-Linux hosts (no ``/proc``) the sampler falls back to periodic
``getrusage`` reads, which is fine inside a short-lived child. Fork itself
is POSIX-only; on Windows the benchmark falls back to in-process sampling
in the parent (and inherits its bias — flagged in the JSON header via
``isolation: "in-process"``).

JSON schema
-----------
Structurally the same as bench_eval's output but with ``measurements_mb``
in place of ``measurements_ms``. Each key still maps to
``{min, median, p90, mean, stddev, n}`` (from :func:`benchmark.stats.summarize`)
so :mod:`benchmark.compare_mem` can run the same Welch's t-test on it.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import os
import subprocess
import sys
import threading
import time
from typing import Any
from collections.abc import Callable

import numpy as np

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
from benchmark.stats import summarize

DEFAULT_REPEATS = 5
DEFAULT_WARMUP = 1
SAMPLER_INTERVAL_S = 0.005

_EVAL_ACCEPTS_TOGGLES = (
    "speed_toggles" in inspect.signature(Panoptica_Evaluator.__init__).parameters
)


# --------------------------------------------------------------------------- #
# Peak-RSS sampling
# --------------------------------------------------------------------------- #
def _read_rss_bytes_proc() -> int | None:
    """Return current process RSS in bytes via /proc/self/status, or None."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    # "VmRSS:\t   12345 kB"
                    parts = line.split()
                    return int(parts[1]) * 1024
    except OSError:
        return None
    return None


def _read_rss_bytes_rusage() -> int:
    """ru_maxrss is a highwater mark. Linux reports it in kB, macOS in bytes."""
    import resource

    val = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return val * 1024 if sys.platform.startswith("linux") else val


_HAS_PROC = _read_rss_bytes_proc() is not None


def _read_rss_bytes() -> int:
    if _HAS_PROC:
        v = _read_rss_bytes_proc()
        if v is not None:
            return v
    return _read_rss_bytes_rusage()


class _PeakSampler:
    """Background thread that tracks peak RSS between start() and stop()."""

    def __init__(self, interval_s: float = SAMPLER_INTERVAL_S) -> None:
        self._interval = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak = 0
        self.baseline = 0

    def _run(self) -> None:
        while not self._stop.is_set():
            v = _read_rss_bytes()
            if v > self.peak:
                self.peak = v
            # Sleep in small chunks so stop() is responsive.
            self._stop.wait(self._interval)

    def start(self) -> None:
        gc.collect()
        self.baseline = _read_rss_bytes()
        self.peak = self.baseline
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> int:
        """Return peak observed RSS (bytes)."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        # Read one more time in case the last delta happened between polls.
        v = _read_rss_bytes()
        if v > self.peak:
            self.peak = v
        return self.peak


def _peak_mib_in_process(fn: Callable[[], Any]) -> float:
    """Peak RSS growth (MiB) measured *in this process*. Only meaningful for
    the very first call after a fresh baseline — see :func:`_peak_mib_forked`
    for the correct per-sample isolation."""
    sampler = _PeakSampler()
    sampler.start()
    try:
        fn()
    finally:
        peak = sampler.stop()
    delta_bytes = max(peak - sampler.baseline, 0)
    return delta_bytes / (1024.0 * 1024.0)


_CAN_FORK = hasattr(os, "fork")


def _peak_mib_forked(fn: Callable[[], Any]) -> float:
    """Peak RSS growth (MiB) for one call to ``fn``, measured inside a fork().

    Each call gets a fresh child address space, so the "how much did this call
    grow the process?" number is truthful even on the 5th repeat — Python's
    non-shrinking heap can't hide new allocations. The child prints one
    float (MiB) to a pipe and exits; the parent reads and reaps it.
    """
    if not _CAN_FORK:
        return _peak_mib_in_process(fn)

    r_fd, w_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        # Child: measure and print, then _exit (skip parent's atexit handlers).
        try:
            os.close(r_fd)
            gc.collect()
            sampler = _PeakSampler()
            sampler.start()
            try:
                fn()
            finally:
                peak = sampler.stop()
            # Also consult ru_maxrss — cheap belt-and-braces for a peak the
            # 5 ms sampler might have missed.
            try:
                import resource

                rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                if sys.platform.startswith("linux"):
                    rss_bytes *= 1024
                if rss_bytes > peak:
                    peak = rss_bytes
            except Exception:
                pass
            delta_mib = max(peak - sampler.baseline, 0) / (1024.0 * 1024.0)
            os.write(w_fd, f"{delta_mib}\n".encode())
        except BaseException as exc:  # noqa: BLE001 — surface *anything* to parent
            try:
                os.write(w_fd, f"ERR {type(exc).__name__}: {exc}\n".encode())
            except Exception:
                pass
        finally:
            try:
                os.close(w_fd)
            except Exception:
                pass
            os._exit(0)

    # Parent
    os.close(w_fd)
    chunks: list[bytes] = []
    with os.fdopen(r_fd, "rb") as f:
        while True:
            b = f.read(4096)
            if not b:
                break
            chunks.append(b)
    _, status = os.waitpid(pid, 0)
    raw = b"".join(chunks).decode().strip()
    if not raw or raw.startswith("ERR "):
        raise RuntimeError(
            f"forked mem measurement failed (exit_status={status}, output={raw!r})"
        )
    return float(raw)


def measure_peak(
    fn: Callable[[], Any],
    repeats: int = DEFAULT_REPEATS,
    warmup: int = DEFAULT_WARMUP,
) -> dict[str, float]:
    """Return ``summarize`` stats over ``repeats`` peak-RSS-delta samples (MiB).

    Each sample runs in a fresh fork so per-call peaks aren't masked by
    Python's non-shrinking heap. Warmup runs also fork — the parent stays
    untouched between measurements, which keeps the CoW baseline small.
    """
    for _ in range(warmup):
        _peak_mib_forked(fn)
    samples: list[float] = []
    for _ in range(repeats):
        samples.append(_peak_mib_forked(fn))
    return summarize(samples)


# --------------------------------------------------------------------------- #
# Evaluator plumbing (mirrors bench_eval for row-by-row alignment)
# --------------------------------------------------------------------------- #
def _largest_instance_masks(
    ref: np.ndarray, pred: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    labels, counts = np.unique(ref[ref > 0], return_counts=True)
    label = int(labels[int(np.argmax(counts))])
    return ref == label, pred == label


def _build_evaluator(speed_toggles: Any = None) -> Panoptica_Evaluator:
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
    repeats: int = DEFAULT_REPEATS,
    warmup: int = DEFAULT_WARMUP,
) -> dict[str, dict[str, float]]:
    ref_labels = tuple(int(x) for x in np.unique(ref) if x > 0)
    ref_mask, pred_mask = _largest_instance_masks(ref, pred)
    ref_bin = (ref > 0).astype(np.uint8)
    pred_bin = (pred > 0).astype(np.uint8)

    evaluator = _build_evaluator()

    measurements: dict[str, dict[str, float]] = {}
    measurements["end_to_end"] = measure_peak(
        lambda: evaluator.evaluate(pred_bin, ref_bin), repeats=repeats, warmup=warmup
    )
    measurements["matching_iou_all_pairs"] = measure_peak(
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

    measurements["surface_unshared"] = measure_peak(
        surface_unshared, repeats=repeats, warmup=warmup
    )
    measurements["surface_shared"] = measure_peak(
        surface_shared, repeats=repeats, warmup=warmup
    )

    n_ref = int(np.max(ref)) if ref.size else 0
    measurements["voronoi_regions"] = measure_peak(
        lambda: _get_voronoi_regions(ref, n_ref), repeats=repeats, warmup=warmup
    )

    return measurements


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _print_case(
    name: str,
    shape: tuple[int, ...],
    n_instances: int,
    mb: dict[str, dict[str, float]],
) -> None:
    print(f"\n### {name}  shape={shape}  instances={n_instances}")
    for key, stats in mb.items():
        spread = stats["p90"] - stats["min"]
        print(
            f"{key:32s} median {stats['median']:8.2f} MiB  "
            f"(min {stats['min']:6.2f}, p90 {stats['p90']:6.2f}, spread {spread:5.2f})"
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


def run_case(
    case: SyntheticCase,
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
        "measurements_mb": measurements,
    }
    _print_case(case.name, case.shape, n_actual, measurements)
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="smaller, faster sizes")
    parser.add_argument(
        "--json",
        dest="json_path",
        default=None,
        help="Also emit machine-readable JSON to this path (for benchmark/compare_mem.py).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help=f"Sampled iterations per measurement (default {DEFAULT_REPEATS}).",
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
        "sampler": "proc-status" if _HAS_PROC else "getrusage",
        "isolation": "fork" if _CAN_FORK else "in-process",
        "cases": [],
    }
    if not _CAN_FORK:
        print(
            "warning: os.fork() unavailable; falling back to in-process sampling "
            "(only the first iteration per measurement is a reliable peak — "
            "subsequent iterations will be squashed to ~0 by Python's "
            "non-shrinking heap).",
            file=sys.stderr,
        )
    # Prime any lazy imports once before the first measurement so the first
    # fork()'s CoW baseline reflects a fully-warm parent.
    _ = _build_evaluator()

    for case in cases:
        doc["cases"].append(
            run_case(case, repeats=args.repeats, warmup=args.warmup)
        )

    if args.json_path:
        with open(args.json_path, "w") as f:
            json.dump(doc, f, indent=2, sort_keys=True)
        print(f"\nWrote {args.json_path}")


if __name__ == "__main__":
    main()
