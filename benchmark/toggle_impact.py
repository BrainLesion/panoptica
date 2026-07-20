"""Local benchmark that quantifies each :class:`PanopticaSpeedToggles` lever's speed impact.

Runs the same seeded synthetic multi-instance volumes as :mod:`benchmark.bench_eval`,
then for every case sweeps the speed toggles and reports the per-toggle Δ from the
default configuration. Useful for answering "which toggle is actually saving me time?"
without touching CI.

**What's timed**: each iteration is a full ``Panoptica_Aggregator.evaluate(...)`` call,
including the per-subject file backend append and the aggregator's lock overhead.
So the reported numbers include real read/write cost, not just raw evaluator time.

**Parallel_aggregator caveat**: the row's ``mean ± stddev`` column shows *per-worker*
wall-clock — what each individual ``evaluate()`` call took *inside a worker*,
including lock waits, CPU contention, and BLAS thread oversubscription. The
``Δ vs default`` column instead uses the *amortized* per-iteration cost
(``batch_wall_ms / repeats``), which is the honest "does the pool save me time
end-to-end?" number. A "Note" block after the batch throughput section makes both
numbers explicit.

**Default mode** is *one-at-a-time* (OAT): baseline is ``PanopticaSpeedToggles()`` and
each variant flips a single lever. There is also an extra ``parallel_aggregator`` variant
that keeps default toggles but submits all ``repeats`` calls through a
``multiprocessing.Pool`` — this is a different parallelism axis from
``parallel_instance_eval`` (which parallelizes the inner per-instance loop within one
sample). Add ``--factorial`` to sweep the full 2×2×2 toggle grid.

**Statistical significance**: each non-default variant is compared to the default via
Welch's t-test on the raw per-iteration samples. A row is only highlighted 🟢 / 🔴
when ``p < --alpha``. Otherwise the Δ % is shown unmarked.

**Sanity check**: for each case, the report prints a cold-start line (the very first
call, before warmup) alongside the steady-state median so the reader can confirm the
number they'd see doing an ad-hoc ``time.perf_counter()`` around a single call.

Runs entirely offline — no GitHub Actions, no PR comment, no JSON round-trip.

Examples
--------

    python benchmark/toggle_impact.py                       # OAT, default sizes
    python benchmark/toggle_impact.py --quick               # smaller sizes
    python benchmark/toggle_impact.py --quick --factorial   # all 2^N toggle combinations
    python benchmark/toggle_impact.py --quick --case "2D quick" --repeats 21
    python benchmark/toggle_impact.py --quick --alpha 0.01  # stricter significance
    python benchmark/toggle_impact.py --quick --workers 8   # parallel_aggregator pool size
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import sys
import time
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, TextIO

import numpy as np
from scipy import stats  # type: ignore[import-untyped]

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from panoptica import InputType, Panoptica_Evaluator
from panoptica.instance_approximator import ConnectedComponentsInstanceApproximator
from panoptica.instance_matcher import NaiveThresholdMatching
from panoptica.metrics import Metric
from panoptica.panoptica_aggregator import Panoptica_Aggregator
from panoptica.utils.speed_toggles import PanopticaSpeedToggles

from benchmark.data import (
    SyntheticCase,
    benchmark_cases,
)

DEFAULT_REPEATS = 50
DEFAULT_WARMUP = 2
DEFAULT_ALPHA = 0.05


# --------------------------------------------------------------------------- #
# Toggle sweeps
# --------------------------------------------------------------------------- #
_BOOLEAN_FIELDS = (
    "crop_at_start",
    "precompute_instance_bboxes",
    "parallel_instance_eval",
)


def _one_at_a_time() -> dict[str, PanopticaSpeedToggles]:
    """Baseline + each single toggle flipped from its default."""
    baseline = PanopticaSpeedToggles()
    variants: dict[str, PanopticaSpeedToggles] = {"default": baseline}
    for field in _BOOLEAN_FIELDS:
        flipped = not getattr(baseline, field)
        variants[f"{field}={flipped}"] = replace(baseline, **{field: flipped})
    return variants


def _full_factorial() -> dict[str, PanopticaSpeedToggles]:
    """All 2^N boolean combinations. ``default`` is first."""
    variants: dict[str, PanopticaSpeedToggles] = {}
    baseline = PanopticaSpeedToggles()
    defaults = {f: getattr(baseline, f) for f in _BOOLEAN_FIELDS}
    for mask in range(1 << len(_BOOLEAN_FIELDS)):
        kwargs = {
            field: bool((mask >> i) & 1) ^ defaults[field]
            for i, field in enumerate(_BOOLEAN_FIELDS)
        }
        name = (
            "default"
            if mask == 0
            else ",".join(
                f"{field}={kwargs[field]}"
                for field in _BOOLEAN_FIELDS
                if kwargs[field] != defaults[field]
            )
        )
        variants[name] = PanopticaSpeedToggles(**kwargs)
    return variants


# --------------------------------------------------------------------------- #
# Evaluator / aggregator plumbing
# --------------------------------------------------------------------------- #
def _build_evaluator(toggles: PanopticaSpeedToggles) -> Panoptica_Evaluator:
    return Panoptica_Evaluator(
        expected_input=InputType.SEMANTIC,
        instance_approximator=ConnectedComponentsInstanceApproximator(),
        instance_matcher=NaiveThresholdMatching(
            matching_metric=Metric.IOU, matching_threshold=0.3
        ),
        instance_metrics=[Metric.DSC, Metric.IOU, Metric.ASSD, Metric.HD95, Metric.NSD],
        global_metrics=[Metric.DSC],
        verbose=False,
        speed_toggles=toggles,
    )


def _make_aggregator(
    toggles: PanopticaSpeedToggles, out_dir: Path, tag: str
) -> Panoptica_Aggregator:
    return Panoptica_Aggregator(
        _build_evaluator(toggles),
        output_file=out_dir / f"agg_{tag}.jsonl",
        log_times=True,
        continue_file=False,
    )


# --------------------------------------------------------------------------- #
# Sequential and parallel measurement
# --------------------------------------------------------------------------- #
def _time_calls(
    aggregator: Panoptica_Aggregator,
    pred_bin: np.ndarray,
    ref_bin: np.ndarray,
    n_calls: int,
    tag: str,
    start_idx: int = 0,
) -> list[float]:
    """Sequentially run ``n_calls`` aggregator evaluates, return per-call ms."""
    samples: list[float] = []
    for i in range(n_calls):
        subject = f"{tag}_{start_idx + i:04d}"
        t0 = time.perf_counter()
        aggregator.evaluate(pred_bin, ref_bin, subject_name=subject)
        samples.append((time.perf_counter() - t0) * 1e3)
    return samples


def _measure_variant_sequential(
    name: str,
    toggles: PanopticaSpeedToggles,
    pred_bin: np.ndarray,
    ref_bin: np.ndarray,
    out_dir: Path,
    repeats: int,
    warmup: int,
) -> dict[str, Any]:
    """Cold-start + warmup + repeats. Returns per-variant timing block."""
    aggregator = _make_aggregator(toggles, out_dir, name.replace(",", "_"))

    # Cold-start: first ever call for this fresh aggregator.
    cold = _time_calls(aggregator, pred_bin, ref_bin, 1, f"{name}_cold")

    # Warmup: discarded to stabilise the caches / lazy imports.
    if warmup > 0:
        _time_calls(aggregator, pred_bin, ref_bin, warmup, f"{name}_warmup")

    # Timed samples.
    t_batch_start = time.perf_counter()
    samples = _time_calls(aggregator, pred_bin, ref_bin, repeats, f"{name}_run")
    batch_wall_ms = (time.perf_counter() - t_batch_start) * 1e3

    # Grab phase_times once (an extra call is cheap and gives phase attribution
    # without polluting the timing samples).
    phase_times = _harvest_phase_times(aggregator, pred_bin, ref_bin)

    return {
        "cold_start_ms": cold[0],
        "samples_ms": samples,
        "batch_wall_ms": batch_wall_ms,
        "phase_times_ms": phase_times,
        "mode": "sequential",
        "workers": 1,
    }


# Module-global state for the pool. On posix fork, children inherit this without
# pickling; only the subject_name (a plain str) crosses the queue. This sidesteps
# the fact that ``Panoptica_Aggregator`` isn't cleanly picklable (it references
# module-level ``multiprocessing.Lock`` instances).
_POOL_STATE: dict[str, Any] = {}


def _pool_worker(subject_name: str) -> float:
    aggregator = _POOL_STATE["aggregator"]
    pred_bin = _POOL_STATE["pred_bin"]
    ref_bin = _POOL_STATE["ref_bin"]
    t0 = time.perf_counter()
    aggregator.evaluate(pred_bin, ref_bin, subject_name=subject_name)
    return (time.perf_counter() - t0) * 1e3


def _measure_variant_parallel_aggregator(
    toggles: PanopticaSpeedToggles,
    pred_bin: np.ndarray,
    ref_bin: np.ndarray,
    out_dir: Path,
    repeats: int,
    warmup: int,
    workers: int,
) -> dict[str, Any]:
    """Run repeats calls concurrently through a ``multiprocessing.Pool``."""
    aggregator = _make_aggregator(toggles, out_dir, "parallel_agg")

    # Cold-start on the aggregator (parent process) — same as sequential.
    cold = _time_calls(aggregator, pred_bin, ref_bin, 1, "parallel_cold")
    # Warmup runs sequentially to prime any shared caches.
    if warmup > 0:
        _time_calls(aggregator, pred_bin, ref_bin, warmup, "parallel_warmup")

    # Populate module globals so forked workers inherit them.
    _POOL_STATE["aggregator"] = aggregator
    _POOL_STATE["pred_bin"] = pred_bin
    _POOL_STATE["ref_bin"] = ref_bin

    subjects = [f"parallel_run_{i:04d}" for i in range(repeats)]
    t_batch_start = time.perf_counter()
    with Pool(processes=workers) as pool:
        samples = pool.map(_pool_worker, subjects)
    batch_wall_ms = (time.perf_counter() - t_batch_start) * 1e3

    # Clear the state so subsequent variants don't accidentally inherit stale refs.
    _POOL_STATE.clear()

    return {
        "cold_start_ms": cold[0],
        "samples_ms": samples,
        "batch_wall_ms": batch_wall_ms,
        # batch_wall_ms / repeats is the honest "how long does one iteration
        # cost end-to-end when the pool is running" — this is what drives the
        # Δ vs default in the main table, not the per-worker samples above.
        "amortized_per_iter_ms": batch_wall_ms / repeats if repeats > 0 else 0.0,
        "phase_times_ms": {},
        "mode": "parallel_aggregator",
        "workers": workers,
    }


def _harvest_phase_times(
    aggregator: Panoptica_Aggregator,
    pred_bin: np.ndarray,
    ref_bin: np.ndarray,
) -> dict[str, float]:
    """Extra evaluator-only call so we can pull ``phase_times`` for attribution."""
    result_grouped = aggregator.panoptica_evaluator.evaluate(
        pred_bin, ref_bin, verbose=False, log_times=False
    )
    result = next(iter(result_grouped.values()))
    phase_times = getattr(result, "phase_times", None) or {}
    # Drop noisy per-metric entries; keep only phase_* level.
    return {
        f"phase_{name}": secs * 1e3
        for name, secs in phase_times.items()
        if not name.startswith("metric_")
    }


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def _summary(samples_ms: list[float]) -> dict[str, float]:
    """Full summary including mean, stddev, median, min, p90, n."""
    n = len(samples_ms)
    ordered = sorted(samples_ms)
    mean = sum(samples_ms) / n
    stddev = statistics.stdev(samples_ms) if n > 1 else 0.0
    median = ordered[n // 2] if n % 2 else 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])
    p90_idx = min(int(0.9 * (n - 1) + 0.5), n - 1)
    return {
        "min": ordered[0],
        "median": median,
        "p90": ordered[p90_idx],
        "mean": mean,
        "stddev": stddev,
        "n": float(n),
    }


def _welch_pvalue(default_samples: list[float], variant_samples: list[float]) -> float:
    """Two-sided Welch's t-test p-value. Falls back to 1.0 if samples are degenerate."""
    if len(default_samples) < 2 or len(variant_samples) < 2:
        return 1.0
    if (
        statistics.stdev(default_samples) == 0
        and statistics.stdev(variant_samples) == 0
    ):
        # Identical constants — no evidence of difference.
        return 1.0
    result = stats.ttest_ind(default_samples, variant_samples, equal_var=False)
    p = float(result.pvalue)
    return p if not math.isnan(p) else 1.0


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _fmt_pct(pct: float) -> str:
    if pct >= 0:
        return f"+{pct:7.1f}%"
    return f"−{abs(pct):7.1f}%"


def _fmt_pvalue(p: float) -> str:
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def _significance_marker(pct: float, p: float, alpha: float) -> str:
    if p >= alpha:
        return ""
    return " 🔴" if pct > 0 else (" 🟢" if pct < 0 else "")


def _print_sanity_check(default_block: dict[str, Any]) -> None:
    cold = default_block["cold_start_ms"]
    warm = _summary(default_block["samples_ms"])["median"]
    ratio = cold / warm if warm > 0 else float("nan")
    n = len(default_block["samples_ms"])
    print("  Sanity check (default toggles):")
    print(f"    cold-start (1st call)     {cold:8.1f} ms")
    print(f"    steady-state median (n={n})  {warm:8.1f} ms")
    print(f"    warmup ratio              {ratio:7.2f}×")
    if ratio > 2.0:
        print(
            "    (large warmup effect — lazy imports / cache fills; ad-hoc\n"
            "     single-call timings measure cold-start, not steady-state.)"
        )
    print()


def _print_case_report(
    case: SyntheticCase,
    n_instances: int,
    per_variant: dict[str, dict[str, Any]],
    alpha: float,
) -> None:
    print(f"\n=== {case.name}  shape={case.shape}  instances={n_instances} ===\n")
    _print_sanity_check(per_variant["default"])

    default_samples = per_variant["default"]["samples_ms"]
    default_summary = _summary(default_samples)
    default_median = default_summary["median"]

    # Sort non-default variants by |Δ%| for display.
    ordered = ["default"] + sorted(
        (k for k in per_variant if k != "default"),
        key=lambda k: (
            abs(_summary(per_variant[k]["samples_ms"])["median"] - default_median)
            / max(default_median, 1e-9)
        ),
        reverse=True,
    )

    print(
        f"  {'variant':40s}  {'mean ± stddev':>17s}  {'median':>8s}  "
        f"{'Δ vs default':>16s}  {'p-value':>10s}"
    )
    print("  " + "-" * 108)
    for name in ordered:
        block = per_variant[name]
        summary = _summary(block["samples_ms"])
        mean_txt = f"{summary['mean']:6.1f} ± {summary['stddev']:5.1f} ms"
        med_txt = f"{summary['median']:6.1f}"

        if name == "default":
            delta_txt = f"{'baseline':>16s}"
            p_txt = f"{'—':>10s}"
        else:
            # For parallel_aggregator, the honest "does parallel help?" number
            # is amortized-per-iter (batch_wall / repeats) vs sequential mean —
            # not the per-worker mean, which is inflated by lock waits, CPU
            # contention, and BLAS oversubscription. For every other variant,
            # per-call mean == amortized (iterations don't overlap), so keeping
            # today's median-vs-median is fine.
            if block.get("mode") == "parallel_aggregator":
                headline = block.get("amortized_per_iter_ms", summary["mean"])
                default_headline = default_summary["mean"]
            else:
                headline = summary["median"]
                default_headline = default_median
            pct = (
                (headline - default_headline) / default_headline * 100
                if default_headline > 0
                else 0.0
            )
            p = _welch_pvalue(default_samples, block["samples_ms"])
            marker = _significance_marker(pct, p, alpha)
            delta_txt = f"{_fmt_pct(pct)}{marker}"
            p_txt = f"{_fmt_pvalue(p):>10s}"

        display_name = name
        if block.get("mode") == "parallel_aggregator":
            display_name = f"parallel_aggregator (×{block['workers']})"
        print(
            f"  {display_name:40s}  {mean_txt:>17s}  {med_txt:>8s}  "
            f"{delta_txt:>16s}  {p_txt}"
        )

    _print_batch_throughput(per_variant)
    _print_phase_attribution(per_variant, alpha)


def _print_batch_throughput(per_variant: dict[str, dict[str, Any]]) -> None:
    seq = per_variant.get("default")
    par = None
    for block in per_variant.values():
        if block.get("mode") == "parallel_aggregator":
            par = block
            break
    if seq is None or par is None:
        return
    seq_batch = seq["batch_wall_ms"]
    par_batch = par["batch_wall_ms"]
    speedup = seq_batch / par_batch if par_batch > 0 else float("nan")
    n_reps = len(seq["samples_ms"])
    par_workers = par["workers"]
    par_amortized = par.get("amortized_per_iter_ms", par_batch / max(n_reps, 1))
    par_per_worker_mean = _summary(par["samples_ms"])["mean"]

    print(
        f"\n  Batch throughput (default toggles, {n_reps} iterations):\n"
        f"    sequential                    {seq_batch:8.1f} ms total\n"
        f"    parallel_aggregator × {par_workers:<4}    {par_batch:8.1f} ms total   → {speedup:.2f}× speedup"
    )
    print(
        f"\n  Note on the parallel_aggregator row:\n"
        f"    - effective/iter (drives Δ vs default) = batch_wall / repeats = {par_amortized:.1f} ms\n"
        f"    - per-worker mean±sd shown in the table = {par_per_worker_mean:.1f} ms, which includes\n"
        f"      lock waits, CPU contention, and BLAS thread oversubscription."
    )


def _print_phase_attribution(
    per_variant: dict[str, dict[str, Any]],
    alpha: float,
) -> None:
    default_phases = per_variant["default"].get("phase_times_ms", {}) or {}
    if not default_phases:
        return
    non_default = [n for n in per_variant if n != "default"]
    printed_header = False
    for name in non_default:
        block = per_variant[name]
        if block.get("mode") == "parallel_aggregator":
            continue  # phase_times not harvested for parallel variant
        phases = block.get("phase_times_ms", {}) or {}
        rows: list[tuple[str, float, float, float]] = []
        for phase, h in phases.items():
            b = default_phases.get(phase)
            if b is None:
                continue
            if b < 0.5 and h < 0.5:
                continue
            pct = (h - b) / b * 100 if b > 0 else 0.0
            rows.append((phase, b, h, pct))
        if not rows:
            continue
        if not printed_header:
            print(
                "\n  Phase-level attribution (single-shot probe, "
                "no p-value — sorted by biggest mover):"
            )
            printed_header = True
        rows.sort(key=lambda r: abs(r[3]), reverse=True)
        print(f"\n    [{name}]")
        for phase, b, h, pct in rows[:6]:
            marker = " 🔴" if pct >= 10 else (" 🟢" if pct <= -10 else "")
            print(
                f"      {phase:32s} {b:6.1f} → {h:6.1f} ms  ({_fmt_pct(pct)}{marker})"
            )


def _print_ranking(
    all_results: list[tuple[SyntheticCase, dict[str, dict[str, Any]]]],
    alpha: float,
) -> None:
    """Aggregate ranking: which variant helped/hurt most on average."""
    per_variant_deltas: dict[str, list[tuple[float, float]]] = {}
    for _, per_variant in all_results:
        default_samples = per_variant["default"]["samples_ms"]
        default_median = _summary(default_samples)["median"]
        for name, block in per_variant.items():
            if name == "default":
                continue
            summary = _summary(block["samples_ms"])
            med = summary["median"]
            pct = (
                (med - default_median) / default_median * 100
                if default_median > 0
                else 0.0
            )
            p = _welch_pvalue(default_samples, block["samples_ms"])
            key = name
            if block.get("mode") == "parallel_aggregator":
                key = f"parallel_aggregator (×{block['workers']})"
            per_variant_deltas.setdefault(key, []).append((pct, p))

    if not per_variant_deltas:
        return

    print(
        "\n=== Overall ranking (mean Δ% across cases; "
        "marker = significant in every case) ===\n"
    )
    ranked = sorted(
        per_variant_deltas.items(),
        key=lambda kv: sum(p for p, _ in kv[1]) / len(kv[1]),
        reverse=True,
    )
    print(f"  {'variant':40s}  {'mean Δ%':>10s}  per-case (Δ%, p)")
    print("  " + "-" * 96)
    for name, pairs in ranked:
        mean_pct = sum(p for p, _ in pairs) / len(pairs)
        all_significant = all(p < alpha for _, p in pairs)
        marker = _significance_marker(mean_pct, 0.0 if all_significant else 1.0, alpha)
        per_case_txt = ", ".join(
            f"{_fmt_pct(pct).strip()} (p={_fmt_pvalue(p)})" for pct, p in pairs
        )
        print(f"  {name:40s}  {mean_pct:+8.1f}%{marker:2s}  {per_case_txt}")


# --------------------------------------------------------------------------- #
# CSV output — written incrementally so a later crash doesn't lose earlier cases
# --------------------------------------------------------------------------- #
_CSV_COLUMNS = [
    "case_name",
    "shape",
    "n_instances",
    "variant",
    "mode",
    "workers",
    "cold_start_ms",
    "mean_ms",
    "stddev_ms",
    "median_ms",
    "min_ms",
    "p90_ms",
    "n_samples",
    "batch_wall_ms",
    "amortized_per_iter_ms",
    "delta_pct_vs_default",
    "p_value",
    "significant",
]


def _open_csv(path: Path) -> tuple[TextIO, csv.writer]:
    csv_file = path.open("w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(_CSV_COLUMNS)
    csv_file.flush()
    return csv_file, writer


def _write_case_rows(
    writer: csv.writer,
    case: SyntheticCase,
    n_instances: int,
    per_variant: dict[str, dict[str, Any]],
    alpha: float,
) -> None:
    """Write one row per variant for this case. Flushed by caller."""
    default_samples = per_variant["default"]["samples_ms"]
    default_summary = _summary(default_samples)
    default_median = default_summary["median"]
    default_mean = default_summary["mean"]

    shape_txt = "x".join(str(x) for x in case.shape)

    for name, block in per_variant.items():
        summary = _summary(block["samples_ms"])
        mode = block.get("mode", "sequential")
        workers = block.get("workers", 1)
        amortized = block.get("amortized_per_iter_ms")

        if name == "default":
            delta_pct: Any = ""
            p_value: Any = ""
            significant: Any = ""
        else:
            # Same headline logic as the printed table: parallel_aggregator's Δ
            # is amortized-per-iter vs sequential mean; everything else is
            # median vs default median.
            if mode == "parallel_aggregator" and amortized is not None:
                headline = amortized
                default_headline = default_mean
            else:
                headline = summary["median"]
                default_headline = default_median
            delta_pct_val = (
                (headline - default_headline) / default_headline * 100
                if default_headline > 0
                else 0.0
            )
            p_value_val = _welch_pvalue(default_samples, block["samples_ms"])
            delta_pct = round(delta_pct_val, 2)
            p_value = round(p_value_val, 6)
            significant = p_value_val < alpha

        writer.writerow(
            [
                case.name,
                shape_txt,
                n_instances,
                name,
                mode,
                workers,
                round(block.get("cold_start_ms", 0.0), 2),
                round(summary["mean"], 2),
                round(summary["stddev"], 2),
                round(summary["median"], 2),
                round(summary["min"], 2),
                round(summary["p90"], 2),
                int(summary["n"]),
                round(block.get("batch_wall_ms", 0.0), 2),
                round(amortized, 2) if amortized is not None else "",
                delta_pct,
                p_value,
                significant,
            ]
        )


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def _run_case(
    case: SyntheticCase,
    variants: dict[str, PanopticaSpeedToggles],
    repeats: int,
    warmup: int,
    workers: int,
    include_parallel_aggregator: bool,
    out_dir: Path,
) -> tuple[int, dict[str, dict[str, Any]]]:
    pred, ref = case.build()
    n_instances = int(np.max(ref)) if ref.size else 0
    pred_bin = (pred > 0).astype(np.uint8)
    ref_bin = (ref > 0).astype(np.uint8)

    per_variant: dict[str, dict[str, Any]] = {}
    for name, toggles in variants.items():
        per_variant[name] = _measure_variant_sequential(
            name, toggles, pred_bin, ref_bin, out_dir, repeats=repeats, warmup=warmup
        )

    if include_parallel_aggregator:
        per_variant["parallel_aggregator"] = _measure_variant_parallel_aggregator(
            variants["default"],
            pred_bin,
            ref_bin,
            out_dir,
            repeats=repeats,
            warmup=warmup,
            workers=workers,
        )

    return n_instances, per_variant


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--factorial",
        action="store_true",
        help="Sweep all 2^N combinations instead of only one-at-a-time (OAT).",
    )
    parser.add_argument("--case", default=None, help="Substring filter on case name.")
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help=f"Timed iterations per variant (default {DEFAULT_REPEATS}).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help=f"Discarded warmup iterations per variant (default {DEFAULT_WARMUP}).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(2, (os.cpu_count() or 4) // 2),
        help=(
            "Pool size for the parallel_aggregator variant "
            "(default: half of CPU count, capped to --repeats — no point spawning "
            "more workers than there are tasks)."
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help=f"Significance threshold for Welch's t-test (default {DEFAULT_ALPHA}).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("toggle_impact_results.csv"),
        help=(
            "Path to write results as CSV, one row per (case, variant). "
            "Flushed after each case so a crash in a later case still leaves "
            "the earlier ones on disk. Overwrites existing file. "
            "(default: toggle_impact_results.csv in cwd)"
        ),
    )
    args = parser.parse_args()

    # Never spawn more workers than there are tasks — otherwise we pay fork cost
    # for workers that never get scheduled.
    args.workers = max(1, min(args.workers, args.repeats))

    variants = _full_factorial() if args.factorial else _one_at_a_time()
    include_parallel_aggregator = (
        not args.factorial and args.repeats >= 2 and args.workers >= 2
    )
    cases = benchmark_cases(
        shapes2d=[(64, 64), (256, 256), (512, 64)],
        shapes3d=[(64, 64, 64), (160, 160, 160), (32, 512, 128)],
        n_instances=[2, 5, 20],
        include_spine_example=False,
    )

    mode = "factorial (2^N combinations)" if args.factorial else "one-at-a-time"
    extra_variant = (
        f" + parallel_aggregator (×{args.workers})"
        if include_parallel_aggregator
        else ""
    )
    print(
        f"Toggle-impact sweep: {mode}, {len(cases)} case(s), "
        f"{len(variants)} variant(s){extra_variant}, "
        f"repeats={args.repeats}, warmup={args.warmup}, alpha={args.alpha}."
    )
    print(f"Writing per-case results to: {args.csv.resolve()}")

    from panoptica.utils.logger import set_log_level

    set_log_level("WARNING")  # suppress aggregator/evaluator debug spam

    csv_file, csv_writer = _open_csv(args.csv)
    try:
        with TemporaryDirectory(prefix="panoptica_toggle_impact_") as tmp:
            out_dir = Path(tmp)
            all_results: list[tuple[SyntheticCase, dict[str, dict[str, Any]]]] = []
            for case in cases:
                n_instances, per_variant = _run_case(
                    case,
                    variants,
                    repeats=args.repeats,
                    warmup=args.warmup,
                    workers=args.workers,
                    include_parallel_aggregator=include_parallel_aggregator,
                    out_dir=out_dir,
                )
                _print_case_report(case, n_instances, per_variant, alpha=args.alpha)
                # Persist this case immediately so a later crash can't lose it.
                _write_case_rows(csv_writer, case, n_instances, per_variant, args.alpha)
                csv_file.flush()
                os.fsync(csv_file.fileno())
                all_results.append((case, per_variant))
    finally:
        csv_file.close()

    if len(all_results) > 1:
        _print_ranking(all_results, alpha=args.alpha)


if __name__ == "__main__":
    main()
