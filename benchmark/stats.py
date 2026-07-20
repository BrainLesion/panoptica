"""Shared statistics helpers for the benchmark timing suite.

Single home for the summary and Welch's t-test used by:

* :mod:`benchmark.bench_eval` — computes summaries per measurement and stores
  ``{min, median, p90, mean, stddev, n}`` in the JSON output.
* :mod:`benchmark.toggle_impact` — keeps raw samples in-process for the local
  significance test (uses ``welch_pvalue_from_samples``).
* :mod:`benchmark.compare` — reads summary stats from JSON, so uses
  ``welch_pvalue_from_summary`` (needs ``mean``, ``stddev``, ``n``).

Everything here is pure functions with no side effects. Importing this module
is cheap even though it reaches into scipy.
"""

from __future__ import annotations

import math
import statistics

from scipy import stats  # type: ignore[import-untyped]


def summarize(samples_ms: list[float]) -> dict[str, float]:
    """Full summary of ``samples_ms``: ``min``, ``median``, ``p90``, ``mean``, ``stddev``, ``n``.

    ``median`` linearly interpolates for even n. ``p90`` uses the nearest-rank
    method (rounded to the nearest available sample). ``stddev`` uses ``ddof=1``
    (sample stddev). ``n`` is stored as a float so it survives a JSON round-trip
    the same way as the other fields.
    """
    n = len(samples_ms)
    if n == 0:
        return {
            "min": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "mean": 0.0,
            "stddev": 0.0,
            "n": 0.0,
        }
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


def welch_pvalue_from_samples(a: list[float], b: list[float]) -> float:
    """Two-sided Welch's t-test p-value from raw samples.

    Returns ``1.0`` on degenerate inputs (n<2 on either side, or both sides
    have zero variance — indistinguishable constants).
    """
    if len(a) < 2 or len(b) < 2:
        return 1.0
    if statistics.stdev(a) == 0 and statistics.stdev(b) == 0:
        return 1.0
    p = float(stats.ttest_ind(a, b, equal_var=False).pvalue)
    return p if not math.isnan(p) else 1.0


def welch_pvalue_from_summary(a: dict[str, float], b: dict[str, float]) -> float | None:
    """Two-sided Welch's t-test p-value from summary stats.

    Requires ``mean``, ``stddev``, and ``n`` on both sides — the values
    :func:`summarize` produces. Returns ``None`` when either side is missing
    any of those keys (legacy JSON without the full schema), so callers can
    fall back to their own heuristic. Returns ``1.0`` on degenerate inputs
    with the same semantics as :func:`welch_pvalue_from_samples`.
    """
    required = ("mean", "stddev", "n")
    if not all(k in a and k in b for k in required):
        return None
    n_a = int(a["n"])
    n_b = int(b["n"])
    if n_a < 2 or n_b < 2:
        return 1.0
    if a["stddev"] == 0 and b["stddev"] == 0:
        return 1.0
    result = stats.ttest_ind_from_stats(
        mean1=a["mean"],
        std1=a["stddev"],
        nobs1=n_a,
        mean2=b["mean"],
        std2=b["stddev"],
        nobs2=n_b,
        equal_var=False,
    )
    p = float(result.pvalue)
    return p if not math.isnan(p) else 1.0
