"""Compare two benchmark JSON files and emit a scan-friendly markdown report.

Both inputs are produced by :mod:`benchmark.bench_eval` via ``--json``. Stdout is
GitHub-flavoured markdown starting with the sticky marker ``<!-- panoptica-benchmark -->``
so the workflow can find and update the same PR comment across pushes.

Each measurement is expected to be a dict ``{min, median, p90}`` (in milliseconds).
Comparisons use the **median** as the primary point estimate and the
``(p90 - min)`` spread as the noise band. When ``--fail-on-regression-pct`` is
set, a measurement counts as a regression only when *both*:

* head_median > baseline_median * (1 + threshold/100)
* head_median > baseline_p90

The second condition kills failures triggered by baseline noise — head must
be clearly worse than the baseline's own observed spread. Only workload-level
entries (non-``metric_*``) with a baseline median ``>= GATE_MIN_BASELINE_MS`` are
gated; sub-millisecond timings are surfaced only inside the collapsible breakdown.

Older baseline JSONs where each measurement is a scalar (float ms) still load —
the scalar is treated as ``{min=median=p90=value}``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.stats import welch_pvalue_from_summary

# Sub-millisecond measurements are pure noise on shared CI runners; individual
# metric_* entries typically fall here. We surface them in the full breakdown but
# never use them to gate a PR.
GATE_MIN_BASELINE_MS = 1.0
KEY_TABLE_MAX_ROWS = 10
KEY_TABLE_MIN_DELTA_MS = 0.05
MARKER = "<!-- panoptica-benchmark -->"
DEFAULT_ALPHA = 0.05  # Welch's t-test significance threshold for markers + gate

# Measurements that always land in the key table regardless of absolute size.
# The gate (and its 1-ms noise floor) is unchanged; this just guarantees
# visibility so `phase_preprocess` doesn't hide in the breakdown for small
# cases while appearing in the main table for larger ones. Extend either set
# to pin more rows.
KEY_TABLE_ALWAYS_KEYS = frozenset({"end_to_end"})
KEY_TABLE_ALWAYS_PREFIXES = ("phase_",)


def _in_key_table_whitelist(key: str) -> bool:
    return key in KEY_TABLE_ALWAYS_KEYS or key.startswith(KEY_TABLE_ALWAYS_PREFIXES)


def _load(path: str) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _cases_by_name(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {c["name"]: c for c in doc.get("cases", [])}


def _stats(raw: Any) -> dict[str, float]:
    """Normalize a measurement value to a stats dict.

    Fields always present: ``min``, ``median``, ``p90``.
    Optionally present when the source JSON came from a bench_eval.py that
    calls :func:`benchmark.stats.summarize` (``mean``, ``stddev``, ``n``) —
    those enable Welch's t-test in :func:`_row_pvalue`. Legacy entries (bare
    floats or dicts missing the newer fields) still load; the t-test path
    then returns ``None`` and callers fall back to the p90-band heuristic.
    """
    if isinstance(raw, dict):
        out = {
            "min": float(raw.get("min", raw.get("median", 0.0))),
            "median": float(raw.get("median", raw.get("min", 0.0))),
            "p90": float(raw.get("p90", raw.get("median", raw.get("min", 0.0)))),
        }
        for k in ("mean", "stddev", "n"):
            if k in raw:
                out[k] = float(raw[k])
        return out
    val = float(raw)
    return {"min": val, "median": val, "p90": val}


class Row:
    """One measurement's baseline+head stats and derived deltas.

    ``b`` or ``h`` can be ``None`` when the key is present in only one document —
    this happens during the transition period when main's ``panoptica`` predates
    features exposed by HEAD (e.g. ``phase_times`` before this PR merges).
    """

    __slots__ = ("key", "b", "h", "delta", "pct", "spread_b", "spread_h")

    def __init__(
        self,
        key: str,
        baseline: dict[str, float] | None,
        head: dict[str, float] | None,
    ) -> None:
        self.key = key
        self.b = baseline
        self.h = head
        if baseline is None or head is None:
            self.delta = 0.0
            self.pct = 0.0
            self.spread_b = 0.0
            self.spread_h = 0.0
        else:
            b_med = baseline["median"]
            h_med = head["median"]
            self.delta = h_med - b_med
            self.pct = (self.delta / b_med * 100.0) if b_med > 0 else 0.0
            self.spread_b = max(baseline["p90"] - baseline["min"], 0.0)
            self.spread_h = max(head["p90"] - head["min"], 0.0)

    @property
    def only_in_head(self) -> bool:
        return self.b is None and self.h is not None

    @property
    def only_in_baseline(self) -> bool:
        return self.h is None and self.b is not None

    @property
    def both_present(self) -> bool:
        return self.b is not None and self.h is not None


def _diff_measurements(
    baseline: dict[str, Any],
    head: dict[str, Any],
) -> list[Row]:
    all_keys = sorted(set(baseline) | set(head))
    rows: list[Row] = []
    for key in all_keys:
        b_raw = baseline.get(key)
        h_raw = head.get(key)
        b_stats = _stats(b_raw) if b_raw is not None else None
        h_stats = _stats(h_raw) if h_raw is not None else None
        rows.append(Row(key, b_stats, h_stats))
    return rows


def _is_gated(row: Row) -> bool:
    """Only workload-level measurements above the noise floor gate the PR."""
    if not row.both_present:
        return False
    assert row.b is not None
    if row.b["median"] < GATE_MIN_BASELINE_MS:
        return False
    return not row.key.startswith("metric_")


def _row_pvalue(row: Row) -> float | None:
    """Welch's t-test on the row's summary stats. Returns ``None`` when the JSON
    predates the ``mean``/``stddev``/``n`` fields."""
    if not row.both_present:
        return None
    assert row.b is not None and row.h is not None
    return welch_pvalue_from_summary(row.b, row.h)


def _row_verdict(row: Row, alpha: float, min_pct: float = 10.0) -> str | None:
    """Statistical verdict for a row: ``"regression"``, ``"win"``, or ``None``.

    Two conditions must both hold:

    1. ``|Δ %|`` exceeds ``min_pct`` — the change is practically visible.
    2. The two distributions differ significantly. We use Welch's t-test on
       ``(mean, stddev, n)`` (via :func:`benchmark.stats.welch_pvalue_from_summary`)
       when the JSON has those fields; otherwise fall back to the legacy p90-band
       separation (``head.min > baseline.p90`` or symmetric).

    This one predicate powers the row marker, the win/regression badge, and the
    PR-fail gate — so they can never disagree.
    """
    if not row.both_present:
        return None
    assert row.b is not None and row.h is not None
    if row.pct >= min_pct:
        p = _row_pvalue(row)
        if p is not None:
            return "regression" if p < alpha else None
        return "regression" if row.h["min"] > row.b["p90"] else None
    if row.pct <= -min_pct:
        p = _row_pvalue(row)
        if p is not None:
            return "win" if p < alpha else None
        return "win" if row.h["p90"] < row.b["min"] else None
    return None


def _is_regression(row: Row, threshold_pct: float, alpha: float) -> bool:
    """Gate-side regression check: same predicate as the marker, but with the
    caller-controlled |Δ%| threshold (typically the ``--fail-on-regression-pct``
    value, default 50)."""
    return _row_verdict(row, alpha=alpha, min_pct=threshold_pct) == "regression"


def _fmt_pct(pct: float) -> str:
    """Signed percent with a real minus sign so columns stay aligned."""
    if pct >= 0:
        return f"+{pct:.1f}%"
    return f"−{abs(pct):.1f}%"


def _fmt_pvalue(p: float | None) -> str:
    """P-value cell for the tables. ``None`` (no summary stats) → em-dash."""
    if p is None:
        return "—"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def _pct_marker(row: Row, alpha: float) -> str:
    """Decorate a row with 🔴 / 🟢 when its verdict is significant at ``alpha``."""
    verdict = _row_verdict(row, alpha=alpha)
    if verdict == "regression":
        return " 🔴"
    if verdict == "win":
        return " 🟢"
    return ""


def _fmt_ms_with_spread(stats: dict[str, float]) -> str:
    spread = max(stats["p90"] - stats["min"], 0.0) / 2.0
    return f"{stats['median']:.2f} ±{spread:.2f}"


def _emit_header(
    baseline: dict[str, Any],
    head: dict[str, Any],
    gate_pass: bool,
    wins: int,
    regressions: int,
    head_only_total: int,
    alpha: float,
) -> str:
    gate_badge = "✅ PASS" if gate_pass else "🔴 FAIL"
    repeats = head.get("repeats", "?")
    warmup = head.get("warmup", "?")
    parts = [
        MARKER,
        "",
        f"## 📊 Benchmark vs `{head.get('commit', '?')}`",
        "",
        (
            f"**Gate:** {gate_badge} &nbsp;·&nbsp; "
            f"🟢 {wins} win{'s' if wins != 1 else ''} &nbsp;·&nbsp; "
            f"🔴 {regressions} regression{'s' if regressions != 1 else ''} &nbsp;·&nbsp; "
            f"Python {head.get('python', '?')} &nbsp;·&nbsp; "
            f"repeats={repeats}, warmup={warmup} &nbsp;·&nbsp; α={alpha}"
        ),
        "",
        (
            "> Baseline is measured on the same runner (main's `panoptica` + this PR's benchmark harness). "
            "The committed `benchmark/baseline.json` is only a fallback for `workflow_dispatch` runs."
        ),
        "",
        (
            f"> Values are `median ±(p90−min)/2`. The key table always lists "
            f"`end_to_end` and every `phase_*` present in both docs (regardless of "
            f"size), plus any other workload measurement whose baseline ≥ 1 ms, "
            f"sorted by |Δ%|. A row is decorated 🔴 / 🟢 only when |Δ%| ≥ 10 % "
            f"**and** Welch's t-test on the (mean, stddev, n) summary yields "
            f"`p < {alpha}` — the same predicate powers the win/regression counter "
            f"and the PR-fail gate, so badge and table always agree. When the JSON "
            f"predates the mean/stddev/n fields, we fall back to the legacy "
            f"p90-band separation. The PR-fail gate also requires baseline ≥ 1 ms "
            f"to keep sub-ms noise out of the verdict."
        ),
        "",
    ]
    if head_only_total > 0:
        parts.append(
            f"> ℹ️ **{head_only_total} measurement{'s' if head_only_total != 1 else ''} "
            f"present only in head** — likely because main's `panoptica` predates the "
            f"feature that exposes them (e.g. `phase_times`, per-metric timings). Their "
            f"absolute values are shown in each case's full breakdown but they can't be "
            f"compared until the feature lands in `main`."
        )
        parts.append("")
    return "\n".join(parts)


def _emit_gate_callout(
    worst_row: Row | None,
    threshold: float,
    offender_case: str | None,
) -> str:
    if worst_row is None or offender_case is None:
        return ""
    assert worst_row.b is not None and worst_row.h is not None
    p = _row_pvalue(worst_row)
    p_txt = _fmt_pvalue(p)
    return (
        f"> 🚨 **Regression gate FAILED** — `{worst_row.key}` in `{offender_case}` "
        f"regressed by `{_fmt_pct(worst_row.pct)}` "
        f"(baseline median {worst_row.b['median']:.2f} ms, "
        f"head median {worst_row.h['median']:.2f} ms, p={p_txt}). "
        f"Threshold: `{_fmt_pct(threshold)}`.\n\n"
    )


def _find_row(rows: list[Row], key: str) -> Row | None:
    for row in rows:
        if row.key == key:
            return row
    return None


def _emit_case_hero(case_name: str, rows: list[Row]) -> str:
    hero = _find_row(rows, "end_to_end")
    if hero is None or not hero.both_present:
        return f"### {case_name}\n"
    assert hero.b is not None and hero.h is not None
    return (
        f"### {case_name} — `end_to_end` **{hero.b['median']:.1f} → {hero.h['median']:.1f} ms** "
        f"({_fmt_pct(hero.pct)})\n"
    )


_MISSING = "—"


def _fmt_row(row: Row, alpha: float) -> str:
    if row.only_in_head:
        assert row.h is not None
        return (
            f"| `{row.key}` | {_MISSING} | {_fmt_ms_with_spread(row.h)} "
            f"| _new in head_ | {_MISSING} |"
        )
    if row.only_in_baseline:
        assert row.b is not None
        return (
            f"| `{row.key}` | {_fmt_ms_with_spread(row.b)} | {_MISSING} "
            f"| _absent in head_ | {_MISSING} |"
        )
    assert row.b is not None and row.h is not None
    return (
        f"| `{row.key}` | {_fmt_ms_with_spread(row.b)} | {_fmt_ms_with_spread(row.h)} "
        f"| {_fmt_pct(row.pct)}{_pct_marker(row, alpha)} "
        f"| {_fmt_pvalue(_row_pvalue(row))} |"
    )


def _emit_key_table(
    rows: list[Row], alpha: float, max_rows: int = KEY_TABLE_MAX_ROWS
) -> str:
    # Key table only shows comparable rows — head-only ones belong in the breakdown.
    #
    # Two sources fill the table:
    #   1. Whitelist rows (`end_to_end`, every `phase_*`): always present when
    #      the row exists in both docs, regardless of absolute size. This is what
    #      keeps `phase_preprocess` visible for 2D cases even though its baseline
    #      is below the 1-ms gate floor.
    #   2. Existing gated set (workload measurements above the noise floor)
    #      that are NOT already in the whitelist, sorted by |Δ%|.
    # Whitelist comes first so the max_rows cap trims the gated tail, never
    # the guaranteed rows.
    whitelist = sorted(
        [r for r in rows if r.both_present and _in_key_table_whitelist(r.key)],
        key=lambda r: abs(r.pct),
        reverse=True,
    )
    whitelist_keys = {r.key for r in whitelist}
    gated = sorted(
        [
            r
            for r in rows
            if _is_gated(r)
            and abs(r.delta) >= KEY_TABLE_MIN_DELTA_MS
            and r.key not in whitelist_keys
        ],
        key=lambda r: abs(r.pct),
        reverse=True,
    )
    combined = whitelist + gated
    if not combined:
        return "_No key measurements to show._\n\n"

    top = combined[:max_rows]
    lines = [
        "| Measurement | baseline ms (median ±½·range) | head ms (median ±½·range) | Δ % | p |",
        "| --- | ---: | ---: | :--- | ---: |",
    ]
    for row in top:
        lines.append(_fmt_row(row, alpha))
    lines.append("")
    return "\n".join(lines)


def _emit_full_breakdown(rows: list[Row], alpha: float) -> str:
    # Compared rows first (biggest |Δ%| at top), then head-only, then baseline-only.
    compared = sorted(
        [r for r in rows if r.both_present], key=lambda r: abs(r.pct), reverse=True
    )
    head_only = sorted(
        [r for r in rows if r.only_in_head],
        key=lambda r: r.h["median"] if r.h else 0.0,
        reverse=True,
    )
    baseline_only = sorted(
        [r for r in rows if r.only_in_baseline],
        key=lambda r: r.b["median"] if r.b else 0.0,
        reverse=True,
    )
    ordered = compared + head_only + baseline_only
    lines = [
        f"<details><summary>Full breakdown ({len(ordered)} measurements — "
        f"{len(compared)} compared, {len(head_only)} head-only, "
        f"{len(baseline_only)} baseline-only)</summary>",
        "",
        "| Measurement | baseline ms (median ±½·range) | head ms (median ±½·range) | Δ % | p |",
        "| --- | ---: | ---: | :--- | ---: |",
    ]
    for row in ordered:
        lines.append(_fmt_row(row, alpha))
    lines.append("")
    lines.append("</details>")
    lines.append("")
    return "\n".join(lines)


def _emit_report(
    baseline: dict[str, Any],
    head: dict[str, Any],
    fail_threshold: float | None,
    alpha: float,
) -> tuple[str, bool, Row | None, str | None]:
    """Return (markdown, gate_pass, worst_regression_row, offender_case)."""
    b_cases = _cases_by_name(baseline)
    h_cases = _cases_by_name(head)
    all_case_names = [c["name"] for c in head.get("cases", [])] + [
        n for n in b_cases if n not in {c["name"] for c in head.get("cases", [])}
    ]

    per_case: list[tuple[str, list[Row]]] = []
    wins = 0
    regressions = 0
    head_only_total = 0
    worst_row: Row | None = None
    worst_case: str | None = None

    for name in all_case_names:
        if name not in b_cases or name not in h_cases:
            per_case.append((name, []))
            continue
        rows = _diff_measurements(
            b_cases[name]["measurements_ms"],
            h_cases[name]["measurements_ms"],
        )
        per_case.append((name, rows))
        for row in rows:
            if row.only_in_head:
                head_only_total += 1
            if not _is_gated(row):
                continue
            # Wins/regressions counter, PR-fail gate, and the row marker all
            # come from the same _row_verdict predicate. If badge says
            # "0 regressions", the tables can't contain a 🔴, by construction.
            verdict = _row_verdict(row, alpha=alpha)
            if verdict == "regression":
                regressions += 1
            elif verdict == "win":
                wins += 1
            if fail_threshold is not None and _is_regression(
                row, fail_threshold, alpha
            ):
                if worst_row is None or row.pct > worst_row.pct:
                    worst_row = row
                    worst_case = name

    gate_pass = fail_threshold is None or worst_row is None

    body: list[str] = []
    body.append(
        _emit_header(
            baseline, head, gate_pass, wins, regressions, head_only_total, alpha
        )
    )
    if not gate_pass and fail_threshold is not None:
        body.append(_emit_gate_callout(worst_row, fail_threshold, worst_case))

    for name, rows in per_case:
        if not rows:
            body.append(f"### {name}\n\n_(present in only one document — skipping)_\n")
            continue
        body.append(_emit_case_hero(name, rows))
        body.append(_emit_key_table(rows, alpha))
        body.append(_emit_full_breakdown(rows, alpha))

    return "\n".join(body).rstrip() + "\n", gate_pass, worst_row, worst_case


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "baseline", help="Baseline JSON produced by bench_eval.py --json"
    )
    parser.add_argument("head", help="Head JSON produced by bench_eval.py --json")
    parser.add_argument(
        "--fail-on-regression-pct",
        type=float,
        default=None,
        help=(
            "Fail if any gated measurement's head median exceeds baseline median by "
            "this percentage AND the two distributions differ significantly "
            "(Welch's t-test at --alpha). Default: never fail."
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help=(
            f"Significance threshold for Welch's t-test on summary stats "
            f"(default {DEFAULT_ALPHA}). Powers the row 🔴 / 🟢 marker, the "
            f"win/regression badge, and --fail-on-regression-pct. Falls back "
            f"to p90-band separation when a JSON predates the mean/stddev/n "
            f"fields."
        ),
    )
    args = parser.parse_args()

    baseline = _load(args.baseline)
    head = _load(args.head)

    markdown, gate_pass, worst_row, worst_case = _emit_report(
        baseline, head, args.fail_on_regression_pct, alpha=args.alpha
    )
    print(markdown)

    if not gate_pass and worst_row is not None:
        assert worst_row.b is not None and worst_row.h is not None
        p = _row_pvalue(worst_row)
        p_txt = _fmt_pvalue(p)
        print(
            f"\n**Regression gate failed**: `{worst_row.key}` in "
            f"`{worst_case}` regressed by {_fmt_pct(worst_row.pct)} "
            f"(baseline median {worst_row.b['median']:.2f} ms, "
            f"head median {worst_row.h['median']:.2f} ms, p={p_txt}).",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
