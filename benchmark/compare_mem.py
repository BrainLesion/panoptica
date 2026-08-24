"""Compare two memory-benchmark JSON files and emit a scan-friendly markdown report.

Both inputs are produced by :mod:`benchmark.bench_mem` via ``--json``. Stdout is
GitHub-flavoured markdown starting with the sticky marker
``<!-- panoptica-benchmark-mem -->`` so the workflow can find and update the
memory PR comment across pushes independently of the speed comment.

Each measurement is a dict ``{min, median, p90, mean, stddev, n}`` (in MiB, peak
RSS growth per call). The gate mirrors :mod:`benchmark.compare` but with a MiB
noise floor: a measurement counts as a regression only when *both*:

* head_median > baseline_median * (1 + threshold/100)
* Welch's t-test on ``(mean, stddev, n)`` yields p < alpha

with a baseline ≥ ``GATE_MIN_BASELINE_MIB`` floor to keep sub-MiB jitter out of
the verdict.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Row / _stats / _diff_measurements / _row_pvalue / _fmt_pct / _fmt_pvalue are
# unit-agnostic — reuse them so the two comparators can never disagree on the
# statistical primitives.
from benchmark.compare import (  # noqa: E402
    Row,
    _diff_measurements,
    _fmt_pct,
    _fmt_pvalue,
    _row_pvalue,
    _stats,  # noqa: F401  (re-exported for potential future callers)
)

GATE_MIN_BASELINE_MIB = 1.0
KEY_TABLE_MAX_ROWS = 10
KEY_TABLE_MIN_DELTA_MIB = 0.05
MARKER = "<!-- panoptica-benchmark-mem -->"
DEFAULT_ALPHA = 0.05

KEY_TABLE_ALWAYS_KEYS = frozenset({"end_to_end"})


def _in_key_table_whitelist(key: str) -> bool:
    return key in KEY_TABLE_ALWAYS_KEYS


def _load(path: str) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _cases_by_name(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {c["name"]: c for c in doc.get("cases", [])}


def _measurements(case: dict[str, Any]) -> dict[str, Any]:
    """Accept both ``measurements_mb`` (mem bench) and ``measurements_ms`` (legacy
    time bench) — the latter lets somebody accidentally point compare_mem at a
    timing JSON without a KeyError."""
    if "measurements_mb" in case:
        return case["measurements_mb"]
    return case.get("measurements_ms", {})


def _is_gated(row: Row) -> bool:
    """Only workload-level measurements above the MiB noise floor gate the PR."""
    if not row.both_present:
        return False
    assert row.b is not None
    if row.b["median"] < GATE_MIN_BASELINE_MIB:
        return False
    return not row.key.startswith("metric_")


def _row_verdict(row: Row, alpha: float, min_pct: float = 10.0) -> str | None:
    if not _is_gated(row):
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
    return _row_verdict(row, alpha=alpha, min_pct=threshold_pct) == "regression"


def _pct_marker(row: Row, alpha: float) -> str:
    verdict = _row_verdict(row, alpha=alpha)
    if verdict == "regression":
        return " 🔴"
    if verdict == "win":
        return " 🟢"
    return ""


def _fmt_mib_with_spread(stats: dict[str, float]) -> str:
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
    sampler = head.get("sampler", "?")
    parts = [
        MARKER,
        "",
        f"## 🧠 Peak-RSS benchmark vs `{head.get('commit', '?')}`",
        "",
        (
            f"**Gate:** {gate_badge} &nbsp;·&nbsp; "
            f"🟢 {wins} win{'s' if wins != 1 else ''} &nbsp;·&nbsp; "
            f"🔴 {regressions} regression{'s' if regressions != 1 else ''} &nbsp;·&nbsp; "
            f"Python {head.get('python', '?')} &nbsp;·&nbsp; "
            f"repeats={repeats}, warmup={warmup} &nbsp;·&nbsp; sampler={sampler} "
            f"&nbsp;·&nbsp; α={alpha}"
        ),
        "",
        (
            "> Values are peak RSS growth **per call** in MiB, formatted as "
            "`median ±(p90−min)/2`. Baseline is measured on the same runner "
            "(main's `panoptica` + this PR's mem-bench harness). A row is "
            f"decorated 🔴 / 🟢 only when |Δ%| ≥ 10 % **and** Welch's t-test "
            f"on the (mean, stddev, n) summary yields `p < {alpha}`. The "
            f"PR-fail gate also requires baseline ≥ {GATE_MIN_BASELINE_MIB} MiB "
            f"so sub-MiB jitter doesn't gate the PR."
        ),
        "",
    ]
    if head_only_total > 0:
        parts.append(
            f"> ℹ️ **{head_only_total} measurement{'s' if head_only_total != 1 else ''} "
            f"present only in head** — likely because main's `panoptica` predates "
            f"the code path they exercise."
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
        f"> 🚨 **Memory-regression gate FAILED** — `{worst_row.key}` in "
        f"`{offender_case}` grew by `{_fmt_pct(worst_row.pct)}` "
        f"(baseline median {worst_row.b['median']:.2f} MiB, "
        f"head median {worst_row.h['median']:.2f} MiB, p={p_txt}). "
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
        f"### {case_name} — `end_to_end` peak **{hero.b['median']:.1f} → "
        f"{hero.h['median']:.1f} MiB** ({_fmt_pct(hero.pct)})\n"
    )


_MISSING = "—"


def _fmt_row(row: Row, alpha: float) -> str:
    if row.only_in_head:
        assert row.h is not None
        return (
            f"| `{row.key}` | {_MISSING} | {_fmt_mib_with_spread(row.h)} "
            f"| _new in head_ | {_MISSING} |"
        )
    if row.only_in_baseline:
        assert row.b is not None
        return (
            f"| `{row.key}` | {_fmt_mib_with_spread(row.b)} | {_MISSING} "
            f"| _absent in head_ | {_MISSING} |"
        )
    assert row.b is not None and row.h is not None
    return (
        f"| `{row.key}` | {_fmt_mib_with_spread(row.b)} | {_fmt_mib_with_spread(row.h)} "
        f"| {_fmt_pct(row.pct)}{_pct_marker(row, alpha)} "
        f"| {_fmt_pvalue(_row_pvalue(row))} |"
    )


def _emit_key_table(
    rows: list[Row], alpha: float, max_rows: int = KEY_TABLE_MAX_ROWS
) -> str:
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
            and abs(r.delta) >= KEY_TABLE_MIN_DELTA_MIB
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
        "| Measurement | baseline MiB (median ±½·range) | head MiB (median ±½·range) | Δ % | p |",
        "| --- | ---: | ---: | :--- | ---: |",
    ]
    for row in top:
        lines.append(_fmt_row(row, alpha))
    lines.append("")
    return "\n".join(lines)


def _emit_full_breakdown(rows: list[Row], alpha: float) -> str:
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
        "| Measurement | baseline MiB (median ±½·range) | head MiB (median ±½·range) | Δ % | p |",
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
            _measurements(b_cases[name]),
            _measurements(h_cases[name]),
        )
        per_case.append((name, rows))
        for row in rows:
            if row.only_in_head:
                head_only_total += 1
            if not _is_gated(row):
                continue
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
    parser.add_argument("baseline", help="Baseline JSON produced by bench_mem.py --json")
    parser.add_argument("head", help="Head JSON produced by bench_mem.py --json")
    parser.add_argument(
        "--fail-on-regression-pct",
        type=float,
        default=None,
        help=(
            "Fail if any gated measurement's head median exceeds baseline median by "
            "this percentage AND Welch's t-test at --alpha is significant. "
            "Default: never fail."
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help=f"Significance threshold for Welch's t-test (default {DEFAULT_ALPHA}).",
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
            f"\n**Memory regression gate failed**: `{worst_row.key}` in "
            f"`{worst_case}` grew by {_fmt_pct(worst_row.pct)} "
            f"(baseline median {worst_row.b['median']:.2f} MiB, "
            f"head median {worst_row.h['median']:.2f} MiB, p={p_txt}).",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
