"""Compare two benchmark JSON files and emit a scan-friendly markdown report.

Both inputs are produced by :mod:`benchmark.bench_eval` via ``--json``. Stdout is
GitHub-flavoured markdown starting with the sticky marker ``<!-- panoptica-benchmark -->``
so the workflow can find and update the same PR comment across pushes.

When ``--fail-on-regression-pct`` is set, exits non-zero if any *gated* measurement's
percent change exceeds that threshold. Gated measurements are workload-level entries
(non-``metric_*``) with a baseline ``>= GATE_MIN_BASELINE_MS`` — sub-millisecond
timings are pure noise on shared CI runners and are surfaced only inside the
collapsible full breakdown.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

# Sub-millisecond measurements are pure noise on shared CI runners; individual
# metric_* entries typically fall here. We surface them in the full breakdown but
# never use them to gate a PR.
GATE_MIN_BASELINE_MS = 1.0
KEY_TABLE_MAX_ROWS = 8
KEY_TABLE_MIN_DELTA_MS = 0.05
MARKER = "<!-- panoptica-benchmark -->"


def _load(path: str) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _cases_by_name(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {c["name"]: c for c in doc.get("cases", [])}


def _diff_measurements(
    baseline: dict[str, float],
    head: dict[str, float],
) -> list[tuple[str, float, float, float, float]]:
    """Return rows of (measurement, baseline_ms, head_ms, delta_ms, delta_pct)."""
    all_keys = sorted(set(baseline) | set(head))
    rows: list[tuple[str, float, float, float, float]] = []
    for key in all_keys:
        b = baseline.get(key)
        h = head.get(key)
        if b is None or h is None:
            continue
        delta = h - b
        pct = (delta / b * 100.0) if b > 0 else 0.0
        rows.append((key, b, h, delta, pct))
    return rows


def _is_gated(key: str, baseline_ms: float) -> bool:
    """Only workload-level measurements above the noise floor gate the PR."""
    if baseline_ms < GATE_MIN_BASELINE_MS:
        return False
    return not key.startswith("metric_")


def _fmt_pct(pct: float) -> str:
    """Signed percent with a real minus sign so columns stay aligned."""
    if pct >= 0:
        return f"+{pct:.1f}%"
    return f"−{abs(pct):.1f}%"


def _pct_marker(pct: float) -> str:
    if pct >= 10:
        return " 🔴"
    if pct <= -10:
        return " 🟢"
    return ""


def _emit_header(
    baseline: dict[str, Any],
    head: dict[str, Any],
    gate_pass: bool,
    wins: int,
    regressions: int,
) -> str:
    gate_badge = "✅ PASS" if gate_pass else "🔴 FAIL"
    parts = [
        MARKER,
        "",
        f"## 📊 Benchmark vs `{baseline.get('commit', '?')}`",
        "",
        (
            f"**Gate:** {gate_badge} &nbsp;·&nbsp; "
            f"🟢 {wins} win{'s' if wins != 1 else ''} &nbsp;·&nbsp; "
            f"🔴 {regressions} regression{'s' if regressions != 1 else ''} &nbsp;·&nbsp; "
            f"Python {head.get('python', '?')}"
        ),
        "",
        (
            "> Baseline is the committed `benchmark/baseline.json`. Refresh with"
            " `python benchmark/bench_eval.py --quick --json benchmark/baseline.json`."
        ),
        "",
    ]
    return "\n".join(parts)


def _emit_gate_callout(
    worst_pct: float,
    threshold: float,
    offender_case: str | None,
    offender_key: str | None,
) -> str:
    if offender_key is None or offender_case is None:
        return ""
    return (
        f"> 🚨 **Regression gate FAILED** — worst gated regression "
        f"`{_fmt_pct(worst_pct)}` exceeds threshold `{_fmt_pct(threshold)}` "
        f"(`{offender_key}` in `{offender_case}`).\n\n"
    )


def _find_row(rows: list[tuple], key: str) -> tuple | None:
    for row in rows:
        if row[0] == key:
            return row
    return None


def _emit_case_hero(case_name: str, rows: list[tuple]) -> str:
    hero = _find_row(rows, "end_to_end")
    if hero is None:
        return f"### {case_name}\n"
    _, b, h, _, pct = hero
    return (
        f"### {case_name} — `end_to_end` **{b:.1f} → {h:.1f} ms** ({_fmt_pct(pct)})\n"
    )


def _fmt_row(row: tuple[str, float, float, float, float]) -> str:
    key, b, h, _, pct = row
    return f"| `{key}` | {b:.2f} | {h:.2f} | {_fmt_pct(pct)}{_pct_marker(pct)} |"


def _emit_key_table(rows: list[tuple], max_rows: int = KEY_TABLE_MAX_ROWS) -> str:
    gated_rows = [r for r in rows if _is_gated(r[0], r[1])]
    # Drop rows whose absolute delta is below the noise floor even if the % is large.
    gated_rows = [r for r in gated_rows if abs(r[3]) >= KEY_TABLE_MIN_DELTA_MS]
    gated_rows.sort(key=lambda r: abs(r[4]), reverse=True)
    if not gated_rows:
        return "_No gated measurements moved meaningfully._\n\n"

    top = gated_rows[:max_rows]
    lines = [
        "| Measurement | baseline ms | head ms | Δ % |",
        "| --- | ---: | ---: | :--- |",
    ]
    for row in top:
        lines.append(_fmt_row(row))
    lines.append("")
    return "\n".join(lines)


def _emit_full_breakdown(rows: list[tuple]) -> str:
    ordered = sorted(rows, key=lambda r: abs(r[4]), reverse=True)
    lines = [
        f"<details><summary>Full breakdown ({len(ordered)} measurements)</summary>",
        "",
        "| Measurement | baseline ms | head ms | Δ % |",
        "| --- | ---: | ---: | :--- |",
    ]
    for row in ordered:
        lines.append(_fmt_row(row))
    lines.append("")
    lines.append("</details>")
    lines.append("")
    return "\n".join(lines)


def _emit_report(
    baseline: dict[str, Any],
    head: dict[str, Any],
    fail_threshold: float | None,
) -> tuple[str, float]:
    """Return (markdown, worst_gated_regression_pct)."""
    b_cases = _cases_by_name(baseline)
    h_cases = _cases_by_name(head)
    all_case_names = [c["name"] for c in head.get("cases", [])] + [
        n for n in b_cases if n not in {c["name"] for c in head.get("cases", [])}
    ]

    per_case: list[tuple[str, list[tuple]]] = []
    wins = 0
    regressions = 0
    worst_pct = 0.0
    worst_case: str | None = None
    worst_key: str | None = None

    for name in all_case_names:
        if name not in b_cases or name not in h_cases:
            per_case.append((name, []))
            continue
        rows = _diff_measurements(
            b_cases[name]["measurements_ms"],
            h_cases[name]["measurements_ms"],
        )
        per_case.append((name, rows))
        for key, b, _h, _d, pct in rows:
            if not _is_gated(key, b):
                continue
            if pct >= 10:
                regressions += 1
            elif pct <= -10:
                wins += 1
            if pct > worst_pct:
                worst_pct = pct
                worst_case = name
                worst_key = key

    gate_pass = fail_threshold is None or worst_pct <= fail_threshold

    body: list[str] = []
    body.append(_emit_header(baseline, head, gate_pass, wins, regressions))
    if not gate_pass and fail_threshold is not None:
        body.append(
            _emit_gate_callout(worst_pct, fail_threshold, worst_case, worst_key)
        )

    for name, rows in per_case:
        if not rows:
            body.append(f"### {name}\n\n_(present in only one document — skipping)_\n")
            continue
        body.append(_emit_case_hero(name, rows))
        body.append(_emit_key_table(rows))
        body.append(_emit_full_breakdown(rows))

    return "\n".join(body).rstrip() + "\n", worst_pct


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
            "If any gated measurement regressed by more than this percentage, "
            "exit non-zero. Default: don't fail regardless of size."
        ),
    )
    args = parser.parse_args()

    baseline = _load(args.baseline)
    head = _load(args.head)

    markdown, worst_pct = _emit_report(baseline, head, args.fail_on_regression_pct)
    print(markdown)

    if (
        args.fail_on_regression_pct is not None
        and worst_pct > args.fail_on_regression_pct
    ):
        print(
            f"\n**Regression gate failed**: worst gated regression "
            f"{_fmt_pct(worst_pct)} exceeds threshold "
            f"{_fmt_pct(args.fail_on_regression_pct)}.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
