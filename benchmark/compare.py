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


def _stats(raw: Any) -> dict[str, float]:
    """Normalize a measurement value to ``{min, median, p90}``.

    New-schema entries are dicts; legacy entries are bare floats and get treated
    as a degenerate distribution with zero spread.
    """
    if isinstance(raw, dict):
        # Newer JSON — all three should be present.
        return {
            "min": float(raw.get("min", raw.get("median", 0.0))),
            "median": float(raw.get("median", raw.get("min", 0.0))),
            "p90": float(raw.get("p90", raw.get("median", raw.get("min", 0.0)))),
        }
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


def _is_regression(row: Row, threshold_pct: float) -> bool:
    """Real regression: worse than the threshold AND outside the baseline noise band."""
    if not row.both_present:
        return False
    assert row.b is not None and row.h is not None
    b_med = row.b["median"]
    if b_med <= 0:
        return False
    return (
        row.h["median"] > b_med * (1.0 + threshold_pct / 100.0)
        and row.h["median"] > row.b["p90"]
    )


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
) -> str:
    gate_badge = "✅ PASS" if gate_pass else "🔴 FAIL"
    repeats = head.get("repeats", "?")
    warmup = head.get("warmup", "?")
    parts = [
        MARKER,
        "",
        f"## 📊 Benchmark {head.get('commit', '?')} vs `{baseline.get('commit', '?')}`",
        "",
        (
            f"**Gate:** {gate_badge} &nbsp;·&nbsp; "
            f"🟢 {wins} win{'s' if wins != 1 else ''} &nbsp;·&nbsp; "
            f"🔴 {regressions} regression{'s' if regressions != 1 else ''} &nbsp;·&nbsp; "
            f"Python {head.get('python', '?')} &nbsp;·&nbsp; "
            f"repeats={repeats}, warmup={warmup}"
        ),
        "",
        (
            "> Baseline is measured on the same runner (main's `panoptica` + this PR's benchmark harness). "
            "The committed `benchmark/baseline.json` is only a fallback for `workflow_dispatch` runs."
        ),
        "",
        (
            "> Values are `median ±(p90−min)/2`. A row counts as a regression only when "
            "head's median exceeds both the threshold and the baseline's p90."
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
    return (
        f"> 🚨 **Regression gate FAILED** — `{worst_row.key}` in `{offender_case}` "
        f"regressed by `{_fmt_pct(worst_row.pct)}` (baseline median {worst_row.b['median']:.2f} ms, "
        f"p90 {worst_row.b['p90']:.2f}; head median {worst_row.h['median']:.2f} ms). "
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


def _fmt_row(row: Row) -> str:
    if row.only_in_head:
        assert row.h is not None
        return (
            f"| `{row.key}` | {_MISSING} | {_fmt_ms_with_spread(row.h)} "
            f"| _new in head_ |"
        )
    if row.only_in_baseline:
        assert row.b is not None
        return (
            f"| `{row.key}` | {_fmt_ms_with_spread(row.b)} | {_MISSING} "
            f"| _absent in head_ |"
        )
    assert row.b is not None and row.h is not None
    return (
        f"| `{row.key}` | {_fmt_ms_with_spread(row.b)} | {_fmt_ms_with_spread(row.h)} "
        f"| {_fmt_pct(row.pct)}{_pct_marker(row.pct)} |"
    )


def _emit_key_table(rows: list[Row], max_rows: int = KEY_TABLE_MAX_ROWS) -> str:
    # Key table only shows comparable rows — head-only ones belong in the breakdown.
    gated_rows = [r for r in rows if _is_gated(r)]
    gated_rows = [r for r in gated_rows if abs(r.delta) >= KEY_TABLE_MIN_DELTA_MS]
    gated_rows.sort(key=lambda r: abs(r.pct), reverse=True)
    if not gated_rows:
        return "_No gated measurements moved meaningfully._\n\n"

    top = gated_rows[:max_rows]
    lines = [
        "| Measurement | baseline ms (median ±½·range) | head ms (median ±½·range) | Δ % |",
        "| --- | ---: | ---: | :--- |",
    ]
    for row in top:
        lines.append(_fmt_row(row))
    lines.append("")
    return "\n".join(lines)


def _emit_full_breakdown(rows: list[Row]) -> str:
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
        "| Measurement | baseline ms (median ±½·range) | head ms (median ±½·range) | Δ % |",
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
            assert row.b is not None and row.h is not None
            # Only count wins/regressions when the two distributions are cleanly
            # separated — head's worst sample must beat baseline's best (win) or
            # head's best sample must be worse than baseline's worst (regression).
            # Anything softer just counts noise and disagrees with the gate.
            if row.pct >= 10 and row.h["min"] > row.b["p90"]:
                regressions += 1
            elif row.pct <= -10 and row.h["p90"] < row.b["min"]:
                wins += 1
            # Track the worst *real* regression relative to the noise band, not just
            # the largest %. A row with pct=+80% but inside baseline p90 isn't worse
            # than one with pct=+55% that's clearly outside it.
            if fail_threshold is not None and _is_regression(row, fail_threshold):
                if worst_row is None or row.pct > worst_row.pct:
                    worst_row = row
                    worst_case = name

    gate_pass = fail_threshold is None or worst_row is None

    body: list[str] = []
    body.append(
        _emit_header(baseline, head, gate_pass, wins, regressions, head_only_total)
    )
    if not gate_pass and fail_threshold is not None:
        body.append(_emit_gate_callout(worst_row, fail_threshold, worst_case))

    for name, rows in per_case:
        if not rows:
            body.append(f"### {name}\n\n_(present in only one document — skipping)_\n")
            continue
        body.append(_emit_case_hero(name, rows))
        body.append(_emit_key_table(rows))
        body.append(_emit_full_breakdown(rows))

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
            "this percentage AND exceeds baseline p90. Default: never fail."
        ),
    )
    args = parser.parse_args()

    baseline = _load(args.baseline)
    head = _load(args.head)

    markdown, gate_pass, worst_row, worst_case = _emit_report(
        baseline, head, args.fail_on_regression_pct
    )
    print(markdown)

    if not gate_pass and worst_row is not None:
        print(
            f"\n**Regression gate failed**: `{worst_row.key}` in "
            f"`{worst_case}` regressed by {_fmt_pct(worst_row.pct)} "
            f"(baseline p90 {worst_row.b['p90']:.2f} ms, head median "
            f"{worst_row.h['median']:.2f} ms).",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
