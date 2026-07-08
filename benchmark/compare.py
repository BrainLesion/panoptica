"""Compare two benchmark JSON files and emit a markdown delta table.

Both inputs are produced by :mod:`benchmark.bench_eval` via ``--json``. The output is
GitHub-flavoured markdown suitable for ``$GITHUB_STEP_SUMMARY``. When
``--fail-on-regression-pct`` is set, exits non-zero if any measurement's percent
change exceeds that threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any


def _load(path: str) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _cases_by_name(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {c["name"]: c for c in doc.get("cases", [])}


def _fmt_delta_pct(pct: float) -> str:
    marker = ""
    if pct >= 10:
        marker = " 🔴"
    elif pct <= -10:
        marker = " 🟢"
    return f"{pct:+.1f}%{marker}"


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
            # Only compare metrics present in both.
            continue
        delta = h - b
        pct = (delta / b * 100.0) if b > 0 else 0.0
        rows.append((key, b, h, delta, pct))
    return rows


# Sub-millisecond measurements are pure noise on shared CI runners; individual
# metric_* entries typically fall here. We surface them in the table but never
# use them to gate a PR.
GATE_MIN_BASELINE_MS = 1.0


def _is_gated(key: str, baseline_ms: float) -> bool:
    """Only workload-level measurements above the noise floor gate the PR."""
    if baseline_ms < GATE_MIN_BASELINE_MS:
        return False
    return not key.startswith("metric_")


def _emit_table(baseline: dict[str, Any], head: dict[str, Any]) -> tuple[str, float]:
    """Return (markdown, worst_gated_regression_pct)."""
    lines: list[str] = []
    lines.append(
        f"Baseline commit: `{baseline.get('commit', '?')}` · "
        f"Head commit: `{head.get('commit', '?')}` · "
        f"Python: `{head.get('python', '?')}`"
    )
    lines.append("")
    lines.append(
        f"Regression gate only considers non-metric measurements with baseline "
        f"≥ {GATE_MIN_BASELINE_MS:.0f} ms (others are shown for information)."
    )
    lines.append("")

    b_cases = _cases_by_name(baseline)
    h_cases = _cases_by_name(head)
    all_case_names = [c["name"] for c in head.get("cases", [])] + [
        n for n in b_cases if n not in {c["name"] for c in head.get("cases", [])}
    ]

    worst_pct = 0.0

    for name in all_case_names:
        if name not in b_cases or name not in h_cases:
            lines.append(f"### {name}\n\n_(present in only one document — skipping)_\n")
            continue

        rows = _diff_measurements(
            b_cases[name]["measurements_ms"],
            h_cases[name]["measurements_ms"],
        )
        lines.append(f"### {name}")
        lines.append("")
        lines.append("| Measurement | baseline ms | head ms | Δ ms | Δ % | gated |")
        lines.append("| --- | ---: | ---: | ---: | --- | :---: |")
        for key, b, h, delta, pct in rows:
            gated = _is_gated(key, b)
            if gated:
                worst_pct = max(worst_pct, pct)
            gate_mark = "✓" if gated else ""
            lines.append(
                f"| `{key}` | {b:.2f} | {h:.2f} | {delta:+.2f} | {_fmt_delta_pct(pct)} | {gate_mark} |"
            )
        lines.append("")

    return "\n".join(lines), worst_pct


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", help="Baseline JSON produced by bench_eval.py --json")
    parser.add_argument("head", help="Head JSON produced by bench_eval.py --json")
    parser.add_argument(
        "--fail-on-regression-pct",
        type=float,
        default=None,
        help=(
            "If any measurement regressed by more than this percentage, exit non-zero. "
            "Default: don't fail regardless of size."
        ),
    )
    args = parser.parse_args()

    baseline = _load(args.baseline)
    head = _load(args.head)

    markdown, worst_pct = _emit_table(baseline, head)
    print(markdown)

    if args.fail_on_regression_pct is not None and worst_pct > args.fail_on_regression_pct:
        print(
            f"\n**Regression gate failed**: worst regression {worst_pct:+.1f}% "
            f"exceeds threshold {args.fail_on_regression_pct:+.1f}%.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
