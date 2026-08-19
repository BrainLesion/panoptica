"""Building blocks for exporting evaluation results as LaTeX tables.

The helpers here are deliberately independent of ``Panoptica_Statistic``: they take
plain strings and numbers, so they can be reused for any table. The statistic-facing
entry points live in :mod:`panoptica.panoptica_statistics`
(``Panoptica_Statistic.get_latex_table`` and ``make_latex_table_over_setups``).
"""

import re
from collections.abc import Sequence
from typing import Literal

import numpy as np

from panoptica.utils.logger import logger
from panoptica.utils.serialization import parse_autc_key, parse_threshold_key

MetricDirection = Literal["increasing", "decreasing"]

LATEX_TABLE_REQUIREMENTS = r"% Requires \usepackage{booktabs,tabularx,array}"

# Column prefixes that PanopticaResult puts in front of an underlying Metric name.
# Order matters: the longest/most specific prefix has to be tried first.
_METRIC_NAME_PREFIXES = ("global_bin_", "region_avg_", "sq_", "pq_")

# Direction of the metrics that are derived inside PanopticaResult and therefore have
# no entry in the Metric enum.
_DERIVED_METRIC_DIRECTIONS: dict[str, MetricDirection] = {
    "pq": "increasing",
    "sq": "increasing",
    "rq": "increasing",
    "prec": "increasing",
    "rec": "increasing",
    "tp": "increasing",
    "fp": "decreasing",
    "fn": "decreasing",
    "computation_time": "decreasing",
}

# Tokens that should keep their canonical casing in a printed label.
_LABEL_TOKEN_OVERRIDES = {
    "pq": "PQ",
    "sq": "SQ",
    "rq": "RQ",
    "dsc": "DSC",
    "cldsc": "clDSC",
    "iou": "IoU",
    "assd": "ASSD",
    "hd": "HD",
    "hd95": "HD95",
    "nsd": "NSD",
    "rvd": "RVD",
    "rvae": "RVAE",
    "cedi": "CEDI",
    "tp": "TP",
    "fp": "FP",
    "fn": "FN",
    "autc": "AUTC",
    "std": "Std",
    "n": "N",
    "bin": "Bin",
    "avg": "Avg",
    "ref": "Ref",
    "pred": "Pred",
    "prec": "Precision",
    "rec": "Recall",
}

_LATEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}

# One pass over the string, otherwise the braces that \textbackslash{} introduces would
# be escaped again by the "{" / "}" rules.
_LATEX_ESCAPE_PATTERN = re.compile("|".join(re.escape(c) for c in _LATEX_ESCAPES))

_direction_map_cache: dict[str, MetricDirection] | None = None


def escape_latex(text: str) -> str:
    """Escapes the characters that LaTeX treats specially.

    Args:
        text (str): Raw text, e.g. a metric or group name.

    Returns:
        str: The same text, safe to paste into a LaTeX document.
    """
    return _LATEX_ESCAPE_PATTERN.sub(
        lambda match: _LATEX_ESCAPES[match.group()], str(text)
    )


def _metric_direction_map() -> dict[str, MetricDirection]:
    """Builds (once) the base-metric-name to direction lookup.

    The ``Metric`` enum is imported lazily because ``panoptica.metrics`` imports from
    ``panoptica.utils``; importing it at module level would close that cycle.
    """
    global _direction_map_cache
    if _direction_map_cache is None:
        from panoptica.metrics import Metric

        direction_map: dict[str, MetricDirection] = {
            m.value.name.lower(): ("decreasing" if m.decreasing else "increasing")
            for m in Metric
        }
        direction_map.update(_DERIVED_METRIC_DIRECTIONS)
        _direction_map_cache = direction_map
    return _direction_map_cache


def base_metric_name(metric: str) -> str:
    """Reduces a result column name to the underlying metric name.

    Strips the AUTC / matching-threshold wrappers and the ``sq_``, ``pq_``,
    ``global_bin_`` and ``region_avg_`` prefixes, so that e.g. ``t0.5_global_bin_hd95``
    becomes ``hd95``.

    Args:
        metric (str): Column name as stored in a ``Panoptica_Statistic``.

    Returns:
        str: The bare metric name (unchanged if nothing matched).
    """
    name = metric
    autc_base = parse_autc_key(name)
    if autc_base is not None:
        name = autc_base
    threshold_parsed = parse_threshold_key(name)
    if threshold_parsed is not None:
        name = threshold_parsed[1]
    for prefix in _METRIC_NAME_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    return name


def metric_direction(metric: str) -> MetricDirection | None:
    """Infers whether higher or lower values of a metric are better.

    Resolution order: standard deviation columns have no direction; otherwise the name
    is reduced via :func:`base_metric_name` and looked up in a table built from the
    ``Metric`` enum's ``decreasing`` flag plus the metrics derived inside
    ``PanopticaResult``.

    Args:
        metric (str): Column name as stored in a ``Panoptica_Statistic``.

    Returns:
        MetricDirection | None: ``"increasing"`` if higher is better, ``"decreasing"``
        if lower is better, or ``None`` when the direction is unknown. Callers should
        treat ``None`` as "do not draw an arrow and do not mark a best value".
    """
    if metric.endswith("_std"):
        return None
    direction = _metric_direction_map().get(base_metric_name(metric))
    if direction is None:
        logger.debug(f"Unknown metric direction for {metric}, no arrow will be drawn")
    return direction


def metric_arrow(metric: str) -> str:
    """Returns the LaTeX arrow indicating the preferred direction of a metric.

    Args:
        metric (str): Column name as stored in a ``Panoptica_Statistic``.

    Returns:
        str: ``"$\\uparrow$"``, ``"$\\downarrow$"`` or an empty string.
    """
    direction = metric_direction(metric)
    if direction == "increasing":
        return r"$\uparrow$"
    if direction == "decreasing":
        return r"$\downarrow$"
    return ""


def prettify_metric_name(metric: str) -> str:
    """Turns a raw column name into a human readable, LaTeX-safe label.

    ``sq_dsc`` becomes ``SQ DSC``, ``n_ref_instances`` becomes ``N Ref Instances``.

    Args:
        metric (str): Column name as stored in a ``Panoptica_Statistic``.

    Returns:
        str: Escaped label ready to be placed in a table header.
    """
    tokens = [t for t in metric.split("_") if t != ""]
    pretty = " ".join(
        _LABEL_TOKEN_OVERRIDES.get(t.lower(), t.capitalize()) for t in tokens
    )
    return escape_latex(pretty if pretty else metric)


def prettify_group_name(group: str) -> str:
    """Turns a raw group name into a human readable, LaTeX-safe label.

    Args:
        group (str): Group name as stored in a ``Panoptica_Statistic``.

    Returns:
        str: Escaped label ready to be placed in a table.
    """
    tokens = [t for t in group.split("_") if t != ""]
    pretty = " ".join(t.capitalize() for t in tokens)
    return escape_latex(pretty if pretty else group)


def _is_missing(value: float | None) -> bool:
    return value is None or not np.isfinite(value)


def format_mean_std(
    avg: float | None,
    std: float | None = None,
    ndigits: int = 3,
    show_std: bool = True,
    bold: bool = False,
    empty_placeholder: str = "--",
) -> str:
    """Formats one table cell as ``mean`` or ``mean ± std``.

    Args:
        avg (float | None): The mean value, or ``None`` / NaN when missing.
        std (float | None, optional): The standard deviation. Defaults to None.
        ndigits (int, optional): Digits after the decimal point. Defaults to 3.
        show_std (bool, optional): Whether to append ``\\pm std``. Defaults to True.
        bold (bool, optional): Whether to highlight the mean with ``\\mathbf``.
            Defaults to False.
        empty_placeholder (str, optional): Rendered when the mean is missing.
            Defaults to ``"--"``.

    Returns:
        str: A LaTeX math-mode cell, e.g. ``$0.842 \\pm 0.113$``.
    """
    if _is_missing(avg):
        return empty_placeholder
    mean_str = f"{avg:.{ndigits}f}"
    if bold:
        mean_str = r"\mathbf{" + mean_str + "}"
    if show_std and not _is_missing(std):
        return f"${mean_str} \\pm {std:.{ndigits}f}$"
    return f"${mean_str}$"


def select_best_indices(
    values: Sequence[float | None], direction: MetricDirection | None
) -> set[int]:
    """Finds the positions holding the best value of a series.

    Args:
        values (Sequence[float | None]): Candidate values; ``None`` and non-finite
            entries are ignored.
        direction (MetricDirection | None): Which end counts as best. ``None`` means
            the direction is unknown, in which case nothing is selected.

    Returns:
        set[int]: Indices of the best value. All tied positions are returned; the set
        is empty when ``direction`` is ``None`` or no value is usable.
    """
    if direction is None:
        return set()
    # Built with an explicit loop so the None/NaN entries are narrowed away.
    usable: list[tuple[int, float]] = []
    for i, v in enumerate(values):
        if not _is_missing(v):
            usable.append((i, float(v)))  # type: ignore[arg-type]
    if len(usable) == 0:
        return set()
    best = (
        max(v for _, v in usable)
        if direction == "increasing"
        else min(v for _, v in usable)
    )
    return {i for i, v in usable if v == best}


def render_latex_table(
    column_headers: list[str],
    rows: list[tuple[str, list[str]]],
    corner_label: str = "",
    column_groups: list[tuple[str, int]] | None = None,
    caption: str | None = None,
    label: str | None = None,
    use_booktabs: bool = True,
    use_tabularx: bool = True,
    tabularx_width: str = r"\textwidth",
    wrap_in_table_env: bool = True,
    position: str = "t",
    requirement_comment: bool = True,
) -> str:
    """Renders already-formatted cells as a LaTeX table.

    Args:
        column_headers (list[str]): Header of each data column (LaTeX-ready).
        rows (list[tuple[str, list[str]]]): One ``(row label, cells)`` pair per body
            row. Every cell list must have the same length as ``column_headers``.
        corner_label (str, optional): Content of the top-left cell. Defaults to "".
        column_groups (list[tuple[str, int]] | None, optional): ``(label, span)`` pairs
            drawn as a spanning header row above ``column_headers``. The spans must add
            up to the number of columns. Defaults to None.
        caption (str | None, optional): ``\\caption`` text. Defaults to None.
        label (str | None, optional): ``\\label`` key. Defaults to None.
        use_booktabs (bool, optional): Use ``\\toprule``/``\\midrule``/``\\bottomrule``
            instead of ``\\hline``. Defaults to True.
        use_tabularx (bool, optional): Use a ``tabularx`` environment stretched to
            ``tabularx_width`` instead of a plain ``tabular``. Defaults to True.
        tabularx_width (str, optional): Width of the tabularx. Defaults to
            ``"\\textwidth"``.
        wrap_in_table_env (bool, optional): Wrap everything in a floating ``table``
            environment. Defaults to True.
        position (str, optional): Float placement specifier. Defaults to "t".
        requirement_comment (bool, optional): Prepend a comment naming the LaTeX
            packages the snippet needs. Only emitted when they are actually used.
            Defaults to True.

    Raises:
        ValueError: If a row has the wrong number of cells, or the column group spans
            do not add up to the number of columns.

    Returns:
        str: The complete LaTeX snippet.
    """
    n_cols = len(column_headers)
    for row_label, cells in rows:
        if len(cells) != n_cols:
            raise ValueError(
                f"Row {row_label!r} has {len(cells)} cells but there are {n_cols} columns"
            )
    if column_groups is not None:
        span_total = sum(span for _, span in column_groups)
        if span_total != n_cols:
            raise ValueError(
                f"column_groups spans add up to {span_total} but there are {n_cols} columns"
            )

    if use_tabularx:
        col_spec = "l" + r">{\centering\arraybackslash}X" * n_cols
        begin_tabular = f"\\begin{{tabularx}}{{{tabularx_width}}}{{{col_spec}}}"
        end_tabular = r"\end{tabularx}"
    else:
        begin_tabular = f"\\begin{{tabular}}{{l{'c' * n_cols}}}"
        end_tabular = r"\end{tabular}"

    top_rule = r"\toprule" if use_booktabs else r"\hline"
    mid_rule = r"\midrule" if use_booktabs else r"\hline"
    bottom_rule = r"\bottomrule" if use_booktabs else r"\hline"

    lines: list[str] = []
    if requirement_comment and (use_booktabs or use_tabularx):
        lines.append(LATEX_TABLE_REQUIREMENTS)
    if wrap_in_table_env:
        lines.append(f"\\begin{{table}}[{position}]")
        lines.append(r"\centering")
        if caption is not None:
            lines.append(f"\\caption{{{caption}}}")
        if label is not None:
            lines.append(f"\\label{{{label}}}")
    lines.append(begin_tabular)
    lines.append(top_rule)

    if column_groups is not None:
        group_cells = [
            f"\\multicolumn{{{span}}}{{c}}{{{group_label}}}"
            for group_label, span in column_groups
        ]
        lines.append(" & ".join([corner_label] + group_cells) + r" \\")
        # Underline each spanning label; column 1 holds the row labels.
        rules = []
        first = 2
        for _, span in column_groups:
            last = first + span - 1
            rules.append(
                f"\\cmidrule(lr){{{first}-{last}}}"
                if use_booktabs
                else f"\\cline{{{first}-{last}}}"
            )
            first = last + 1
        lines.append(" ".join(rules))
        lines.append(" & ".join([""] + column_headers) + r" \\")
    else:
        lines.append(" & ".join([corner_label] + column_headers) + r" \\")

    lines.append(mid_rule)
    for row_label, cells in rows:
        lines.append(" & ".join([row_label] + cells) + r" \\")
    lines.append(bottom_rule)
    lines.append(end_tabular)
    if wrap_in_table_env:
        lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"
