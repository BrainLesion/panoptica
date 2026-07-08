"""TSV (tab-separated values) file backend."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from panoptica.core.errors import InputValidationError
from panoptica.io.file_backend.registry import FileBackend


def _read_first_tsv_row(path: Path) -> list[str]:
    """Read the first row of a TSV file."""
    try:
        with open(path, encoding="utf8", newline="") as f:
            reader = csv.reader(f, delimiter="\t", lineterminator="\n")
            first_row = next(reader, [])
            return first_row
    except StopIteration:
        return []


def _load_first_tsv_column(path: Path) -> list[str]:
    """Load the first column (subject names) from a TSV file."""
    try:
        with open(path, encoding="utf8", newline="") as f:
            reader = csv.reader(f, delimiter="\t", lineterminator="\n")
            return [row[0] for row in reader if row]
    except (IndexError, FileNotFoundError):
        return []


def _append_tsv_rows(path: Path, rows: list[list[Any]]) -> None:
    """Append rows to a TSV file."""
    with open(path, "a", encoding="utf8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerows(rows)


def _write_tsv_rows(path: Path, rows: list[list[Any]]) -> None:
    """Overwrite a TSV file with rows."""
    with open(path, "w", encoding="utf8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerows(rows)


def _canonical_tsv_value(val: Any) -> str:
    """Convert a value to its TSV representation (string, empty for None/NaN/inf)."""
    if val is None:
        return ""
    if isinstance(val, bool):
        return str(int(val))
    if isinstance(val, float):
        if np.isnan(val) or np.isinf(val):
            return ""
        return str(val)
    return str(val)


class TSVBackend(FileBackend):
    """Tab-separated values format.

    One row per subject (plus optional per-instance rows when
    output_individual_instance_metrics=True).

    Header is ``["subject_name", "{group}-{metric}", ...]``. Empty cells
    represent missing values and round-trip to ``None`` on read.
    """

    def prepare_for_append(
        self,
        group_names: list[str],
        metric_names: list[str],
        collect_existing: bool = True,
    ) -> list[str]:
        """Prepare file for appending; return existing subject names."""
        header = ["subject_name"] + [
            f"{g}-{m}" for g in group_names for m in metric_names
        ]

        if not self.path.exists():
            _append_tsv_rows(self.path, [header])
        else:
            existing_header = _read_first_tsv_row(self.path)
            if len(existing_header) == 0:
                _append_tsv_rows(self.path, [header])
            else:
                if header != existing_header:
                    raise InputValidationError(
                        f"{self.path}: Header mismatch. "
                        f"Expected {header}, found {existing_header}. "
                        "This file is incompatible with the current configuration."
                    )

        if not collect_existing:
            return []
        all_first_column = _load_first_tsv_column(self.path)
        return [s for s in all_first_column if s != "subject_name"]

    def append_subject(
        self,
        subject_name: str,
        result_grouped: dict[str, object],
        group_names: list[str],
        metric_names: list[str],
        output_individual_instance_metrics: bool,
    ) -> None:
        """Append one subject's per-group results."""
        content: list = [subject_name]
        for groupname in group_names:
            result = result_grouped[groupname]
            if not hasattr(result, "to_dict"):
                raise InputValidationError(
                    f"Result object for group {groupname} does not implement "
                    "to_dict() method (Serializable protocol). "
                    "Cannot serialize to TSV."
                )
            # Summary only; per-instance metrics excluded from the TSV.
            result_dict = result.to_dict(False)  # type: ignore[attr-defined]
            for m in metric_names:
                content.append(_canonical_tsv_value(result_dict.get(m)))

        _append_tsv_rows(self.path, [content])

    def load_raw(
        self, verbose: bool = True
    ) -> tuple[list[str], dict[str, dict[str, list[float | None]]]]:
        """Read TSV and return raw data for statistics."""
        with open(self.path, encoding="utf8", newline="") as f:
            reader = csv.reader(f, delimiter="\t", lineterminator="\n")
            rows = list(reader)

        if not rows:
            return [], {}

        header = rows[0]
        if header[0] != "subject_name":
            raise InputValidationError(
                f"{self.path}: First column must be 'subject_name', found {header[0]!r}."
            )

        # Parse header to extract (group, metric) pairs
        keys_in_order: list[tuple[str, str]] = []
        for col_name in header[1:]:
            parts = col_name.split("-", maxsplit=1)
            if len(parts) == 2:
                keys_in_order.append((parts[0], parts[1]))
            else:
                # Ungrouped metric (shouldn't happen in panoptica, but handle for compatibility)
                keys_in_order.append(("ungrouped", parts[0]))

        # Extract unique metric and group names preserving order
        metric_names: list[str] = []
        for _, m in keys_in_order:
            if m not in metric_names:
                metric_names.append(m)
        group_names = list(dict.fromkeys(g for g, _ in keys_in_order))

        if verbose:
            print(f"Loaded {len(rows) - 1} entries from {self.path}")
            print(f"Metrics: {metric_names}")
            print(f"Groups: {group_names}")

        if len(rows) == 1:
            # Header-only file
            return [], {}

        subject_names: list[str] = []
        value_dict: dict[str, dict[str, list[float | None]]] = {}

        for row_idx, row in enumerate(rows[1:]):
            subject_name = row[0] if row else ""
            subject_names.append(subject_name)

            for col_idx, value in enumerate(row[1:]):
                if col_idx >= len(keys_in_order):
                    continue
                group_name, metric_name = keys_in_order[col_idx]

                if group_name not in value_dict:
                    value_dict[group_name] = {m: [] for m in metric_names}

                if len(value) > 0:
                    try:
                        parsed = float(value)
                        if not (np.isnan(parsed) or np.isinf(parsed)):
                            value_dict[group_name][metric_name].append(parsed)
                        else:
                            value_dict[group_name][metric_name].append(None)
                    except ValueError as e:
                        raise InputValidationError(
                            f"{self.path}: row {row_idx}, column {col_idx + 1} "
                            f"({header[col_idx + 1]!r}): could not parse {value!r} as float"
                        ) from e
                else:
                    value_dict[group_name][metric_name].append(None)

        # Pad all metric lists to the same length (in case some had fewer entries)
        for group_name in value_dict:
            for metric_name in value_dict[group_name]:
                while len(value_dict[group_name][metric_name]) < len(subject_names):
                    value_dict[group_name][metric_name].append(None)

        return subject_names, value_dict

    def write_full(
        self,
        subject_names: list[str],
        value_dict: dict[str, dict[str, list[float | None]]],
        group_names: list[str],
        metric_names: list[str],
    ) -> None:
        """Overwrite TSV with complete snapshot."""
        header = ["subject_name"] + [
            f"{g}-{m}" for g in group_names for m in metric_names
        ]
        rows: list[list[Any]] = [header]

        for subject_name in subject_names:
            row: list[Any] = [subject_name]
            for g in group_names:
                if g not in value_dict:
                    value_dict[g] = {m: [] for m in metric_names}
                for m in metric_names:
                    if m not in value_dict[g]:
                        value_dict[g][m] = []
                    idx = subject_names.index(subject_name)
                    if idx < len(value_dict[g][m]):
                        val = value_dict[g][m][idx]
                    else:
                        val = None
                    row.append(_canonical_tsv_value(val))
            rows.append(row)

        _write_tsv_rows(self.path, rows)
