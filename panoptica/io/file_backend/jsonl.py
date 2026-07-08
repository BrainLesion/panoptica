"""JSONL (JSON-lines) file backend."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from panoptica.core.errors import InputValidationError
from panoptica.io.file_backend.registry import FileBackend


def _iter_jsonl_records(path: Path) -> list[dict[str, Any]]:
    """Iterate over JSONL records, raising on malformed JSON."""
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    with open(path, encoding="utf8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as e:
                raise InputValidationError(
                    f"{path}: line {line_no}: malformed JSON: {e}"
                ) from e
    return records


def _append_jsonl_record(path: Path, record: dict[str, Any]) -> None:
    """Append a single JSON record to JSONL file."""
    with open(path, "a", encoding="utf8") as f:
        f.write(json.dumps(record) + "\n")


def _write_jsonl_records(path: Path, records: list[dict[str, Any]]) -> None:
    """Overwrite JSONL file with records."""
    with open(path, "w", encoding="utf8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def _canonical_jsonl_value(val: Any) -> Any:
    """Convert a value to its JSONL representation (None for NaN/inf, otherwise the value)."""
    if val is None:
        return None
    if isinstance(val, float):
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    return val


class JSONLBackend(FileBackend):
    """JSON-lines format.

    One nested JSON object per subject, with nested per-group summary and
    (optional) per-instance metrics.

    Each line looks like::

        {"subject_name": "subj_a",
         "groups": {"liver": {"dice": 0.9, "tp": 5.0,
                              "reference_instances": [{"dice": 0.95}, ...]}}}

    Missing values serialize as JSON ``null`` and round-trip to Python ``None``.
    The ``"reference_instances"`` key is only present when per-instance metrics
    are requested and the result yielded at least one reference instance.

    Crash recovery: a process killed mid-write can leave a truncated last line.
    Re-opening such a file raises ``InputValidationError``; delete the last
    partial line manually to recover.
    """

    def prepare_for_append(
        self,
        group_names: list[str],
        metric_names: list[str],
        collect_existing: bool = True,
    ) -> list[str]:
        """Prepare file for appending; return existing subject names."""
        expected_metrics = set(metric_names)
        expected_groups = set(group_names)

        if not self.path.exists():
            self.path.touch()
            return []

        existing: list[str] = []
        records = _iter_jsonl_records(self.path)

        for record in records:
            self._validate_record_schema(record, expected_groups, expected_metrics)
            if not collect_existing:
                continue
            sn = record.get("subject_name")
            if sn:
                existing.append(sn)

        return existing

    def _validate_record_schema(
        self,
        record: dict[str, Any],
        expected_groups: set[str],
        expected_metrics: set[str],
    ) -> None:
        """Validate that a record's schema matches expected groups/metrics."""
        existing_groups = set(record.get("groups", {}).keys())
        if existing_groups != expected_groups:
            raise InputValidationError(
                f"{self.path}: Schema mismatch. "
                f"Expected groups {sorted(expected_groups)}, "
                f"found {sorted(existing_groups)}."
            )

        for g, g_data in record.get("groups", {}).items():
            existing_metrics = {k for k in g_data.keys() if k != "reference_instances"}
            if existing_metrics != expected_metrics:
                raise InputValidationError(
                    f"{self.path}: Group {g!r} schema mismatch. "
                    f"Expected metrics {sorted(expected_metrics)}, "
                    f"found {sorted(existing_metrics)}."
                )

    def append_subject(
        self,
        subject_name: str,
        result_grouped: dict[str, object],
        group_names: list[str],
        metric_names: list[str],
        output_individual_instance_metrics: bool,
    ) -> None:
        """Append one subject's per-group results."""
        record: dict[str, Any] = {"subject_name": subject_name, "groups": {}}

        for groupname in group_names:
            result = result_grouped[groupname]
            if not hasattr(result, "to_dict"):
                raise InputValidationError(
                    f"Result object for group {groupname} does not implement "
                    "to_dict() method (Serializable protocol). "
                    "Cannot serialize to JSONL."
                )

            group_obj: dict[str, Any] = {}

            summary_dict = result.to_dict(output_individual_instance_metrics)  # type: ignore[attr-defined]
            instance_dicts: list[dict[str, Any]] = summary_dict.pop(
                "reference_instances", []
            )

            for m in metric_names:
                group_obj[m] = _canonical_jsonl_value(summary_dict.get(m))

            if instance_dicts:
                group_obj["reference_instances"] = [
                    {
                        m: _canonical_jsonl_value(inst_dict.get(m))
                        for m in metric_names
                        if inst_dict.get(m) is not None
                    }
                    for inst_dict in instance_dicts
                ]

            record["groups"][groupname] = group_obj

        _append_jsonl_record(self.path, record)

    def load_raw(
        self, verbose: bool = True
    ) -> tuple[list[str], dict[str, dict[str, list[float | None]]]]:
        """Read JSONL and return raw data for statistics."""
        records = _iter_jsonl_records(self.path)
        if not records:
            return [], {}

        # Extract metrics and groups from first record
        first_groups = records[0].get("groups", {})
        if not first_groups:
            return [], {}

        metric_names: list[str] = []
        for m in first_groups[list(first_groups.keys())[0]].keys():
            if m != "reference_instances":
                metric_names.append(m)
        group_names = list(first_groups.keys())

        if verbose:
            print(f"Loaded {len(records)} entries from {self.path}")
            print(f"Metrics: {metric_names}")
            print(f"Groups: {group_names}")

        subject_names: list[str] = []
        value_dict: dict[str, dict[str, list[float | None]]] = {
            g: {m: [] for m in metric_names} for g in group_names
        }

        for record in records:
            sn = record.get("subject_name", "")
            subject_names.append(sn)

            for g in group_names:
                g_data = record.get("groups", {}).get(g, {})
                for m in metric_names:
                    val = g_data.get(m)
                    value_dict[g][m].append(val if val is not None else None)

        return subject_names, value_dict

    def write_full(
        self,
        subject_names: list[str],
        value_dict: dict[str, dict[str, list[float | None]]],
        group_names: list[str],
        metric_names: list[str],
    ) -> None:
        """Overwrite JSONL with complete snapshot."""
        records: list[dict[str, Any]] = []

        for idx, subject_name in enumerate(subject_names):
            record: dict[str, Any] = {"subject_name": subject_name, "groups": {}}

            for g in group_names:
                group_obj: dict[str, Any] = {}
                for m in metric_names:
                    if g in value_dict and m in value_dict[g]:
                        if idx < len(value_dict[g][m]):
                            group_obj[m] = _canonical_jsonl_value(value_dict[g][m][idx])
                        else:
                            group_obj[m] = None
                    else:
                        group_obj[m] = None
                record["groups"][g] = group_obj

            records.append(record)

        _write_jsonl_records(self.path, records)
