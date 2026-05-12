from __future__ import annotations

from pathlib import Path
from panoptica.utils import (
    FileBackend,
    format_instance_subject_name, 
    is_instance_row, 
    parse_instance_subject_name
)
from panoptica.panoptica_result import PanopticaAUTCResult, PanopticaResult
import numpy as np
import json


class JSONLBackend(FileBackend):
    """JSON-lines format. One nested JSON object per subject, with nested
    per-group summary and (optional) matched-instance metrics.

    Each line looks like::

        {"subject_name": "subj_a",
         "groups": {"liver": {"dice": 0.9, "tp": 5.0,
                              "instances": [{"sq_dice": 0.95}, ...]}}}

    Missing values serialize as JSON ``null`` and round-trip to Python
    ``None``. The ``"instances"`` key is only present when the aggregator
    is run with ``output_individual_instance_metrics=True`` and the result
    yielded at least one matched instance.
    """

    def prepare_for_append(
        self,
        class_group_names: list[str],
        evaluation_metrics: list[str],
    ) -> list[str]:
        expected_metrics = set(evaluation_metrics)
        expected_groups = set(class_group_names)

        if not self.path.exists():
            self.path.touch()
            return []

        first_record = _read_first_jsonl_record(self.path)
        if first_record is None:
            print(
                f"{self.path}: Output file given is empty, will start with first subject"
            )
            return []

        existing_groups = set(first_record.get("groups", {}).keys())
        if existing_groups != expected_groups:
            raise ValueError(
                f"{self.path}: schema of existing file does not match current evaluator setup! "
                f"Expected groups {sorted(expected_groups)}, found {sorted(existing_groups)}."
            )
        for g in first_record.get("groups", {}):
            existing_metrics = {
                k for k in first_record["groups"][g].keys() if k != "instances"
            }
            if existing_metrics != expected_metrics:
                raise ValueError(
                    f"{self.path}: schema of existing file does not match current evaluator setup! "
                    f"Group {g!r}: expected metrics {sorted(expected_metrics)}, "
                    f"found {sorted(existing_metrics)}."
                )

        existing: list[str] = []
        for record in _iter_jsonl_records(self.path):
            sn = record["subject_name"]
            existing.append(sn)
            for g, g_data in record.get("groups", {}).items():
                inst_list = g_data.get("instances")
                if inst_list:
                    for inst_idx in range(len(inst_list)):
                        existing.append(
                            format_instance_subject_name(sn, g, inst_idx)
                        )
        return existing

    def append_subject(
        self,
        subject_name: str,
        result_grouped: dict[str, "PanopticaResult | PanopticaAUTCResult"],
        class_group_names: list[str],
        evaluation_metrics: list[str],
        output_individual_instance_metrics: bool,
    ) -> None:
        record: dict = {"subject_name": subject_name, "groups": {}}
        for groupname in class_group_names:
            result = result_grouped[groupname]
            group_obj: dict = {}
            if output_individual_instance_metrics:
                rows_as_dicts = result.to_dict(True)
                summary_dict = rows_as_dicts[0] if len(rows_as_dicts) > 0 else {}
                instance_dicts = list(rows_as_dicts[1:])
            else:
                summary_dict = result.to_dict(False)
                instance_dicts = []

            if result.computation_time is not None:
                summary_dict = dict(summary_dict)
                summary_dict[COMPUTATION_TIME_KEY] = result.computation_time

            for e in evaluation_metrics:
                group_obj[e] = _canonical_jsonl_value(summary_dict.get(e))

            if instance_dicts:
                # Restrict to evaluation_metrics so the JSONL inst-dict set
                # matches what TSV columns can hold — required for symmetric
                # TSV<->JSONL roundtrip byte-identity. Drop None entries so
                # the inst-dict shape matches what write_full produces from a
                # roundtripped Panoptica_Statistic.
                group_obj["instances"] = [
                    {
                        e: _canonical_jsonl_value(r.get(e))
                        for e in evaluation_metrics
                        if r.get(e) is not None
                    }
                    for r in instance_dicts
                ]
            record["groups"][groupname] = group_obj

        _append_jsonl_record(self.path, record)
        print(f"Saved entry {subject_name} into {self.path}")

    def write_full(
        self,
        subj_names: list[str],
        value_dict: dict[str, dict[str, list[float | None]]],
        class_group_names: list[str],
        evaluation_metrics: list[str],
    ) -> None:
        master_indices: list[int] = []
        instances_by_master: dict[int, dict[str, list[dict]]] = {}
        for i, sn in enumerate(subj_names):
            if is_instance_row(sn):
                parsed = parse_instance_subject_name(sn)
                if parsed is None:
                    continue
                orig_subj, orig_group, inst_idx = parsed
                try:
                    master_i = subj_names.index(orig_subj)
                except ValueError:
                    continue
                inst_dict = {
                    m: value_dict[orig_group][m][i]
                    for m in evaluation_metrics
                    if value_dict[orig_group][m][i] is not None
                }
                instances_by_master.setdefault(master_i, {}).setdefault(
                    orig_group, []
                ).append(inst_dict)
            else:
                master_indices.append(i)

        records: list[dict] = []
        for i in master_indices:
            record: dict = {"subject_name": subj_names[i], "groups": {}}
            for g in class_group_names:
                group_obj: dict = {}
                for m in evaluation_metrics:
                    group_obj[m] = _canonical_jsonl_value(value_dict[g][m][i])
                inst_list = instances_by_master.get(i, {}).get(g)
                if inst_list:
                    group_obj["instances"] = inst_list
                record["groups"][g] = group_obj
            records.append(record)

        _write_jsonl_records(self.path, records)

    def load_raw(
        self, verbose: bool = True
    ) -> tuple[list[str], dict[str, dict[str, list[float | None]]]]:
        records = list(_iter_jsonl_records(self.path))
        if not records:
            return [], {}

        first = records[0]
        group_names = list(first["groups"].keys())
        metric_names: list[str] = []
        for g in group_names:
            for k in first["groups"][g].keys():
                if k == "instances":
                    continue
                if k not in metric_names:
                    metric_names.append(k)

        if verbose:
            print(f"Found {len(records)} entries")
            print(f"Found metrics: {metric_names}")
            print(f"Found groups: {group_names}")

        subj_names: list[str] = []
        value_dict: dict[str, dict[str, list[float | None]]] = {
            g: {m: [] for m in metric_names} for g in group_names
        }

        for record in records:
            sn = record["subject_name"]
            subj_names.append(sn)
            for g in group_names:
                g_data = record["groups"][g]
                for m in metric_names:
                    value_dict[g][m].append(_parse_jsonl_value(g_data.get(m)))

            for g in group_names:
                inst_list = record["groups"][g].get("instances") or []
                for inst_idx, inst_dict in enumerate(inst_list):
                    subj_names.append(format_instance_subject_name(sn, g, inst_idx))
                    for inner_g in group_names:
                        for m in metric_names:
                            if inner_g == g:
                                value_dict[inner_g][m].append(
                                    _parse_jsonl_value(inst_dict.get(m))
                                )
                            else:
                                value_dict[inner_g][m].append(None)

        return subj_names, value_dict


def _canonical_jsonl_value(v):
    """Canonicalize a value for JSONL output. Symmetric with the TSV path:
    numerics cast to ``float``; ``None`` / NaN / Inf serialize as JSON
    ``null``."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        f = float(v)
        if np.isnan(f) or f == np.inf:
            return None
        return f
    return v


def _parse_jsonl_value(v) -> float | None:
    """Inverse of ``_canonical_jsonl_value``: JSON ``null`` and NaN/Inf
    map to ``None`` (matching TSV semantics where empty cells round-trip
    to ``None``)."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        f = float(v)
        if np.isnan(f) or f == np.inf:
            return None
        return f
    return v

def _read_first_jsonl_record(path: Path) -> dict | None:
    """Reads the first JSON record from a JSONL file, skipping blank lines.
    Returns ``None`` for an empty file. NOT THREAD SAFE BY ITSELF."""
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    return None


def _iter_jsonl_records(path: Path):
    """Yields parsed JSON records from a JSONL file, skipping blank lines.
    NOT THREAD SAFE BY ITSELF."""
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def _append_jsonl_record(path: Path, record: dict) -> None:
    """Appends a single JSON record as one line to a JSONL file.
    NOT THREAD SAFE BY ITSELF."""
    with open(path, "a", encoding="utf8") as f:
        f.write(json.dumps(record, ensure_ascii=False, separators=(", ", ": ")))
        f.write("\n")


def _write_jsonl_records(path: Path, records: list[dict]) -> None:
    """Writes all records as JSONL, overwriting any existing content.
    NOT THREAD SAFE BY ITSELF."""
    with open(path, "w", encoding="utf8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, separators=(", ", ": ")))
            f.write("\n")
