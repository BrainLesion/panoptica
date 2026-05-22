from __future__ import annotations

from pathlib import Path

from panoptica.utils.serialization import (
    format_instance_subject_name,
    is_instance_row,
    parse_instance_subject_name,
)
from panoptica.utils.file_backend import FileBackend
from panoptica.panoptica_result import PanopticaAUTCResult, PanopticaResult
from panoptica.utils.file_backend import COMPUTATION_TIME_KEY
import numpy as np
import json


class JSONLBackend(FileBackend):
    """JSON-lines format. One nested JSON object per subject, with nested
    per-group summary and (optional) matched-instance metrics.

    Each line looks like::

        {"subject_name": "subj_a",
         "groups": {"liver": {"dice": 0.9, "tp": 5.0,
                              "reference_instances": [{"sq_dice": 0.95}, ...]}}}

    Missing values serialize as JSON ``null`` and round-trip to Python
    ``None``. The ``"reference_instances"`` key is only present when the
    aggregator is run with ``output_individual_instance_metrics=True`` and
    the result yielded at least one reference instance.

    Crash recovery: a process killed mid-write can leave a truncated last
    line. Re-opening such a file raises ``ValueError("malformed JSON on
    line N")``; delete the last partial line manually to recover.
    """

    def prepare_for_append(
        self,
        class_group_names: list[str],
        evaluation_metrics: list[str],
        collect_existing: bool = True,
    ) -> list[str]:
        expected_metrics = set(evaluation_metrics)
        expected_groups = set(class_group_names)

        if not self.path.exists():
            self.path.touch()
            return []

        existing: list[str] = []
        seen_any = False
        for record in _iter_jsonl_records(self.path):
            seen_any = True
            self._validate_record_schema(record, expected_groups, expected_metrics)
            if not collect_existing:
                # Skip subject-name collection but keep iterating so every record gets schema-validated
                continue
            sn = record["subject_name"]
            existing.append(sn)
            for g, g_data in record.get("groups", {}).items():
                inst_list = g_data.get("reference_instances")
                if inst_list:
                    for inst_idx in range(len(inst_list)):
                        existing.append(format_instance_subject_name(sn, g, inst_idx))

        if not seen_any:
            print(
                f"{self.path}: Output file given is empty, will start with first subject"
            )
        return existing

    def _validate_record_schema(
        self,
        record: dict,
        expected_groups: set[str],
        expected_metrics: set[str],
    ) -> None:
        """Raises ValueError if a record's groups or per-group metric set
        diverges from the evaluator setup. Run on every record so mid-file
        schema drift (e.g. from hand-edits or concatenated runs) is caught,
        not just the first row."""
        existing_groups = set(record.get("groups", {}).keys())
        if existing_groups != expected_groups:
            raise ValueError(
                f"{self.path}: schema of existing file does not match current evaluator setup! "
                f"Expected groups {sorted(expected_groups)}, found {sorted(existing_groups)}."
            )
        for g, g_data in record.get("groups", {}).items():
            existing_metrics = {k for k in g_data.keys() if k != "reference_instances"}
            if existing_metrics != expected_metrics:
                raise ValueError(
                    f"{self.path}: schema of existing file does not match current evaluator setup! "
                    f"Group {g!r}: expected metrics {sorted(expected_metrics)}, "
                    f"found {sorted(existing_metrics)}."
                )

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
                summary_dict = result.to_dict(True)
                instance_dicts = summary_dict.pop("reference_instances", [])
            else:
                summary_dict = result.to_dict(False)
                instance_dicts = []

            if result.computation_time is not None:
                summary_dict[COMPUTATION_TIME_KEY] = result.computation_time

            for e in evaluation_metrics:
                group_obj[e] = _canonical_jsonl_value(summary_dict.get(e))

            if instance_dicts:
                # Row keys are normalize row keys to master keys so they match the JSONL/TSV schema.
                group_obj["reference_instances"] = [
                    {
                        e: _canonical_jsonl_value(r_norm.get(e))
                        for e in evaluation_metrics
                        if r_norm.get(e) is not None
                    }
                    for r_norm in (
                        PanopticaResult.normalize_row_to_master_schema(r)
                        for r in instance_dicts
                    )
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
        # Map subject name -> position in subj_names, so the per-instance master lookup is O(1)
        name_to_index = {sn: i for i, sn in enumerate(subj_names)}

        master_indices: list[int] = []
        # {master_idx: {group: {inst_idx: inst_dict}}}; using an inner dict
        # keyed by inst_idx makes ordering depend on the parsed index, not
        # on the encounter order of subj_names.
        instances_by_master: dict[int, dict[str, dict[int, dict]]] = {}
        for i, sn in enumerate(subj_names):
            if is_instance_row(sn):
                parsed = parse_instance_subject_name(sn)
                if parsed is None:
                    continue
                orig_subj, orig_group, inst_idx = parsed
                master_i = name_to_index.get(orig_subj)
                if master_i is None:
                    continue
                inst_dict = {}
                for m in evaluation_metrics:
                    canonical = _canonical_jsonl_value(value_dict[orig_group][m][i])
                    if canonical is not None:
                        inst_dict[m] = canonical
                instances_by_master.setdefault(master_i, {}).setdefault(orig_group, {})[
                    inst_idx
                ] = inst_dict
            else:
                master_indices.append(i)

        records: list[dict] = []
        for i in master_indices:
            record: dict = {"subject_name": subj_names[i], "groups": {}}
            for g in class_group_names:
                group_obj: dict = {}
                for m in evaluation_metrics:
                    group_obj[m] = _canonical_jsonl_value(value_dict[g][m][i])
                inst_by_idx = instances_by_master.get(i, {}).get(g)
                if inst_by_idx:
                    group_obj["reference_instances"] = [
                        inst_by_idx[k] for k in sorted(inst_by_idx)
                    ]
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
                if k == "reference_instances":
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

        expected_groups = set(group_names)
        expected_metrics = set(metric_names)
        for record_idx, record in enumerate(records):
            sn = record["subject_name"]
            subj_names.append(sn)
            record_groups = record.get("groups", {})
            actual_groups = set(record_groups.keys())
            if actual_groups != expected_groups:
                missing = expected_groups - actual_groups
                extra = actual_groups - expected_groups
                raise ValueError(
                    f"{self.path}: record {record_idx} (subject {sn!r}) "
                    f"group set diverges from record 0. "
                    f"Missing: {sorted(missing)}, extra: {sorted(extra)}."
                )
            for g in group_names:
                actual_metrics = {
                    k for k in record_groups[g].keys() if k != "reference_instances"
                }
                if actual_metrics != expected_metrics:
                    missing_m = expected_metrics - actual_metrics
                    extra_m = actual_metrics - expected_metrics
                    raise ValueError(
                        f"{self.path}: record {record_idx} (subject {sn!r}) "
                        f"group {g!r} metric set diverges from record 0. "
                        f"Missing: {sorted(missing_m)}, extra: {sorted(extra_m)}."
                    )
            for g in group_names:
                g_data = record_groups[g]
                for m in metric_names:
                    value_dict[g][m].append(_parse_jsonl_value(g_data.get(m)))

            for g in group_names:
                inst_list = record_groups[g].get("reference_instances") or []
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
    numerics cast to ``float``; ``None`` / NaN / +/-Inf serialize as JSON
    ``null``. NumPy scalar dtypes (``np.float32``, ``np.int64``, ...) are
    cast through ``float`` too — without this, ``json.dumps`` would raise
    on any non-``float64`` NumPy scalar that leaks through from a metric."""
    if v is None:
        return None
    if isinstance(v, (int, float, np.integer, np.floating)):
        f = float(v)
        if np.isnan(f) or np.isinf(f):
            return None
        return f
    return v


def _parse_jsonl_value(v) -> float | None:
    """Inverse of ``_canonical_jsonl_value``: JSON ``null`` and NaN/+/-Inf
    map to ``None`` (matching TSV semantics where empty cells round-trip
    to ``None``)."""
    if v is None:
        return None
    if isinstance(v, (int, float, np.integer, np.floating)):
        f = float(v)
        if np.isnan(f) or np.isinf(f):
            return None
        return f
    return v


def _iter_jsonl_records(path: Path):
    """Yields parsed JSON records from a JSONL file, skipping blank lines.
    NOT THREAD SAFE BY ITSELF.

    Raises ``ValueError`` (with file path and 1-based line number) on a
    malformed line, so a partial write from a crash or a hand-edit gets
    pinpointed rather than surfacing as a context-free ``JSONDecodeError``.
    """
    with open(path, "r", encoding="utf8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"{path}: malformed JSON on line {line_number}: {e.msg}"
                ) from e


_JSONL_SEPARATORS = (",", ":")


def _append_jsonl_record(path: Path, record: dict) -> None:
    """Appends a single JSON record as one line to a JSONL file. The
    record payload and trailing newline are emitted in a single
    ``write()`` so that, when combined with ``O_APPEND``, lines from
    concurrent writers can't interleave below the OS' atomic-write
    threshold. Concurrent writers above that threshold still need an
    external lock — see ``FileLock`` in the aggregator."""
    line = json.dumps(record, ensure_ascii=False, separators=_JSONL_SEPARATORS) + "\n"
    with open(path, "a", encoding="utf8") as f:
        f.write(line)


def _write_jsonl_records(path: Path, records: list[dict]) -> None:
    """Writes all records as JSONL, overwriting any existing content.
    NOT THREAD SAFE BY ITSELF."""
    with open(path, "w", encoding="utf8") as f:
        for record in records:
            f.write(
                json.dumps(record, ensure_ascii=False, separators=_JSONL_SEPARATORS)
                + "\n"
            )
