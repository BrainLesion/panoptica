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


_MATCHED_KEY = "references_matched"
_UNMATCHED_KEY = "references_unmatched"
_BUCKET_KEYS = (_MATCHED_KEY, _UNMATCHED_KEY)
_IS_MATCHED = "is_matched"


class JSONLBackend(FileBackend):
    """JSON-lines format. One nested JSON object per subject, with nested
    per-group summary and (optional) split matched / unmatched reference
    instance metrics.

    Each line looks like::

        {"subject_name": "subj_a",
         "groups": {"liver": {"dice": 0.9, "tp": 5.0,
                              "references_matched":   [{"sq_dice": 0.95, ...}, ...],
                              "references_unmatched": [{"instance_volume_ref": 50.0}, ...]}}}

    Missing values serialize as JSON ``null`` and round-trip to Python
    ``None``. The ``references_matched`` / ``references_unmatched`` keys
    are each only present when the aggregator is run with
    ``output_individual_instance_metrics=True`` and the result yielded at
    least one row of the respective kind. The ``is_matched`` flag is not
    serialised — it is implied by the bucket key and synthesised on read
    so the value_dict shape matches what ``TSVBackend.load_raw`` produces
    (preserving byte-identical TSV ↔ JSONL roundtrip).
    """

    def prepare_for_append(
        self,
        class_group_names: list[str],
        evaluation_metrics: list[str],
    ) -> list[str]:
        # is_matched is carried as a TSV column but is implicit in the JSONL
        # bucket key; exclude it when comparing against on-disk metric set.
        expected_metrics = set(evaluation_metrics) - {_IS_MATCHED}
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
                k
                for k in first_record["groups"][g].keys()
                if k not in _BUCKET_KEYS
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
                n_inst = sum(len(g_data.get(k) or []) for k in _BUCKET_KEYS)
                for inst_idx in range(n_inst):
                    existing.append(format_instance_subject_name(sn, g, inst_idx))
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
                if e == _IS_MATCHED:
                    continue
                group_obj[e] = _canonical_jsonl_value(summary_dict.get(e))

            matched = [d for d in instance_dicts if d.get(_IS_MATCHED) == 1]
            unmatched = [d for d in instance_dicts if d.get(_IS_MATCHED) == 0]

            # Restrict to evaluation_metrics so the JSONL inst-dict set
            # matches what TSV columns can hold — required for symmetric
            # TSV<->JSONL roundtrip byte-identity. Drop None entries so
            # the inst-dict shape matches what write_full produces from a
            # roundtripped Panoptica_Statistic. Skip is_matched: the bucket
            # key already encodes it.
            def _project(rows):
                return [
                    {
                        e: _canonical_jsonl_value(r.get(e))
                        for e in evaluation_metrics
                        if e != _IS_MATCHED and r.get(e) is not None
                    }
                    for r in rows
                ]

            if matched:
                group_obj[_MATCHED_KEY] = _project(matched)
            if unmatched:
                group_obj[_UNMATCHED_KEY] = _project(unmatched)
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
        # group -> bucket_key -> list[inst_dict], keyed by master subject index.
        instances_by_master: dict[int, dict[str, dict[str, list[dict]]]] = {}
        for i, sn in enumerate(subj_names):
            if is_instance_row(sn):
                parsed = parse_instance_subject_name(sn)
                if parsed is None:
                    continue
                orig_subj, orig_group, _ = parsed
                try:
                    master_i = subj_names.index(orig_subj)
                except ValueError:
                    continue
                is_matched_col = value_dict[orig_group].get(_IS_MATCHED)
                is_matched_val = is_matched_col[i] if is_matched_col else None
                bucket_key = (
                    _MATCHED_KEY if is_matched_val == 1 else _UNMATCHED_KEY
                )
                inst_dict = {
                    m: value_dict[orig_group][m][i]
                    for m in evaluation_metrics
                    if m != _IS_MATCHED and value_dict[orig_group][m][i] is not None
                }
                buckets = instances_by_master.setdefault(master_i, {}).setdefault(
                    orig_group, {_MATCHED_KEY: [], _UNMATCHED_KEY: []}
                )
                buckets[bucket_key].append(inst_dict)
            else:
                master_indices.append(i)

        records: list[dict] = []
        for i in master_indices:
            record: dict = {"subject_name": subj_names[i], "groups": {}}
            for g in class_group_names:
                group_obj: dict = {}
                for m in evaluation_metrics:
                    if m == _IS_MATCHED:
                        continue
                    group_obj[m] = _canonical_jsonl_value(value_dict[g][m][i])
                buckets = instances_by_master.get(i, {}).get(g)
                if buckets:
                    if buckets[_MATCHED_KEY]:
                        group_obj[_MATCHED_KEY] = buckets[_MATCHED_KEY]
                    if buckets[_UNMATCHED_KEY]:
                        group_obj[_UNMATCHED_KEY] = buckets[_UNMATCHED_KEY]
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
                if k in _BUCKET_KEYS:
                    continue
                if k not in metric_names:
                    metric_names.append(k)
        # is_matched is not serialised in JSONL; synthesise it as a column so
        # the returned value_dict has the same shape as TSVBackend.load_raw.
        if _IS_MATCHED not in metric_names:
            metric_names.append(_IS_MATCHED)

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
                    if m == _IS_MATCHED:
                        # Master row: is_matched is None (matches TSV's empty cell).
                        value_dict[g][m].append(None)
                    else:
                        value_dict[g][m].append(_parse_jsonl_value(g_data.get(m)))

            for g in group_names:
                g_data = record["groups"][g]
                matched_list = g_data.get(_MATCHED_KEY) or []
                unmatched_list = g_data.get(_UNMATCHED_KEY) or []
                # Sequential inst_idx across matched-then-unmatched mirrors the
                # ordering TSV produces from result.to_dict(True)[1:].
                inst_idx = 0
                for bucket, flag in (
                    (matched_list, 1),
                    (unmatched_list, 0),
                ):
                    for inst_dict in bucket:
                        subj_names.append(
                            format_instance_subject_name(sn, g, inst_idx)
                        )
                        inst_idx += 1
                        for inner_g in group_names:
                            for m in metric_names:
                                if inner_g != g:
                                    value_dict[inner_g][m].append(None)
                                elif m == _IS_MATCHED:
                                    value_dict[inner_g][m].append(flag)
                                else:
                                    value_dict[inner_g][m].append(
                                        _parse_jsonl_value(inst_dict.get(m))
                                    )

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
