from __future__ import annotations

from pathlib import Path

from panoptica.panoptica_result import PanopticaAUTCResult, PanopticaResult
from panoptica.utils.file_backend import FileBackend
from panoptica.utils.serialization import (
    format_instance_subject_name,
    is_instance_row,
    parse_instance_subject_name,
)
from panoptica.utils.file_backend import COMPUTATION_TIME_KEY
import numpy as np
import csv


class TSVBackend(FileBackend):
    """Tab-separated values format. One row per subject (plus optional
    per-instance rows when ``output_individual_instance_metrics=True``).

    Header is ``["subject_name", "{group}-{metric}", ...]``. Empty cells
    represent missing values and round-trip to ``None`` on read.
    """

    def prepare_for_append(
        self,
        class_group_names: list[str],
        evaluation_metrics: list[str],
        collect_existing: bool = True,
    ) -> list[str]:
        header = ["subject_name"] + [
            f"{g}-{m}" for g in class_group_names for m in evaluation_metrics
        ]

        if not self.path.exists():
            _append_tsv_rows(self.path, [header])
        else:
            existing_header = _read_first_tsv_row(self.path)
            if len(existing_header) == 0:
                print(
                    f"{self.path}: Output file given is empty, will start with header"
                )
                _append_tsv_rows(self.path, [header])
            else:
                # TODO should also hash panoptica_evaluator to be safe, and save into header of file
                if header != existing_header:
                    raise ValueError(
                        f"{self.path}: Header does not match! You are using a different setup!"
                    )

        # Skip the full subject-column scan when the caller has signalled
        # that the result will be discarded (continue_file=False).
        if not collect_existing:
            return []
        all_first_column = _load_first_tsv_column(self.path)
        return [s for s in all_first_column if s != "subject_name"]

    def append_subject(
        self,
        subject_name: str,
        result_grouped: dict[str, PanopticaResult | PanopticaAUTCResult],
        class_group_names: list[str],
        evaluation_metrics: list[str],
        output_individual_instance_metrics: bool,
    ) -> None:
        if output_individual_instance_metrics:
            all_rows: list[list] = []
            summary_row: list = [subject_name]
            group_instance_rows: dict[str, list[dict]] = {}
            for groupname in class_group_names:
                result = result_grouped[groupname]
                summary_dict = result.to_dict(True)
                # Row keys live under "reference_instances"; pop so they don't
                # bleed into the summary loop below.
                group_instance_rows[groupname] = summary_dict.pop(
                    "reference_instances", []
                )
                if result.computation_time is not None:
                    summary_dict = dict(summary_dict)
                    summary_dict[COMPUTATION_TIME_KEY] = result.computation_time
                for e in evaluation_metrics:
                    summary_row.append(_canonical_tsv_value(summary_dict.get(e)))
            all_rows.append(summary_row)
            for groupname in class_group_names:
                instance_rows = group_instance_rows[groupname]
                for inst_idx, r_dict in enumerate(instance_rows):
                    # Normalize row keys to master keys so they line up with the
                    # TSV column schema (which is master-keyed).
                    r_dict = PanopticaResult.normalize_row_to_master_schema(r_dict)
                    row: list = [
                        format_instance_subject_name(subject_name, groupname, inst_idx)
                    ]
                    for current_groupname in class_group_names:
                        if current_groupname == groupname:
                            for e in evaluation_metrics:
                                row.append(_canonical_tsv_value(r_dict.get(e)))
                        else:
                            for _ in evaluation_metrics:
                                row.append("")
                    all_rows.append(row)

            _append_tsv_rows(self.path, all_rows)
        else:
            content: list = [subject_name]
            for groupname in class_group_names:
                result = result_grouped[groupname]
                result_dict = result.to_dict(False)
                if result.computation_time is not None:
                    result_dict[COMPUTATION_TIME_KEY] = result.computation_time
                for e in evaluation_metrics:
                    content.append(_canonical_tsv_value(result_dict.get(e)))
            _append_tsv_rows(self.path, [content])

        print(f"Saved entry {subject_name} into {self.path}")

    def write_full(
        self,
        subj_names: list[str],
        value_dict: dict[str, dict[str, list[float | None]]],
        class_group_names: list[str],
        evaluation_metrics: list[str],
    ) -> None:
        header = ["subject_name"] + [
            f"{g}-{m}" for g in class_group_names for m in evaluation_metrics
        ]
        rows: list[list] = [header]
        for i, sn in enumerate(subj_names):
            row: list = [sn]
            inst_info = parse_instance_subject_name(sn) if is_instance_row(sn) else None
            inst_group = inst_info[1] if inst_info is not None else None
            for g in class_group_names:
                if inst_group is not None and g != inst_group:
                    for _ in evaluation_metrics:
                        row.append("")
                else:
                    for m in evaluation_metrics:
                        row.append(_canonical_tsv_value(value_dict[g][m][i]))
            rows.append(row)
        _write_tsv_rows(self.path, rows)

    def load_raw(
        self, verbose: bool = True
    ) -> tuple[list[str], dict[str, dict[str, list[float | None]]]]:
        with open(self.path, "r", encoding="utf8", newline="") as tsvfile:
            rd = csv.reader(tsvfile, delimiter="\t", lineterminator="\n")
            rows = list(rd)

        header = rows[0]
        if header[0] != "subject_name":
            raise ValueError(
                "First column is not subject_names, something wrong with the file?"
            )

        keys_in_order: list[tuple[str, str]] = [
            tuple(c.split("-", maxsplit=1)) for c in header[1:]  # type: ignore[misc]
        ]
        keys_in_order = [
            k if len(k) == 2 else ("ungrouped", k[0]) for k in keys_in_order
        ]
        metric_names: list[str] = []
        for k in keys_in_order:
            if k[1] not in metric_names:
                metric_names.append(k[1])
        # Preserve column ordering — a set comprehension drops it.
        group_names = list(dict.fromkeys(k[0] for k in keys_in_order))

        if verbose:
            print(f"Found {len(rows) - 1} entries")
            print(f"Found metrics: {metric_names}")
            print(f"Found groups: {group_names}")

        # Header-only file
        if len(rows) == 1:
            return [], {}

        subj_names: list[str] = []
        value_dict: dict[str, dict[str, list[float | None]]] = {}

        for r in rows[1:]:
            sn = r[0]
            subj_names.append(sn)
            for idx, value in enumerate(r[1:]):
                group_name, metric_name = keys_in_order[idx]
                if group_name not in value_dict:
                    value_dict[group_name] = {m: [] for m in metric_names}
                if len(value) > 0:
                    parsed = float(value)
                    if not np.isnan(parsed) and not np.isinf(parsed):
                        value_dict[group_name][metric_name].append(parsed)
                    else:
                        value_dict[group_name][metric_name].append(None)
                else:
                    value_dict[group_name][metric_name].append(None)

        return subj_names, value_dict


def _canonical_tsv_value(v):
    """Canonicalize a value for TSV output.

    Casts ``int`` / ``float`` through ``float`` so that ``5`` and ``5.0``
    produce the same on-disk byte representation — required for the TSV
    <-> JSONL byte-identical roundtrip after values have been re-typed
    by ``load_raw``. ``None``, ``NaN`` and +/-``Inf`` all map to ``""``
    (matching the read path which drops NaN/Inf to ``None``). NumPy
    scalar dtypes (``np.float32``, ``np.int64``, ...) are normalised
    through the same path so they don't escape as the literal strings
    ``"nan"`` / ``"inf"`` via ``csv.writer``'s default ``str(...)``,
    which would break the byte-identical roundtrip.
    """
    if v is None:
        return ""
    if isinstance(v, (int, float, np.integer, np.floating)):
        f = float(v)
        if np.isnan(f) or np.isinf(f):
            return ""
        return f
    return v


def _read_first_tsv_row(path: Path) -> list[str]:
    """Reads the first row of a TSV file. NOT THREAD SAFE BY ITSELF."""
    with open(path, "r", encoding="utf8", newline="") as tsvfile:
        rd = csv.reader(tsvfile, delimiter="\t", lineterminator="\n")
        rows = list(rd)
        return rows[0] if rows else []


def _load_first_tsv_column(path: Path) -> list[str]:
    """Loads the entries from the first column of a TSV file.

    NOT THREAD SAFE BY ITSELF.

    Cost is O(N) in rows on every aggregator construction (full file scan plus
    a set for duplicate detection). Fine for typical study sizes; if a TSV ever
    grows to hundreds of thousands of instance-level rows, prefer streaming.

    Raises:
        ValueError: If the file contains duplicate entries.
    """
    with open(path, "r", encoding="utf8", newline="") as tsvfile:
        rd = csv.reader(tsvfile, delimiter="\t", lineterminator="\n")
        rows = list(rd)
    id_list = [row[0] for row in rows]
    if len(id_list) != len(set(id_list)):
        raise ValueError(f"{path}: file has duplicate entries!")
    return id_list


def _append_tsv_rows(path: Path, content: list[list]) -> None:
    """Appends rows to a TSV file. NOT THREAD SAFE BY ITSELF."""
    with open(path, "a", encoding="utf8", newline="") as tsvfile:
        writer = csv.writer(tsvfile, delimiter="\t", lineterminator="\n")
        for c in content:
            writer.writerow(["" if v is None else v for v in c])


def _write_tsv_rows(path: Path, content: list[list]) -> None:
    """Writes rows to a TSV file, overwriting any existing content.
    NOT THREAD SAFE BY ITSELF."""
    with open(path, "w", encoding="utf8", newline="") as tsvfile:
        writer = csv.writer(tsvfile, delimiter="\t", lineterminator="\n")
        for c in content:
            writer.writerow(["" if v is None else v for v in c])
