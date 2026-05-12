from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from panoptica.utils.file_type import FileType, derive_file_type
from panoptica.utils.serialization import format_instance_subject_name

if TYPE_CHECKING:
    from panoptica.panoptica_result import PanopticaAUTCResult, PanopticaResult


COMPUTATION_TIME_KEY = "computation_time"


class FileBackend(ABC):
    """Strategy for reading/writing Panoptica aggregator results to a file.

    Concrete subclasses encapsulate one on-disk format (TSV, JSONL, ...).
    Backends are not thread-safe by themselves; callers must hold the
    relevant file lock around each method call.
    """

    def __init__(self, path: Path):
        self.path = path

    @abstractmethod
    def prepare_for_append(
        self,
        class_group_names: list[str],
        evaluation_metrics: list[str],
    ) -> list[str]:
        """Initialize the file for appending and return existing subject names.

        Creates the file (with any necessary header / metadata) if it does
        not exist. Validates that an existing file is compatible with the
        current evaluator configuration. Returns the list of subject names
        already present in the file so the aggregator can seed its dedup
        buffer.

        Raises:
            ValueError: If existing content is incompatible with the current
                evaluator configuration (e.g. different metric set).
        """

    @abstractmethod
    def append_subject(
        self,
        subject_name: str,
        result_grouped: dict[str, "PanopticaResult | PanopticaAUTCResult"],
        class_group_names: list[str],
        evaluation_metrics: list[str],
        output_individual_instance_metrics: bool,
    ) -> None:
        """Append one subject's per-group results to the file."""

    @abstractmethod
    def load_raw(
        self, verbose: bool = True
    ) -> tuple[list[str], dict[str, dict[str, list[float | None]]]]:
        """Read the file and return the raw data for ``Panoptica_Statistic``.

        Returns the ``(subj_names, value_dict)`` pair that
        ``Panoptica_Statistic.__init__`` consumes. Returning raw data
        (rather than a Statistic instance) avoids a circular import with
        ``panoptica.panoptica_statistics``.
        """


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
    ) -> list[str]:
        header = ["subject_name"] + [
            f"{g}-{m}" for g in class_group_names for m in evaluation_metrics
        ]
        header_hash = hash("+".join(header))

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
                if header_hash != hash("+".join(existing_header)):
                    raise ValueError(
                        f"{self.path}: Hash of header not the same! You are using a different setup!"
                    )

        # Return existing subject names from the file (excluding the header
        # row). On a fresh file this list is empty.
        all_first_column = _load_first_tsv_column(self.path)
        return [s for s in all_first_column if s != "subject_name"]

    def append_subject(
        self,
        subject_name: str,
        result_grouped: dict[str, "PanopticaResult | PanopticaAUTCResult"],
        class_group_names: list[str],
        evaluation_metrics: list[str],
        output_individual_instance_metrics: bool,
    ) -> None:
        if output_individual_instance_metrics:
            all_rows: list[list] = []
            summary_row: list = [subject_name]
            group_rows_as_dicts: dict[str, list[dict]] = {}
            for groupname in class_group_names:
                result = result_grouped[groupname]
                rows_as_dicts = result.to_dict(True)
                group_rows_as_dicts[groupname] = rows_as_dicts
                summary_dict = rows_as_dicts[0] if len(rows_as_dicts) > 0 else {}
                if result.computation_time is not None:
                    summary_dict = dict(summary_dict)
                    summary_dict[COMPUTATION_TIME_KEY] = result.computation_time
                for e in evaluation_metrics:
                    summary_row.append(summary_dict.get(e, ""))
            all_rows.append(summary_row)
            for groupname in class_group_names:
                rows_as_dicts = group_rows_as_dicts[groupname]
                for inst_idx, r_dict in enumerate(rows_as_dicts[1:]):
                    row: list = [
                        format_instance_subject_name(
                            subject_name, groupname, inst_idx
                        )
                    ]
                    for current_groupname in class_group_names:
                        if current_groupname == groupname:
                            for e in evaluation_metrics:
                                row.append(r_dict.get(e, ""))
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
                    content.append(result_dict.get(e, ""))
            _append_tsv_rows(self.path, [content])

        print(f"Saved entry {subject_name} into {self.path}")

    def load_raw(
        self, verbose: bool = True
    ) -> tuple[list[str], dict[str, dict[str, list[float | None]]]]:
        with open(self.path, "r", encoding="utf8", newline="") as tsvfile:
            rd = csv.reader(tsvfile, delimiter="\t", lineterminator="\n")
            rows = [row for row in rd]

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
        group_names = list({k[0] for k in keys_in_order})

        if verbose:
            print(f"Found {len(rows) - 1} entries")
            print(f"Found metrics: {metric_names}")
            print(f"Found groups: {group_names}")

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
                    if not np.isnan(parsed) and parsed != np.inf:
                        value_dict[group_name][metric_name].append(parsed)
                    else:
                        value_dict[group_name][metric_name].append(None)
                else:
                    value_dict[group_name][metric_name].append(None)

        return subj_names, value_dict


class JSONLBackend(FileBackend):
    """JSON-lines format. One nested JSON object per subject, with nested
    per-group summary and (optional) matched-instance metrics.

    Scaffolding only — implementation deferred.
    """

    def prepare_for_append(
        self,
        class_group_names: list[str],
        evaluation_metrics: list[str],
    ) -> list[str]:
        raise NotImplementedError("JSONL backend is not yet implemented.")

    def append_subject(
        self,
        subject_name: str,
        result_grouped: dict[str, "PanopticaResult | PanopticaAUTCResult"],
        class_group_names: list[str],
        evaluation_metrics: list[str],
        output_individual_instance_metrics: bool,
    ) -> None:
        raise NotImplementedError("JSONL backend is not yet implemented.")

    def load_raw(
        self, verbose: bool = True
    ) -> tuple[list[str], dict[str, dict[str, list[float | None]]]]:
        raise NotImplementedError("JSONL backend is not yet implemented.")


_BACKENDS: dict[FileType, type[FileBackend]] = {
    "tsv": TSVBackend,
    "jsonl": JSONLBackend,
}


def get_backend(path: Path) -> FileBackend:
    """Resolves a ``FileBackend`` for the given path by its extension."""
    return _BACKENDS[derive_file_type(path)](path)


# --- TSV helpers (module-private; the JSONL counterparts live alongside
#     JSONLBackend once implemented) -----------------------------------------


def _read_first_tsv_row(path: Path) -> list[str]:
    """Reads the first row of a TSV file. NOT THREAD SAFE BY ITSELF."""
    with open(path, "r", encoding="utf8", newline="") as tsvfile:
        rd = csv.reader(tsvfile, delimiter="\t", lineterminator="\n")
        rows = list(rd)
        return rows[0] if rows else []


def _load_first_tsv_column(path: Path) -> list[str]:
    """Loads the entries from the first column of a TSV file.

    NOT THREAD SAFE BY ITSELF.

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
