"""File backend registry — maps extensions to backends."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from panoptica.core.errors import InputValidationError

FileType = Literal["tsv", "jsonl"]
SUPPORTED_FILE_TYPES: tuple[FileType, ...] = ("tsv", "jsonl")


def derive_file_type(file_path: Path) -> FileType:
    """Derive the supported file type from a path's extension.

    Args:
        file_path: Path whose suffix (e.g. ``.tsv``) identifies the format.

    Returns:
        FileType: The detected file type literal, one of ``SUPPORTED_FILE_TYPES``.

    Raises:
        InputValidationError: If the extension is missing or not in ``SUPPORTED_FILE_TYPES``.
    """
    if not file_path.suffix:
        raise InputValidationError(
            f"No file extension on {file_path}. Use one of: {', '.join(SUPPORTED_FILE_TYPES)}."
        )
    file_type = file_path.suffix.removeprefix(".").lower()
    if file_type not in SUPPORTED_FILE_TYPES:
        raise InputValidationError(
            f"Unsupported file extension {file_path.suffix}. "
            f"Currently supported: {', '.join(SUPPORTED_FILE_TYPES)}."
        )

    return file_type  # type: ignore[return-value]


def get_backend(file_path: Path) -> FileBackend:
    """Resolve a FileBackend for the given path by its extension.

    Args:
        file_path: Path to the output file (determines format).

    Returns:
        A FileBackend instance for the given path.

    Raises:
        InputValidationError: If the file extension is not supported.
    """
    from panoptica.io.file_backend.jsonl import JSONLBackend
    from panoptica.io.file_backend.tsv import TSVBackend

    backends: dict[FileType, type[FileBackend]] = {
        "tsv": TSVBackend,
        "jsonl": JSONLBackend,
    }

    file_type = derive_file_type(file_path)
    return backends[file_type](file_path)


class FileBackend:
    """Base class for file-format backends (TSV, JSONL, etc.).

    Backends implement reading/writing of Serializable result objects to disk.
    Each backend depends only on the Serializable protocol (panoptica.core.protocols),
    never on a concrete result class.

    Backends are not thread-safe by themselves; callers must hold the relevant
    file lock around each method call.
    """

    def __init__(self, path: Path) -> None:
        """Initialize backend for the given path.

        Args:
            path: File path where results will be stored.
        """
        self._path = path

    @property
    def path(self) -> Path:
        """Get the file path."""
        return self._path

    def prepare_for_append(
        self,
        group_names: list[str],
        metric_names: list[str],
        collect_existing: bool = True,
    ) -> list[str]:
        """Initialize the file for appending and return existing subject names.

        Creates the file (with any necessary header/metadata) if it does not exist.
        Validates that an existing file is compatible with the current configuration.
        Returns the list of subject names already present in the file so the
        aggregator can seed its dedup buffer.

        When ``collect_existing`` is False, schema validation still runs but
        the subject-name list is not materialized — useful when the result will
        be discarded anyway. Returns ``[]`` in that case.

        Args:
            group_names: List of label group names (e.g. ["liver", "spleen"]).
            metric_names: List of metric names being evaluated.
            collect_existing: If True, return existing subject names; if False, return [].

        Returns:
            List of subject names already present in the file, or [] if collect_existing=False.

        Raises:
            InputValidationError: If an existing file has incompatible schema.
        """
        raise NotImplementedError

    def append_subject(
        self,
        subject_name: str,
        result_grouped: dict[str, object],
        group_names: list[str],
        metric_names: list[str],
        output_individual_instance_metrics: bool,
    ) -> None:
        """Append one subject's per-group results to the file.

        Args:
            subject_name: Unique name for this subject.
            result_grouped: Dict mapping group name -> Serializable result object (with to_dict() method).
            group_names: List of label group names.
            metric_names: List of metric names.
            output_individual_instance_metrics: If True, include per-instance metrics.

        Raises:
            InputValidationError: If result schema does not match file schema.
        """
        raise NotImplementedError

    def load_raw(
        self, verbose: bool = True
    ) -> tuple[list[str], dict[str, dict[str, list[float | None]]]]:
        """Read the file and return raw data.

        Returns the (subject_names, value_dict) pair that aggregation/statistics
        code consumes. Returning raw data (rather than constructed result objects)
        avoids circular imports.

        Args:
            verbose: If True, log what was found.

        Returns:
            (subj_names, value_dict) where:
            - subj_names: List of subject names in file order.
            - value_dict: Dict[group_name][metric_name] = list of per-subject values (or None for missing).

        Raises:
            InputValidationError: If file is malformed.
        """
        raise NotImplementedError

    def write_full(
        self,
        subject_names: list[str],
        value_dict: dict[str, dict[str, list[float | None]]],
        group_names: list[str],
        metric_names: list[str],
    ) -> None:
        """Overwrite the file with a complete snapshot.

        Used by statistics export for format conversion; complements the
        incremental append_subject path.

        Args:
            subject_names: List of subject names in order.
            value_dict: Dict[group_name][metric_name] = list of per-subject values.
            group_names: List of label group names.
            metric_names: List of metric names.

        Raises:
            InputValidationError: If data is malformed.
        """
        raise NotImplementedError
