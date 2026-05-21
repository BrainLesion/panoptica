from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Literal, get_args

if TYPE_CHECKING:
    from panoptica.panoptica_result import PanopticaAUTCResult, PanopticaResult

COMPUTATION_TIME_KEY = "computation_time"

FileType = Literal["tsv", "jsonl"]
supported_file_types: tuple[FileType, ...] = get_args(FileType)


def derive_file_type(file_path: Path) -> FileType:
    """Derives the supported file type from a path's extension.

    Args:
        file_path (Path): Path whose suffix (e.g. ``.tsv``) identifies the format.

    Returns:
        FileType: The detected file type literal, one of ``supported_file_types``.

    Raises:
        ValueError: If the extension is missing or not in ``supported_file_types``.
    """
    if not file_path.suffix:
        raise ValueError(
            f"No file extension on {file_path}. Use one of: {', '.join(supported_file_types)}."
        )
    file_type = file_path.suffix.removeprefix(".")
    if file_type not in supported_file_types:
        raise ValueError(
            f"You provided the extension {file_path.suffix}, but currently only {', '.join(supported_file_types)} are supported. Either delete it or set a supported extension."
        )

    return file_type


class FileBackend(ABC):
    """Strategy for reading/writing Panoptica aggregator results to a file.

    Concrete subclasses encapsulate one on-disk format (TSV, JSONL, ...).
    Backends are not thread-safe by themselves; callers must hold the
    relevant file lock around each method call.
    """

    def __init__(self, path: Path):
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    @abstractmethod
    def prepare_for_append(
        self,
        class_group_names: list[str],
        evaluation_metrics: list[str],
        collect_existing: bool = True,
    ) -> list[str]:
        """Initialize the file for appending and return existing subject names.

        Creates the file (with any necessary header / metadata) if it does
        not exist. Validates that an existing file is compatible with the
        current evaluator configuration. Returns the list of subject names
        already present in the file so the aggregator can seed its dedup
        buffer.

        When ``collect_existing`` is False, schema validation still runs but
        the subject-name list is not materialised — useful for the
        ``continue_file=False`` aggregator path, where the list would be
        discarded anyway. Returns ``[]`` in that case.

        ``log_times=True`` adds the ``computation_time`` key to the schema; a
        file opened with one ``log_times`` setting cannot be continued with
        the other (the resulting schema mismatch raises ``ValueError``).

        Raises:
            ValueError: If existing content is incompatible with the current
                evaluator configuration (e.g. different metric set).
        """

    @abstractmethod
    def append_subject(
        self,
        subject_name: str,
        result_grouped: dict[str, PanopticaResult | PanopticaAUTCResult],
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

    @abstractmethod
    def write_full(
        self,
        subj_names: list[str],
        value_dict: dict[str, dict[str, list[float | None]]],
        class_group_names: list[str],
        evaluation_metrics: list[str],
    ) -> None:
        """Overwrite the file with a complete ``Panoptica_Statistic`` snapshot.

        Used by ``Panoptica_Statistic.to_file`` for format conversion;
        complements the incremental ``append_subject`` path used by the
        aggregator.
        """


# Concrete backend imports live at the bottom to break the import cycle
from panoptica.utils.file_backend_tsv import TSVBackend
from panoptica.utils.file_backend_jsonl import JSONLBackend

_BACKENDS: dict[FileType, type[FileBackend]] = {
    "tsv": TSVBackend,
    "jsonl": JSONLBackend,
}


def get_backend(path: Path) -> FileBackend:
    """Resolves a ``FileBackend`` for the given path by its extension."""
    return _BACKENDS[derive_file_type(path)](path)
