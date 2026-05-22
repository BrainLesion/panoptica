from __future__ import annotations

from pathlib import Path

from panoptica.utils.file_backend import FileBackend, FileType, derive_file_type
from panoptica.utils.file_backend_tsv import TSVBackend
from panoptica.utils.file_backend_jsonl import JSONLBackend

_BACKENDS: dict[FileType, type[FileBackend]] = {
    "tsv": TSVBackend,
    "jsonl": JSONLBackend,
}


def get_backend(path: Path) -> FileBackend:
    """Resolves a ``FileBackend`` for the given path by its extension."""
    return _BACKENDS[derive_file_type(path)](path)
