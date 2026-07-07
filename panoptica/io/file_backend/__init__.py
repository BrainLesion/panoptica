"""File backend registry and implementations for result serialization.

Backends (TSV, JSONL) persist Serializable result objects without importing
concrete result classes, avoiding circular imports.
"""

from __future__ import annotations

from panoptica.io.file_backend.registry import (
    FileBackend,
    FileType,
    derive_file_type,
    get_backend,
)

__all__ = ["FileBackend", "FileType", "derive_file_type", "get_backend"]
