"""Input/output layer: converters, sanity checks, and file backends.

Includes:
- convert/: converters from numpy, SimpleITK, nibabel, nrrd, cupy to canonical form
- sanity.py: shape/dtype/label-value validation
- file_backend/: TSV and JSONL backends for persisting results
"""

from __future__ import annotations

from panoptica.io.convert import convert_to_numpy
from panoptica.io.file_backend import (
    FileBackend,
    FileType,
    derive_file_type,
    get_backend,
)
from panoptica.io.sanity import sanity_check

__all__ = [
    "convert_to_numpy",
    "sanity_check",
    "FileBackend",
    "FileType",
    "derive_file_type",
    "get_backend",
]
