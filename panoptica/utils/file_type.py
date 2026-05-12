from pathlib import Path
from typing import Literal, get_args

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
