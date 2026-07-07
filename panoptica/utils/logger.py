"""Central logger for panoptica.

Library code emits progress, status and error messages through :data:`logger`
rather than ``print``, so output is structured, level-filterable and can be
silenced or redirected by downstream applications::

    import logging
    logging.getLogger("panoptica").setLevel(logging.WARNING)  # quiet
    logging.getLogger("panoptica").setLevel(logging.DEBUG)    # verbose

A :class:`rich.logging.RichHandler` is attached once so that, like the previous
``print`` calls, messages are visible out of the box; downstream code may remove
or replace it. This module imports nothing from ``panoptica`` and is therefore
free of the package's import cycles.
"""

from __future__ import annotations

import logging

from rich.logging import RichHandler

logger = logging.getLogger("panoptica")

if not logger.handlers:
    _handler = RichHandler(
        show_time=False,
        show_path=False,
        markup=False,
        rich_tracebacks=True,
    )
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def set_log_level(level: int | str) -> None:
    """Set the verbosity of panoptica's output.

    A convenience wrapper around ``logging.getLogger("panoptica").setLevel`` so callers
    do not have to import :mod:`logging` or know the logger's name.

    Args:
        level: A standard logging level, given either as an int
            (e.g. :data:`logging.WARNING`) or as a case-insensitive name
            (``"DEBUG"``, ``"INFO"``, ``"WARNING"``, ``"ERROR"``, ``"CRITICAL"``).
            ``set_log_level("WARNING")`` silences the routine progress/status messages
            while keeping warnings and errors; ``set_log_level("DEBUG")`` is the most
            verbose.
    """
    if isinstance(level, str):
        level = level.upper()
    logger.setLevel(level)
