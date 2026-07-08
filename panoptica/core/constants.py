"""Backend-agnostic constants and small enums shared across all layers."""

from __future__ import annotations

from enum import Enum, auto


class CCABackend(Enum):
    """Connected-components algorithm backend for the CPU path.

    cc3d for 3D, scipy.ndimage for 1D/2D. On GPU the choice is
    made in kernels/ccl.py (cupyx / cucim), not here.
    """

    cc3d = auto()
    scipy = auto()


# Default label value meaning "background" everywhere in panoptica.
BACKGROUND = 0
