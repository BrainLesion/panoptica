"""Which kernels have a GPU form, plus a deduplicated fallback warning.

CuPy has gaps (cc3d, skeletonize, Hungarian have no GPU form). Kernels consult
this module to decide whether to run on GPU or transfer to CPU, and emit
exactly ONE warning per kernel per session when they fall back.
"""

from __future__ import annotations

import warnings

_warned: set[str] = set()


def has_cupy() -> bool:
    try:
        import cupy  # noqa: F401

        return True
    except Exception:
        return False


def has_cucim() -> bool:
    try:
        import cucim  # noqa: F401

        return True
    except Exception:
        return False


# Kernels with no GPU implementation; connected components is excluded since
# it is GPU-capable via cucim.
GPU_UNSUPPORTED = frozenset({"skeletonize", "hungarian"})


def warn_once(kernel: str, reason: str) -> None:
    """Warn a single time per kernel that it fell back to CPU."""
    if kernel in _warned:
        return
    _warned.add(kernel)
    warnings.warn(f"[panoptica] '{kernel}' has no GPU path ({reason}); running on CPU.")
