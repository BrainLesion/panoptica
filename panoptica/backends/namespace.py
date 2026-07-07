"""Resolve the active array namespace (numpy or cupy) from a device string.

This is the ONE place (besides device.py and CPU-pinned kernels) allowed to import
numpy/cupy directly. Everyone else receives the returned `xp` and uses it.
"""

from __future__ import annotations

import numpy as np

from panoptica.core.errors import BackendUnavailable
from panoptica.core.protocols import Xp


def _cupy_available() -> bool:
    try:
        import cupy  # noqa: F401

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def resolve(device: str = "auto") -> tuple[Xp, str]:
    """Return (xp, resolved_device) for a device string.

    device: "auto" | "cpu" | "cuda" | "cuda:<idx>".
      - "auto": cupy if a CUDA device is present, else numpy.
      - "cpu": numpy.
      - "cuda[:i]": cupy on device i (raises BackendUnavailable if no GPU/CuPy).
    """
    dev = device.lower()
    if dev == "cpu":
        return np, "cpu"

    if dev == "auto":
        return _resolve_cuda("cuda") if _cupy_available() else (np, "cpu")

    if dev == "cuda" or dev.startswith("cuda:"):
        return _resolve_cuda(dev)

    raise BackendUnavailable(f"Unknown device string: {device!r}")


def _resolve_cuda(dev: str) -> tuple[Xp, str]:
    try:
        import cupy
    except Exception as e:  # pragma: no cover - depends on install
        raise BackendUnavailable(
            "cuda requested but CuPy is not installed (pip install 'panoptica[cuda]')"
        ) from e
    if cupy.cuda.runtime.getDeviceCount() == 0:
        raise BackendUnavailable("cuda requested but no CUDA device is visible")
    idx = int(dev.split(":", 1)[1]) if ":" in dev else 0
    cupy.cuda.Device(idx).use()
    return cupy, f"cuda:{idx}"
