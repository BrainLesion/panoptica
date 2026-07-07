"""Host<->device array movement, stream sync, and memory accounting.

CuPy is async: `synchronize` MUST be called before stopping a benchmark timer or
the measured GPU time is meaningless.
"""

from __future__ import annotations

import numpy as np

from panoptica.core.protocols import Array


def to_host(a: Array) -> np.ndarray:
    """Return a numpy array on the host, copying down from GPU if needed."""
    get = getattr(a, "get", None)  # cupy.ndarray.get()
    if callable(get):
        return get()  # pyrefly: ignore
    return np.asarray(a)


def to_device(a: Array, device: str) -> Array:
    """Move `a` to `device` ("cpu" -> numpy, "cuda[:i]" -> cupy)."""
    if device == "cpu":
        return to_host(a)
    import cupy

    return cupy.asarray(a)


def synchronize(device: str) -> None:
    """Block until queued work on `device` completes. No-op on CPU."""
    if device == "cpu":
        return
    import cupy

    cupy.cuda.Stream.null.synchronize()


def peak_mem(device: str) -> int:
    """Peak memory in bytes for `device` since the last reset (best effort)."""
    if device == "cpu":
        try:
            import resource

            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        except Exception:
            return 0
    import cupy

    return int(cupy.get_default_memory_pool().used_bytes())
