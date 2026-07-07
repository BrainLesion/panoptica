"""Backends: array-namespace resolution, device movement, capability probing."""

from panoptica.backends.capability import (
    GPU_UNSUPPORTED,
    has_cucim,
    has_cupy,
    warn_once,
)
from panoptica.backends.device import peak_mem, synchronize, to_device, to_host
from panoptica.backends.namespace import resolve

__all__ = [
    "resolve",
    "to_host",
    "to_device",
    "synchronize",
    "peak_mem",
    "has_cupy",
    "has_cucim",
    "warn_once",
    "GPU_UNSUPPORTED",
]
