"""Thread-parallel map for the per-instance CPU loops.

The heavy per-instance kernels (EDT, skeletonize) release the GIL, so a thread
pool over instances scales across cores. But for cheap per-instance work (2D,
small crops) the pool's dispatch/scheduling overhead costs more than it saves.

Rather than gate on a hand-tuned voxel count, the decision is **self-calibrating
and measured**: cost the first item, compare it to the pool's measured per-task
dispatch overhead, and thread only when per-item work clears that bar by a safety
margin. All quantities are measured at runtime, so it adapts to the host (fast
CPU, slow CPU, few vs many cores) with no magic constant baked in.

Above the bar, items are split into `n_jobs` contiguous chunks (one task per
worker, not per instance) over a single persistent pool. GPU work stays serial.
"""

from __future__ import annotations

import atexit
import math
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

_POOL: ThreadPoolExecutor | None = None
_POOL_WORKERS = 0
_OVERHEAD: float | None = None  # measured seconds to round-trip one task
# Thread only when a single item's work exceeds the per-task dispatch cost by
# this factor. Dimensionless safety margin (a ratio, not a hardware/data
# constant) — absorbs first-item measurement noise + GIL contention the empty
# probe doesn't see.
_MARGIN = 4.0


def _get_pool(workers: int) -> ThreadPoolExecutor:
    global _POOL, _POOL_WORKERS
    if _POOL is None or _POOL_WORKERS < workers:
        if _POOL is not None:
            _POOL.shutdown(wait=False)
        _POOL = ThreadPoolExecutor(max_workers=workers)
        _POOL_WORKERS = workers
    return _POOL


def _dispatch_overhead(pool: ThreadPoolExecutor, workers: int) -> float:
    """Measured seconds to round-trip one trivial task through the pool."""
    global _OVERHEAD
    if _OVERHEAD is None:
        best = float("inf")
        for _ in range(3):
            t0 = time.perf_counter()
            list(pool.map(lambda _: None, range(workers)))
            best = min(best, (time.perf_counter() - t0) / workers)
        _OVERHEAD = best
    return _OVERHEAD


def parallel_list(fn: Callable[[Any], Any], items: Sequence[Any], n_jobs: int) -> list:
    items = list(items)
    n = len(items)
    if n_jobs is None or n_jobs <= 1 or n <= 1:
        return [fn(x) for x in items]
    workers = min(n_jobs, n)
    pool = _get_pool(workers)

    # Cost the first item (its result is kept, not wasted), then decide.
    t0 = time.perf_counter()
    first = fn(items[0])
    per_item = time.perf_counter() - t0
    rest = items[1:]
    if not rest or per_item < _MARGIN * _dispatch_overhead(pool, workers):
        return [first] + [fn(x) for x in rest]

    chunk = math.ceil(len(rest) / workers)
    batches = [rest[i : i + chunk] for i in range(0, len(rest), chunk)]
    out: list = [first]
    for r in pool.map(lambda b: [fn(x) for x in b], batches):
        out.extend(r)
    return out


@atexit.register
def _shutdown() -> None:
    if _POOL is not None:
        _POOL.shutdown(wait=False)
