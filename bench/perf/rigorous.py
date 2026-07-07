"""Rigorous perf benchmark: cpu vs gpu (transfer-included and compute-only).

1 warmup discarded, median-of-N timing, GPU stream-sync before stopping the
timer, transfer-included vs compute-only split, peak VRAM, and a device-parity
guard (cpu tp must equal cuda tp) so we never time output that disagrees
across backends. Per-cell median, min, and IQR are all persisted.

Run:  python bench/perf/rigorous.py [--sizes small,medium] [--n 7]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from datasets.generate import (  # noqa: E402
    generate_matched_instance_pair,
    generate_semantic_pair,
    generate_unmatched_instance_pair,
    get_spacing,
)

from panoptica import Evaluator, InputType  # noqa: E402

ALL = ["DSC", "IOU", "ASSD", "RVD", "RVAE", "CEDI", "HD", "HD95", "NSD", "clDSC"]
RESULTS = Path(__file__).resolve().parents[1] / "results" / "perf_rigorous.jsonl"


def _pair(size, it):
    if it == "MATCHED_INSTANCE":
        ref, pred, _ = generate_matched_instance_pair(size=size)
        return ref, pred
    if it == "UNMATCHED_INSTANCE":
        return generate_unmatched_instance_pair(size=size)
    return generate_semantic_pair(size=size)


def _eval(ref, pred, it, device, spacing=None):
    ev = Evaluator(getattr(InputType, it), instance_metrics=ALL, device=device)
    return lambda: ev.evaluate(pred, ref, spacing=spacing)


def _time(fn, n, sync=None, warmup=2):
    """Median-of-n seconds after `warmup` untimed calls; returns (median, min, iqr).

    Two warmups (not one) so GPU memory-pool growth and per-kernel autotuning are
    absorbed before timing — the first call alone under-warms a 10-metric bundle.
    """
    for _ in range(warmup):
        fn()
    if sync:
        sync()
    ts = []
    for _ in range(n):
        s = time.perf_counter()
        fn()
        if sync:
            sync()
        ts.append(time.perf_counter() - s)
    ts.sort()
    med = statistics.median(ts)
    iqr = ts[int(0.75 * (len(ts) - 1))] - ts[int(0.25 * (len(ts) - 1))]
    return med, ts[0], iqr


def _gpu_sync():
    import cupy

    cupy.cuda.Stream.null.synchronize()


def _gpu_peak_mb():
    import cupy

    return cupy.get_default_memory_pool().total_bytes() / 1e6


def run(sizes, n):
    RESULTS.parent.mkdir(exist_ok=True)
    rows = []
    print(f"\n{'size':7} {'input':18} {'cpu':>9} {'iqr':>6} "
          f"{'gpu_xfer':>11} {'gpu_comp':>11} {'gpu/cpu':>8} {'vram':>7}")
    print("-" * 86)
    for size in sizes:
        for it in ("MATCHED_INSTANCE", "UNMATCHED_INSTANCE", "SEMANTIC"):
            ref, pred = _pair(size, it)
            # non-uniform spacing is exercised on anisotropic-spacing keys;
            # uniform keys pass None so their cells are unchanged.
            sp = get_spacing(size)
            spacing = sp if any(x != 1.0 for x in sp) else None

            cpu_tp = _eval(ref, pred, it, "cpu", spacing)().to_dict().get("tp")
            cpu_med, cpu_min, cpu_iqr = _time(_eval(ref, pred, it, "cpu", spacing), n)

            # GPU: transfer-included (numpy inputs) and compute-only (pre-moved)
            import cupy

            cupy.get_default_memory_pool().free_all_blocks()
            # device-parity guard: cuda tp must match cpu tp before we trust timing
            gpu_tp = _eval(ref, pred, it, "cuda", spacing)().to_dict().get("tp")
            parity_ok = gpu_tp == cpu_tp

            gpu_xfer_med, gpu_xfer_min, gpu_xfer_iqr = _time(
                _eval(ref, pred, it, "cuda", spacing), n, sync=_gpu_sync
            )
            ref_d, pred_d = cupy.asarray(ref), cupy.asarray(pred)
            gpu_comp_med, gpu_comp_min, gpu_comp_iqr = _time(
                _eval(ref_d, pred_d, it, "cuda", spacing), n, sync=_gpu_sync
            )
            vram = _gpu_peak_mb()

            row = dict(
                size=size, input_type=it, parity_ok=parity_ok,
                cpu_ms=cpu_med * 1e3, cpu_min_ms=cpu_min * 1e3,
                cpu_iqr_ms=cpu_iqr * 1e3,
                gpu_xfer_ms=gpu_xfer_med * 1e3, gpu_xfer_min_ms=gpu_xfer_min * 1e3,
                gpu_xfer_iqr_ms=gpu_xfer_iqr * 1e3,
                gpu_comp_ms=gpu_comp_med * 1e3, gpu_comp_min_ms=gpu_comp_min * 1e3,
                gpu_comp_iqr_ms=gpu_comp_iqr * 1e3,
                speedup_gpu_comp=cpu_med / gpu_comp_med if gpu_comp_med else None,
                vram_mb=vram, n=n,
            )
            rows.append(row)
            flag = "" if parity_ok else " !PARITY"

            print(f"{size:7} {it:18} {row['cpu_ms']:9.1f} {row['cpu_iqr_ms']:6.1f} "
                  f"{row['gpu_xfer_ms']:11.1f} {row['gpu_comp_ms']:11.1f} "
                  f"{row['speedup_gpu_comp']:8.2f} {vram:7.0f}{flag}", flush=True)

    with open(RESULTS, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(rows)} rows -> {RESULTS}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="small,medium")
    ap.add_argument("--n", type=int, default=7)
    a = ap.parse_args()
    run(a.sizes.split(","), a.n)


if __name__ == "__main__":
    main()
