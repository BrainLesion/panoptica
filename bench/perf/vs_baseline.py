"""panoptica vs the reference standard: must beat it on CPU *and* GPU, all dims.

Times the published reference (`bench.baseline`, its own venv) against our cpu
and gpu backends on the same generated pair. `cpu_x` = reference / ours-cpu;
`gpu_x` = reference / ours-gpu-comp. A cell where the reference errors (e.g.
uint8 label overflow past 255 instances) is a trivial win — we run, it can't —
and is reported `REF-ERR`. Anything with `cpu_x < 1` or `gpu_x < 1` is a GAP.

    python bench/perf/vs_baseline.py [--sizes small,2d,...] [--n 7] [--gpu]
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

from baseline import baseline_available, run_baseline  # noqa: E402
from datasets.generate import (  # noqa: E402
    generate_matched_instance_pair,
    generate_semantic_pair,
    generate_unmatched_instance_pair,
    get_spacing,
)

from panoptica import Evaluator, InputType  # noqa: E402

ALL = ["DSC", "IOU", "ASSD", "RVD", "RVAE", "CEDI", "HD", "HD95", "NSD", "clDSC"]
RESULTS = Path(__file__).resolve().parents[1] / "results" / "vs_baseline.jsonl"


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
    return statistics.median(ts)


def _gpu_sync():
    import cupy

    cupy.cuda.Stream.null.synchronize()


def run(sizes, n, gpu):
    RESULTS.parent.mkdir(exist_ok=True)
    rows = []
    print(f"\n{'size':16} {'input':10} {'ref_ms':>9} {'cpu_ms':>9} {'gpu_ms':>9} "
          f"{'gpuxfer':>9} {'cpu_x':>6} {'gpu_x':>6}  verdict")
    print("-" * 96)
    for size in sizes:
        for it in ("MATCHED_INSTANCE", "UNMATCHED_INSTANCE", "SEMANTIC"):
            ref, pred = _pair(size, it)
            sp = get_spacing(size)
            spacing = sp if any(x != 1.0 for x in sp) else None

            # reference (own venv). ref order matches ours: evaluate(pred, ref).
            try:
                rb = run_baseline(pred, ref, it, instance_metrics=ALL,
                                  spacing=spacing, time_n=n)
                ref_ms = rb["time_ms"]
            except Exception as e:
                ref_ms = None
                ref_err = type(e).__name__

            cpu_ms = _time(_eval(ref, pred, it, "cpu", spacing), n) * 1e3

            gpu_ms = gpu_xfer_ms = None
            if gpu:
                import cupy

                cupy.get_default_memory_pool().free_all_blocks()
                # xfer: numpy inputs, host->device copy inside the timed region
                gpu_xfer_ms = _time(_eval(ref, pred, it, "cuda", spacing), n,
                                    sync=_gpu_sync) * 1e3
                # comp: arrays pre-moved, kernel time only
                rd, pd = cupy.asarray(ref), cupy.asarray(pred)
                gpu_ms = _time(_eval(rd, pd, it, "cuda", spacing), n,
                               sync=_gpu_sync) * 1e3

            cpu_x = ref_ms / cpu_ms if ref_ms else None
            gpu_x = ref_ms / gpu_ms if (ref_ms and gpu_ms) else None
            if ref_ms is None:
                verdict = f"REF-ERR({ref_err}) -> we win (ref can't run)"
            else:
                bad = [d for d, x in (("cpu", cpu_x), ("gpu", gpu_x))
                       if x is not None and x < 1.0]
                verdict = "OK" if not bad else "GAP:" + ",".join(bad)

            rows.append(dict(size=size, input_type=it, ref_ms=ref_ms,
                             cpu_ms=cpu_ms, gpu_ms=gpu_ms, gpu_xfer_ms=gpu_xfer_ms,
                             cpu_x=cpu_x, gpu_x=gpu_x, verdict=verdict, n=n))
            f = lambda v, w, p=1: (f"{v:{w}.{p}f}" if v is not None else f"{'n/a':>{w}}")
            print(f"{size:16} {it[:10]:10} {f(ref_ms,9)} {cpu_ms:9.1f} "
                  f"{f(gpu_ms,9)} {f(gpu_xfer_ms,9)} {f(cpu_x,6,2)} {f(gpu_x,6,2)}  {verdict}",
                  flush=True)

    with open(RESULTS, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(rows)} rows -> {RESULTS}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="small,medium,2d")
    ap.add_argument("--n", type=int, default=7)
    ap.add_argument("--gpu", action="store_true")
    a = ap.parse_args()
    if not baseline_available():
        print("baseline venv missing; run: python -c "
              "'from bench.baseline import ensure_baseline; ensure_baseline()'")
        return
    run(a.sizes.split(","), a.n, a.gpu)


if __name__ == "__main__":
    main()
