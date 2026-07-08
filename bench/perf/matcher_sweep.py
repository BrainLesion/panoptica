"""Matcher scaling sweep: naive vs bipartite (Hungarian) vs merge.

The main perf harness fixes the matcher at ``naive`` — but matching is the GPU's
headline win, and ``bipartite`` is the Hungarian algorithm (superlinear in
instance count). This isolates that: a FIXED 256^3 volume, instance count swept,
a MINIMAL metric set (DSC only) so matcher time dominates the measurement.
UNMATCHED input so matching actually runs.

    python bench/perf/matcher_sweep.py [--counts 50,200,500,1000,2000] [--n 3] [--gpu]
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

from datasets.generate import _make_prediction, _make_reference  # noqa: E402

from panoptica import Evaluator, InputType  # noqa: E402

SHAPE = (256, 256, 256)
RESULTS = Path(__file__).resolve().parents[1] / "results" / "matcher_sweep.jsonl"


def _pair(n_req):
    """Fixed-shape ref with ~n_req instances + a perturbed pred (shift/drop)."""
    rng = np.random.default_rng(1234)
    ref, placed = _make_reference(SHAPE, n_req, rng, rmin=2, rmax=4, morphology="ball")
    pred = _make_prediction(ref, rng, shift=1, drop_frac=0.1)
    return ref, pred, placed


def _time(fn, n, sync=None):
    fn()  # warmup
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


def _eval(ref, pred, matcher, device):
    ev = Evaluator(
        InputType.UNMATCHED_INSTANCE, instance_metrics=["DSC"],
        matcher=matcher, matching_metric="IOU", matching_threshold=0.5, device=device,
    )
    return lambda: ev.evaluate(pred, ref)


def run(counts, n, gpu):
    RESULTS.parent.mkdir(exist_ok=True)
    matchers = ("naive", "bipartite", "merge")
    devices = ["cpu"] + (["cuda"] if gpu else [])
    rows = []
    hdr = f"{'n_req':>6} {'placed':>7}"
    for d in devices:
        for m in matchers:
            hdr += f" {m[:4]+'_'+d[:3]:>12}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for n_req in counts:
        ref, pred, placed = _pair(n_req)
        line = f"{n_req:6d} {placed:7d}"
        rec = {"n_req": n_req, "placed": placed}
        for d in devices:
            sync = _gpu_sync if d == "cuda" else None
            for m in matchers:
                t = _time(_eval(ref, pred, m, d), n, sync=sync) * 1e3
                rec[f"{m}_{d}_ms"] = t
                line += f" {t:12.1f}"
        rows.append(rec)
        print(line, flush=True)
    with open(RESULTS, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(rows)} rows -> {RESULTS}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", default="50,200,500,1000,2000")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--gpu", action="store_true")
    a = ap.parse_args()
    run([int(x) for x in a.counts.split(",")], a.n, a.gpu)


if __name__ == "__main__":
    main()
