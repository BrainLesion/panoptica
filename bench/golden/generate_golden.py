"""Freeze panoptica's full scalar output on seeded pairs -> bench/golden/golden.json.

Complements the hand-verified absolute cases in tests/unit/test_golden.py: those
pin the *definitions*; this pins the *realistic seeded* outputs so any later drift
(a metric formula change, a matching regression) is caught even where a value is
too complex to hand-compute. Re-run deliberately when a change is intended:

    python bench/golden/generate_golden.py

Deterministic: the generator is seeded per key, so output is byte-reproducible.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from datasets.generate import (  # noqa: E402
    generate_matched_instance_pair,
    generate_semantic_pair,
    generate_unmatched_instance_pair,
)

from panoptica import Evaluator, InputType  # noqa: E402

ALL = ["DSC", "IOU", "ASSD", "RVD", "RVAE", "CEDI", "HD", "HD95", "NSD", "clDSC"]
KEYS = ["small", "ellipsoids"]  # one cubic, one non-spherical morphology
GOLDEN = Path(__file__).resolve().parent / "golden.json"


def _pair(size, it):
    if it == "MATCHED_INSTANCE":
        ref, pred, _ = generate_matched_instance_pair(size=size)
        return ref, pred
    if it == "UNMATCHED_INSTANCE":
        return generate_unmatched_instance_pair(size=size)
    return generate_semantic_pair(size=size)


def _scalars(size, it):
    ref, pred = _pair(size, it)
    d = Evaluator(getattr(InputType, it), instance_metrics=ALL,
                  global_metrics=["DSC"], device="cpu").evaluate(pred, ref).to_dict()
    out = {}
    for k, v in d.items():
        arr = np.asarray(v.get() if hasattr(v, "get") else v)
        if arr.ndim == 0:
            out[k] = float(arr)
    return out


def build():
    golden = {}
    for size in KEYS:
        for it in ("MATCHED_INSTANCE", "UNMATCHED_INSTANCE", "SEMANTIC"):
            golden[f"{size}/{it}"] = _scalars(size, it)
    return golden


if __name__ == "__main__":
    GOLDEN.write_text(json.dumps(build(), indent=2, sort_keys=True) + "\n")
    print(f"wrote {GOLDEN} ({len(build())} cells)")
