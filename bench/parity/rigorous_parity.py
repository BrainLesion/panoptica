"""Rigorous self-parity: device parity (cpu vs cuda) + self-consistency.

panoptica is the only version, so parity is checked *internally* rather
than against a v1 oracle:

  * Device parity -- the same config evaluated on cpu and on cuda must agree,
    scalars AND every metric's per-instance list. This is the differential net:
    a backend-specific bug (an xp mismatch, an fp32 reduction error, a CCL
    ordering divergence) shows up as a cpu/cuda disagreement.
  * Self-consistency -- invariants that must hold regardless of backend:
      - grouped eval of a class group == ungrouped eval of the pair masked to
        that group's labels (the group-decomposition property),
      - surface distances scale linearly with voxel spacing while overlap
        metrics / counts are spacing-invariant.

Per-instance alignment: MATCHED input has stable ids -> element-wise compare.
SEMANTIC/UNMATCHED ids can differ across CCL backends -> compare the SORTED
lists (multiset equality).

Usage:
  python bench/parity/rigorous_parity.py                 # default matrix (small+medium)
  python bench/parity/rigorous_parity.py --large         # add 512^3 (slow)
  python bench/parity/rigorous_parity.py --real          # add real spine_seg data
Device-parity cells auto-skip when no CUDA device / cupy is present; the
self-consistency cells always run.
"""

from __future__ import annotations

import argparse
import math
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from datasets.generate import (  # noqa: E402
    generate_matched_instance_pair,
    generate_semantic_pair,
    generate_unmatched_instance_pair,
)
from tolerances import compare  # noqa: E402

from panoptica import Evaluator, InputType  # noqa: E402

ALL = ["DSC", "IOU", "ASSD", "RVD", "RVAE", "CEDI", "HD", "HD95", "NSD", "clDSC"]

_METRIC_TOKENS = [
    ("hd95", "hd95"), ("cldsc", "cldice"), ("assd", "assd"), ("cedi", "cedi"),
    ("rvae", "rvae"), ("rvd", "rvd"), ("nsd", "nsd"), ("dsc", "dsc"),
    ("iou", "iou"), ("hd", "hd"),
]
_EXACT = {
    "tp": "tp", "fp": "fp", "fn": "fn", "prec": "precision", "rec": "recall",
    "rq": "rq", "sq": "sq", "pq": "pq", "sq_std": "sq",
}


def _cuda_available() -> bool:
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _resolve_metric(key: str) -> str | None:
    k = key.lower()
    if k in _EXACT:
        return _EXACT[k]
    for token, name in _METRIC_TOKENS:
        if token in k:
            return name
    return None


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _both_nan(a, b) -> bool:
    return (isinstance(a, float) and math.isnan(a)) and (
        isinstance(b, float) and math.isnan(b)
    )


def _pair(size, it):
    if it == "MATCHED_INSTANCE":
        ref, pred, _ = generate_matched_instance_pair(size=size)
        return ref, pred
    if it == "UNMATCHED_INSTANCE":
        return generate_unmatched_instance_pair(size=size)
    return generate_semantic_pair(size=size)


def _run_eval(ref, pred, it, matcher, thr, device, spacing=None, groups=None, strict=False):
    ev = Evaluator(
        getattr(InputType, it),
        instance_metrics=ALL,
        global_metrics=["DSC"],
        matcher=matcher,
        matching_metric="IOU",
        matching_threshold=thr,
        strict_threshold=strict,
        device=device,
        segmentation_class_groups=groups,
    )
    return ev.evaluate(ref, pred, spacing=spacing)


def _flatten(d):
    """EvalResult.to_dict() -> (scalars, per-instance lists), host floats."""
    scalars, lists = {}, {}
    for k, v in d.items():
        if hasattr(v, "get"):  # cupy -> numpy (host)
            v = v.get()
        arr = np.asarray(v)
        if arr.ndim == 0:
            scalars[k] = float(arr)
        elif k.endswith("_all"):
            lists[k[:-4]] = [float(x) for x in arr.ravel()]
    return scalars, lists


def _cmp_scalars(d1, d2):
    fails, checked, skipped = [], 0, 0
    for k in sorted(set(d1) & set(d2)):
        a, b = d1[k], d2[k]
        if not (_num(a) and _num(b)) or _both_nan(a, b) or a == b:
            continue
        metric = _resolve_metric(k)
        if metric is None:
            skipped += 1
            continue
        checked += 1
        if not compare(a, b, metric):
            fails.append(("scalar:" + k, a, b))
    return checked, skipped, fails


def _cmp_lists(l1, l2, it):
    """Per-instance lists: element-wise for MATCHED, sorted otherwise."""
    fails, checked = [], 0
    for name in sorted(set(l1) & set(l2)):
        a, b = l1[name], l2[name]
        if len(a) != len(b):
            fails.append((f"list:{name}:len", len(a), len(b)))
            continue
        metric = _resolve_metric(name) or name
        va = a if it == "MATCHED_INSTANCE" else sorted(a)
        vb = b if it == "MATCHED_INSTANCE" else sorted(b)
        for i, (x, y) in enumerate(zip(va, vb)):
            if _both_nan(x, y) or x == y:
                continue
            checked += 1
            if not compare(x, y, metric):
                fails.append((f"list:{name}[{i}]", x, y))
                break
    return checked, fails


def run_device_cell(size, it, matcher, thr, label, pair=None, spacing=None, strict=False):
    """Device parity: same config on cpu vs cuda must agree."""
    if not _cuda_available():
        print(f"[{label:44}] SKIP (no cuda device)", flush=True)
        return 0
    ref, pred = pair if pair is not None else _pair(size, it)
    s_cpu, l_cpu = _flatten(_run_eval(ref, pred, it, matcher, thr, "cpu", spacing, strict=strict).to_dict())
    s_gpu, l_gpu = _flatten(_run_eval(ref, pred, it, matcher, thr, "cuda", spacing, strict=strict).to_dict())
    cs, skipped, sf = _cmp_scalars(s_cpu, s_gpu)
    cl, lf = _cmp_lists(l_cpu, l_gpu, it)
    fails = sf + lf
    status = "PASS" if not fails else f"FAIL({len(fails)})"
    print(
        f"[{label:44}] scalars:{cs:3d} lists:{cl:4d} skip:{skipped:2d} -> {status}",
        flush=True,
    )
    for k, a, b in fails[:8]:
        print(f"      {k:26} cpu={a!r:>16} cuda={b!r:>16}")
    return len(fails)


def run_skip_groups_cell(label):
    """Device parity for grouped eval + skip_groups: cpu vs cuda, surviving groups."""
    if not _cuda_available():
        print(f"[{label:44}] SKIP (no cuda device)", flush=True)
        return 0
    from panoptica.core.labels import LabelGroup, SegmentationClassGroups

    ref, pred, _ = generate_matched_instance_pair(size="small")
    groups = SegmentationClassGroups(
        {"a": LabelGroup([1, 2, 3]), "b": LabelGroup([4, 5, 6, 7, 8, 9, 10])}
    )

    def _run(device):
        ev = Evaluator(
            InputType.MATCHED_INSTANCE, instance_metrics=ALL, global_metrics=["DSC"],
            segmentation_class_groups=groups, device=device,
        )
        return ev.evaluate(ref, pred, skip_groups=["a"])

    out_cpu, out_gpu = _run("cpu"), _run("cuda")
    if set(out_cpu) != {"b"} or set(out_gpu) != {"b"}:
        print(f"[{label:44}] FAIL(skip did not drop group 'a')", flush=True)
        return 1
    s_cpu, _ = _flatten(out_cpu["b"].to_dict())
    s_gpu, _ = _flatten(out_gpu["b"].to_dict())
    _, skipped, sf = _cmp_scalars(s_cpu, s_gpu)
    status = "PASS" if not sf else f"FAIL({len(sf)})"
    print(f"[{label:44}] surviving-group scalars cpu==cuda -> {status}", flush=True)
    return len(sf)


def _run_spacing_invariance(label):
    """Surface distances scale linearly with voxel spacing; overlap does not.

    Evaluate the same volume at spacing s and k*s: ASSD/HD/HD95 must scale by k
    (to ~1e-9), while DSC/IOU and tp/fp/fn are spacing-invariant.
    """
    ref, pred = _pair("small", "SEMANTIC")
    ev = Evaluator(
        InputType.SEMANTIC, instance_metrics=["DSC", "ASSD", "HD", "HD95"],
        matcher="naive", matching_threshold=0.5, device="cpu",
    )
    k = 2.5
    base = ev.evaluate(ref, pred, spacing=(1.0, 1.0, 1.0)).to_dict()
    scaled = ev.evaluate(ref, pred, spacing=(k, k, k)).to_dict()
    fails = []
    for m in ("assd_avg", "hd_avg", "hd95_avg"):
        if not compare(base[m] * k, scaled[m], "assd"):
            fails.append((f"scale:{m}", base[m] * k, scaled[m]))
    for m in ("dsc_avg", "tp", "fp", "fn"):
        if not compare(base[m], scaled[m], _resolve_metric(m) or "dsc"):
            fails.append((f"invariant:{m}", base[m], scaled[m]))
    status = "PASS" if not fails else f"FAIL({len(fails)})"
    print(f"[{label:44}] checks:  7 -> {status}", flush=True)
    for k_, a, b in fails[:8]:
        print(f"      {k_:26} expected={a!r:>16} got={b!r:>16}")
    return len(fails)


def _load_spine(it):
    """Real spine segmentation pair (examples/spine_seg/<kind>/{ref,pred}.nii.gz)."""
    import nibabel as nib

    kind = {
        "MATCHED_INSTANCE": "matched_instance",
        "UNMATCHED_INSTANCE": "unmatched_instance",
        "SEMANTIC": "semantic",
    }[it]
    root = Path(__file__).resolve().parents[2] / "examples" / "spine_seg" / kind
    ref = np.asarray(nib.load(str(root / "ref.nii.gz")).dataobj).astype(np.uint32)
    pred = np.asarray(nib.load(str(root / "pred.nii.gz")).dataobj).astype(np.uint32)
    return ref, pred


def _grouped_pair():
    """Two-class-group labelled pair (groups a={1,2,3}, b={4,5,6}).

    Deterministic, well-separated boxes so every group has matched instances
    (TP>0), plus one FN (ref-only) and one FP (pred-only) box so detection
    counts are exercised.
    """
    ref = np.zeros((64, 64, 64), np.uint32)
    pred = np.zeros((64, 64, 64), np.uint32)
    # A 6x2 grid of boxes; base label = 1..6, two matched boxes each.
    for base in range(1, 7):
        for j in range(2):
            z = 4 + (base - 1) * 9
            y = 4 + j * 30
            ref[z : z + 6, y : y + 6, 4:10] = base
            sh = 1 if (base + j) % 2 else 0  # small shift -> partial overlap
            pred[z : z + 6, y + sh : y + 6 + sh, 4:10] = base
    ref[4:10, 4:10, 50:56] = 1  # FN in group a (no matching pred)
    pred[4:10, 20:26, 50:56] = 4  # FP in group b (no matching ref)
    return ref, pred


def _group_defs():
    from panoptica.core.labels import LabelGroup, SegmentationClassGroups

    return SegmentationClassGroups(
        groups={"a": LabelGroup((1, 2, 3)), "b": LabelGroup((4, 5, 6))}
    )


def _mask_to_labels(arr, labels):
    """Keep only the given labels; zero everything else."""
    return np.where(np.isin(arr, list(labels)), arr, 0).astype(arr.dtype)


def run_group_cell(it, matcher, label):
    """Self-consistency: grouped eval of a group == ungrouped eval of the pair
    masked to that group's labels (group-constrained matching decomposes).
    """
    ref, pred = _grouped_pair()
    grouped = _run_eval(ref, pred, it, matcher, 0.5, "cpu", groups=_group_defs())
    group_labels = {"a": (1, 2, 3), "b": (4, 5, 6)}
    total_checked, fails = 0, []
    for g, labels in group_labels.items():
        ref_m = _mask_to_labels(ref, labels)
        pred_m = _mask_to_labels(pred, labels)
        ref_res = _run_eval(ref_m, pred_m, it, matcher, 0.5, "cpu")
        s_group, _ = _flatten(grouped[g].to_dict())
        s_ref, _ = _flatten(ref_res.to_dict())
        cs, _sk, sf = _cmp_scalars(s_ref, s_group)
        total_checked += cs
        fails += [(f"{g}:{k}", a, b) for k, a, b in sf]
    status = "PASS" if not fails else f"FAIL({len(fails)})"
    print(f"[{label:44}] scalars:{total_checked:3d} -> {status}", flush=True)
    for k, a, b in fails[:8]:
        print(f"      {k:26} masked={a!r:>16} grouped={b!r:>16}")
    return len(fails)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--large", action="store_true")
    ap.add_argument("--real", action="store_true")
    a = ap.parse_args()

    if not _cuda_available():
        print("NOTE: no cuda device -> device-parity cells SKIP; "
              "self-consistency cells still run.\n")

    sizes = ["small", "medium"] + (["large"] if a.large else [])
    inputs = ["MATCHED_INSTANCE", "UNMATCHED_INSTANCE", "SEMANTIC"]
    total = 0

    # 1) device parity: sizes x input-types (naive matcher)
    for size in sizes:
        for it in inputs:
            total += run_device_cell(size, it, "naive", 0.5, f"{size}/{it}/naive")

    # 2) device parity: all matchers on small (instance inputs need instances)
    for matcher in ("bipartite", "merge"):
        for it in ("UNMATCHED_INSTANCE", "SEMANTIC"):
            total += run_device_cell("small", it, matcher, 0.5, f"small/{it}/{matcher}")

    # 3) device parity: threshold variations
    for thr in (0.1, 0.75):
        total += run_device_cell("small", "UNMATCHED_INSTANCE", "naive", thr,
                                 f"small/UNMATCHED/naive@{thr}")

    # 3b) device parity: strict threshold (>/<), all matchers
    for matcher in ("naive", "bipartite", "merge"):
        total += run_device_cell(
            "small", "UNMATCHED_INSTANCE", matcher, 0.5,
            f"small/UNMATCHED/{matcher}+strict", strict=True,
        )

    # 3c) device parity: grouped eval + skip_groups
    total += run_skip_groups_cell("groups/skip_groups/device")

    # 4) device parity: 2D
    total += run_device_cell("2d", "SEMANTIC", "naive", 0.5, "2d/SEMANTIC/naive")

    # 5) self-consistency: multi-group decomposition (self-consistency, cpu)
    for it in ("UNMATCHED_INSTANCE", "SEMANTIC"):
        for matcher in ("naive", "bipartite", "merge"):
            total += run_group_cell(it, matcher, f"groups/{it}/{matcher}")

    # 6) self-consistency: anisotropic spacing scale-invariance (internal)
    total += _run_spacing_invariance("aniso/scale-invariance")

    # 7) device parity: real spine data (512x512x17), all input types
    if a.real:
        for it in inputs:
            ref, pred = _load_spine(it)
            total += run_device_cell(
                "real", it, "naive", 0.5, f"real-spine/{it}/naive",
                pair=(ref, pred),
            )

    print(f"\n{'='*70}\nTOTAL FAILURES: {total}")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
