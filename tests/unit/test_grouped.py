"""Vectorized multi-group evaluation: self-consistency of the batched pass.

panoptica evaluates all class groups in one batched pass with a same-group matching
constraint. That constraint makes grouped evaluation *decompose*: a group's
result must equal an ordinary ungrouped evaluation of the pair masked to that
group's labels. We check that invariant directly (self-consistency) across input types,
matchers, and ``single_instance`` groups -- no external reference needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from panoptica import Evaluator, InputType
from panoptica.core.labels import LabelGroup, SegmentationClassGroups

METS = ["DSC", "IOU", "ASSD", "RVD", "HD95", "NSD", "clDSC"]
# Cross-checked keys. SQ/PQ collapse each metric's per-instance avg into one
# scalar, so a wrong per-instance value shows here.
CHECK = ["tp", "fp", "fn", "prec", "rec", "rq", "sq", "pq", "global_bin_dsc"] + [
    f"sq_{m.lower()}" for m in METS if m != "IOU"
]


def _mask_to_labels(arr, labels):
    """Keep only the given labels; zero everything else."""
    return np.where(np.isin(arr, list(labels)), arr, 0).astype(arr.dtype)


def _two_group_pair():
    ref = np.zeros((24, 24), np.uint32)
    pred = np.zeros((24, 24), np.uint32)
    ref[2:7, 2:7] = 1
    pred[2:6, 2:6] = 1
    ref[2:7, 12:17] = 2
    pred[3:7, 12:17] = 2
    ref[14:19, 2:7] = 3
    pred[14:19, 3:8] = 3
    ref[14:19, 14:19] = 4  # FN in group b
    pred[9:12, 9:12] = 4  # FP in group b
    return ref, pred


def _run_eval(ref, pred, it, matcher, groups=None):
    ev = Evaluator(
        it,
        instance_metrics=METS,
        global_metrics=["DSC"],
        matcher=matcher,
        matching_threshold=0.5,
        device="cpu",
        segmentation_class_groups=groups,
    )
    return ev.evaluate(pred, ref)


def _assert_decomposes(ref, pred, it, matcher, grouped, group_labels):
    """Each grouped[g] == ungrouped eval of the pair masked to that group."""
    for g, labels in group_labels.items():
        ref_m = _mask_to_labels(ref, labels)
        pred_m = _mask_to_labels(pred, labels)
        masked = _run_eval(ref_m, pred_m, it, matcher)
        gres = grouped[g]
        for k in CHECK:
            a, b = masked.get(k), gres.get(k)
            if a is None or b is None:
                continue
            a, b = float(a), float(b)
            if np.isnan(a) and np.isnan(b):
                continue
            assert abs(a - b) < 1e-6, f"group {g!r} key {k!r}: masked={a} grouped={b}"


@pytest.mark.parametrize("it", [InputType.SEMANTIC, InputType.UNMATCHED_INSTANCE])
@pytest.mark.parametrize("matcher", ["naive", "bipartite", "merge"])
def test_two_groups_decompose(it, matcher):
    ref, pred = _two_group_pair()
    o2 = _run_eval(
        ref,
        pred,
        it,
        matcher,
        SegmentationClassGroups(
            groups={"a": LabelGroup((1, 2)), "b": LabelGroup((3, 4))}
        ),
    )
    assert set(o2.keys()) == {"a", "b"}
    _assert_decomposes(ref, pred, it, matcher, o2, {"a": (1, 2), "b": (3, 4)})


def test_single_instance_group_matches_any_overlap():
    """A single_instance group matches on any overlap (below-threshold too);
    the normal ``vert`` group still decomposes to its masked eval."""
    ref = np.zeros((24, 24), np.uint32)
    pred = np.zeros((24, 24), np.uint32)
    ref[2:7, 2:7] = 1
    pred[2:6, 2:6] = 1
    ref[2:7, 12:17] = 2
    pred[3:7, 12:17] = 2
    ref[14:20, 14:20] = 26  # sacrum ref
    pred[16:20, 16:22] = 26  # sacrum pred, IoU < 0.5

    groups = SegmentationClassGroups(
        groups={
            "vert": LabelGroup((1, 2)),
            "sacrum": LabelGroup((26,), single_instance=True),
        }
    )
    o2 = _run_eval(ref, pred, InputType.UNMATCHED_INSTANCE, "naive", groups)
    assert o2["sacrum"].get("tp") == 1  # matched despite low overlap
    # the ordinary group decomposes; single_instance's low-overlap match is its
    # own semantic (not reproducible by a thresholded ungrouped eval).
    _assert_decomposes(
        ref, pred, InputType.UNMATCHED_INSTANCE, "naive", o2, {"vert": (1, 2)}
    )


def test_ungrouped_returns_single_result():
    """No groups -> a single EvalResult, not a dict (unchanged default)."""
    ref = np.zeros((12, 12), np.uint32)
    pred = np.zeros((12, 12), np.uint32)
    ref[2:5, 2:5] = 1
    pred[2:5, 2:5] = 1
    r = Evaluator(InputType.MATCHED_INSTANCE, device="cpu").evaluate(pred, ref)
    assert not isinstance(r, dict)
    assert r.get("tp") == 1


def test_spacing_scales_surface_distances():
    """Surface distances scale with voxel spacing; overlap metrics do not."""
    ref = np.zeros((16, 16, 16), np.uint32)
    pred = np.zeros((16, 16, 16), np.uint32)
    ref[3:9, 3:9, 3:9] = 1
    pred[4:10, 3:9, 3:9] = 1
    ev = Evaluator(
        InputType.MATCHED_INSTANCE,
        instance_metrics=["DSC", "ASSD", "HD95"],
        device="cpu",
    )
    base = ev.evaluate(pred, ref, spacing=(1.0, 1.0, 1.0))
    k = 2.5
    scaled = ev.evaluate(pred, ref, spacing=(k, k, k))
    assert abs(base.get("assd_avg") * k - scaled.get("assd_avg")) < 1e-9
    assert abs(base.get("hd95_avg") * k - scaled.get("hd95_avg")) < 1e-9
    assert abs(base.get("dsc_avg") - scaled.get("dsc_avg")) < 1e-12
    assert base.get("tp") == scaled.get("tp")


def test_regionwise_equals_masked_eval():
    """Per-region (Voronoi) eval == ordinary UNMATCHED eval on the region-masked
    pair. the Voronoi partition splits the volume into regions that are each a
    normal unmatched pair, so region i must equal an ungrouped UNMATCHED
    evaluation of the region-masked (ref*mask, pred*mask) volume.
    """
    from panoptica.backends.namespace import resolve
    from panoptica.kernels.voronoi import voronoi_regions

    ref = np.zeros((1, 48, 48), np.uint32)
    pred = np.zeros((1, 48, 48), np.uint32)
    ref[0, 3:9, 3:9] = 1
    pred[0, 3:9, 4:10] = 1
    ref[0, 3:9, 24:30] = 2
    pred[0, 4:10, 24:30] = 2
    ref[0, 24:30, 12:18] = 3
    pred[0, 25:31, 12:18] = 3
    pred[0, 36:40, 36:40] = 7  # stray FP -> attributed to one region

    mets = ["DSC", "IOU", "ASSD", "HD95"]
    out = Evaluator(
        InputType.UNMATCHED_INSTANCE,
        instance_metrics=mets,
        global_metrics=["DSC"],
        matcher="naive",
        matching_threshold=0.5,
        device="cpu",
        per_region_evaluation=True,
    ).evaluate(pred, ref)
    assert set(out) == {"regions", "region_avg_global_bin_dsc"}

    xp, _ = resolve("cpu")
    rm, n = voronoi_regions(ref[0], xp)
    rm = np.asarray(rm)[None]
    assert n == 3
    ev = Evaluator(
        InputType.UNMATCHED_INSTANCE,
        instance_metrics=mets,
        global_metrics=["DSC"],
        matcher="naive",
        matching_threshold=0.5,
        device="cpu",
    )
    for i in range(1, n + 1):
        mask = rm == i
        r1 = ev.evaluate(
            (pred * mask).astype(np.uint32),
            (ref * mask).astype(np.uint32),
        )
        r2 = out["regions"][i]
        for k in [
            "tp",
            "fp",
            "fn",
            "sq",
            "pq",
            "sq_dsc",
            "sq_assd",
            "sq_hd95",
            "global_bin_dsc",
        ]:
            a, b = r1.get(k), r2.get(k)
            if a is None or b is None:
                continue
            a, b = float(a), float(b)
            if np.isnan(a) and np.isnan(b):
                continue
            assert abs(a - b) < 1e-6, f"region {i} key {k!r}: masked={a} region={b}"


def test_skip_groups_omits_and_warns():
    # Two class groups; skipping one drops it from the result dict without
    # affecting the other, and an unknown name warns but is ignored.
    ref = np.zeros((12, 12), np.uint32)
    pred = np.zeros((12, 12), np.uint32)
    ref[1:4, 1:4] = 1
    pred[1:4, 1:4] = 1  # group a
    ref[7:11, 7:11] = 2
    pred[7:11, 7:11] = 2  # group b
    sg = SegmentationClassGroups({"a": LabelGroup([1]), "b": LabelGroup([2])})
    ev = Evaluator(
        InputType.MATCHED_INSTANCE,
        instance_metrics=["DSC"],
        segmentation_class_groups=sg,
        device="cpu",
    )

    full = ev.evaluate(pred, ref)
    assert set(full) == {"a", "b"}

    skipped = ev.evaluate(pred, ref, skip_groups=["b"])
    assert set(skipped) == {"a"}
    # the surviving group is unchanged by the skip
    assert skipped["a"].to_dict()["tp"] == full["a"].to_dict()["tp"]

    with pytest.warns(UserWarning, match="unknown group"):
        out = ev.evaluate(pred, ref, skip_groups=["does_not_exist"])
    assert set(out) == {"a", "b"}  # unknown name ignored, nothing skipped
