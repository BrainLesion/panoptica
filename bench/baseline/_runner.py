"""Runs INSIDE the reference-standard venv (see :mod:`bench.baseline`).

Reads a params JSON, evaluates with the reference ``panoptica``, and prints one
JSON object to stdout: ``{group: {"scalars": {...}, "lists": {metric: [...]}}}``.

This process must see ONLY the installed v1 ``panoptica`` — never import the
rewritten package here, and rely on the caller's ``cwd`` being outside the repo
root so the local ``panoptica/`` dir can't shadow it.
"""

import json
import statistics
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from panoptica import (
    ConnectedComponentsInstanceApproximator,
    InputType,
    MaxBipartiteMatching,
    NaiveThresholdMatching,
    Panoptica_Evaluator,
)
from panoptica.instance_matcher import MaximizeMergeMatching
from panoptica.metrics import Metric, MetricMode
from panoptica.utils.label_group import LabelGroup
from panoptica.utils.segmentation_class import SegmentationClassGroups

_MATCHERS = {
    "naive": NaiveThresholdMatching,
    "bipartite": MaxBipartiteMatching,
    "merge": MaximizeMergeMatching,
}


def _num(v):
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def main(params_path, out_path):
    p = json.loads(open(params_path).read())
    arrays = np.load(p["npz"])
    a, b = arrays["a"], arrays["b"]
    metrics = [getattr(Metric, m) for m in p["instance_metrics"]]

    kw = {}
    if p["global_metrics"]:
        kw["global_metrics"] = [getattr(Metric, m) for m in p["global_metrics"]]
    if p["groups"]:
        # A group value is either [labels] or {"labels": [...], "single_instance": bool}.
        def _lg(v):
            if isinstance(v, dict):
                return LabelGroup(list(v["labels"]), single_instance=v.get("single_instance", False))
            return LabelGroup(list(v))

        kw["segmentation_class_groups"] = SegmentationClassGroups(
            {name: _lg(v) for name, v in p["groups"].items()}
        )

    ev = Panoptica_Evaluator(
        expected_input=getattr(InputType, p["expected_input"]),
        instance_approximator=ConnectedComponentsInstanceApproximator(),
        instance_matcher=_MATCHERS[p["matcher"]](Metric.IOU, p["threshold"]),
        instance_metrics=metrics,
        verbose=False,
        log_times=False,
        **kw,
    )
    evkw = {"voxelspacing": tuple(p["spacing"])} if p["spacing"] else {}

    if p.get("time_n"):  # timing mode: median-of-N inside the venv (no startup)
        res = ev.evaluate(a, b, verbose=False, **evkw)  # warmup
        ts = []
        for _ in range(p["time_n"]):
            s = time.perf_counter()
            ev.evaluate(a, b, verbose=False, **evkw)
            ts.append(time.perf_counter() - s)
        tp = res["ungrouped"].to_dict().get("tp")
        _write(out_path, {"time_ms": statistics.median(ts) * 1e3, "tp": _num(tp)})
        return

    res = ev.evaluate(a, b, verbose=False, **evkw)
    out = {}
    for gname, gres in res.items():
        lists = {}
        for m in metrics:
            try:
                vals = gres.get_list_metric(m, MetricMode.ALL)
            except Exception:
                vals = None
            if vals is not None:
                lists[m.name.lower()] = [float(v) for v in vals]
        out[gname] = {
            "scalars": {k: _num(v) for k, v in gres.to_dict().items()},
            "lists": lists,
        }
    _write(out_path, out)


def _write(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
