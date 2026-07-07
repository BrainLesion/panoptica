"""YAML config: native round-trip + v1 preset loading."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from panoptica import Evaluator, InputType

_CONFIGS = Path(__file__).resolve().parents[2] / "panoptica" / "configs"


def test_native_round_trip(tmp_path):
    ev = Evaluator(
        InputType.UNMATCHED_INSTANCE,
        instance_metrics=["DSC", "ASSD"],
        global_metrics=["IOU"],
        matcher="bipartite",
        matching_threshold=0.3,
    )
    p = tmp_path / "cfg.yaml"
    ev.to_config(p)
    ev = Evaluator.from_config(p)
    assert ev.expected_input is InputType.UNMATCHED_INSTANCE
    assert ev.instance_metrics == ["DSC", "ASSD"]
    assert ev.global_metrics == ["IOU"]
    assert ev.matcher == "bipartite"
    assert ev.matching_threshold == 0.3


def test_v1_presets_load():
    presets = {
        "panoptica_evaluator_default.yaml": (InputType.SEMANTIC, 0.5),
        "panoptica_evaluator_BRATS.yaml": (InputType.SEMANTIC, 0.5),
        "panoptica_evaluator_binaryMS.yaml": (InputType.SEMANTIC, 0.1),
        "panoptica_evaluator_VERSE.yaml": (InputType.UNMATCHED_INSTANCE, 0.5),
    }
    for name, (it, thr) in presets.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ev = Evaluator.from_config(_CONFIGS / name)
        assert ev.expected_input is it
        assert ev.matching_threshold == thr
        assert "DSC" in ev.instance_metrics


def test_loaded_preset_evaluates_like_manual():
    ref = np.zeros((12, 12), dtype=np.uint32)
    pred = np.zeros((12, 12), dtype=np.uint32)
    ref[1:4, 1:4] = 1
    pred[1:4, 1:4] = 1
    ref[7:10, 7:10] = 2
    pred[7:10, 7:10] = 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ev = Evaluator.from_config(_CONFIGS / "panoptica_evaluator_default.yaml")
    ev.device = "cpu"
    r = ev.evaluate(pred, ref)
    manual = Evaluator(
        InputType.SEMANTIC,
        instance_metrics=ev.instance_metrics,
        matcher="naive",
        matching_threshold=0.5,
        device="cpu",
    ).evaluate(pred, ref)
    assert r.get("tp") == manual.get("tp")
    assert abs(r.get("dsc_avg") - manual.get("dsc_avg")) < 1e-12
