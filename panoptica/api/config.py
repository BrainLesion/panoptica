"""YAML config round-trip for the Evaluator, plus a loader for v1 presets.

- ``save_config(evaluator, path)`` / ``load_config(path)`` — native YAML.
- v1 preset YAMLs (``!Panoptica_Evaluator`` with ``!Metric``/``!InputType``/…
  tags, e.g. configs/panoptica_evaluator_BRATS.yaml) are auto-detected and
  mapped to a native Evaluator. Fields panoptica does not act on (custom edge-case handler,
  decision_metric, multi-group evaluation) are warned about, not silently dropped.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

from panoptica.core.enums import InputType
from panoptica.core.errors import InputValidationError
from panoptica.core.labels import LabelGroup, SegmentationClassGroups

_MATCHER_TAG_TO_NAME = {
    "NaiveThresholdMatching": "naive",
    "MaxBipartiteMatching": "bipartite",
    "MaximizeMergeMatching": "merge",
}


class _Tagged:
    """A value carrying its v1 YAML tag suffix (e.g. tag='Metric', value='DSC')."""

    def __init__(self, tag: str, value: Any) -> None:
        self.tag = tag
        self.value = value


def _v1_yaml():
    from ruamel.yaml import YAML
    from ruamel.yaml.nodes import ScalarNode, SequenceNode

    yaml = YAML(typ="safe", pure=True)

    def _multi(constructor, tag_suffix, node):
        if isinstance(node, ScalarNode):
            return _Tagged(tag_suffix, node.value)
        if isinstance(node, SequenceNode):
            return _Tagged(tag_suffix, constructor.construct_sequence(node, deep=True))
        return _Tagged(tag_suffix, constructor.construct_mapping(node, deep=True))

    yaml.constructor.add_multi_constructor("!", _multi)
    return yaml


def _untag(v: Any) -> Any:
    return v.value if isinstance(v, _Tagged) else v


def _metric_ids(seq) -> list[str]:
    return [str(_untag(m)) for m in (seq or [])]


def _parse_groups(sg: _Tagged) -> SegmentationClassGroups:
    groups: dict[str, LabelGroup] = {}
    for name, lg in (sg.value.get("groups") or {}).items():
        m = lg.value if isinstance(lg, _Tagged) else lg
        labels = m.get("value_labels") or m.get("value_labels", ())
        groups[str(name)] = LabelGroup(
            value_labels=tuple(int(x) for x in (labels or ())),
            single_instance=bool(m.get("single_instance", False)),
        )
    return SegmentationClassGroups(groups=groups)


def _from_v1_preset(doc: _Tagged) -> Evaluator:
    from panoptica.api.evaluator import Evaluator

    cfg = doc.value  # mapping of the !Panoptica_Evaluator node
    kwargs: dict[str, Any] = {}

    it = _untag(cfg.get("expected_input"))
    if it is not None:
        kwargs["expected_input"] = InputType[str(it)]

    kwargs["instance_metrics"] = _metric_ids(cfg.get("instance_metrics")) or None
    kwargs["global_metrics"] = _metric_ids(cfg.get("global_metrics")) or None

    matcher = cfg.get("instance_matcher")
    if isinstance(matcher, _Tagged):
        kwargs["matcher"] = _MATCHER_TAG_TO_NAME.get(matcher.tag, "naive")
        mm = matcher.value
        kwargs["matching_metric"] = str(_untag(mm.get("matching_metric", "IOU")))
        kwargs["matching_threshold"] = float(mm.get("matching_threshold", 0.5))
        kwargs["strict_threshold"] = bool(_untag(mm.get("strict_threshold", False)))

    appr = cfg.get("instance_approximator")
    if isinstance(appr, _Tagged):
        backend = appr.value.get("cca_backend")
        if backend is not None:
            warnings.warn(
                f"v1 cca_backend {backend!r} ignored; panoptica auto-selects CC backend."
            )

    # Warn on v1 features panoptica does not (yet) act on.
    if _untag(cfg.get("decision_metric")) is not None:
        warnings.warn(
            "v1 decision_metric is not applied by panoptica (matching decides TP/FP)."
        )
    sg = cfg.get("segmentation_class_groups")
    if isinstance(sg, _Tagged):
        kwargs["segmentation_class_groups"] = _parse_groups(sg)
    if isinstance(cfg.get("edge_case_handler"), _Tagged):
        warnings.warn(
            "v1 edge_case_handler ignored; panoptica uses its built-in per-metric policy."
        )

    return Evaluator(**{k: v for k, v in kwargs.items() if v is not None})


def load_config(path: str | Path) -> Evaluator:
    """Load an Evaluator from a native or a v1-preset YAML (auto-detected)."""
    from panoptica.api.evaluator import Evaluator

    text = Path(path).read_text()
    if text.lstrip().startswith("!Panoptica_Evaluator"):
        doc = _v1_yaml().load(text)
        return _from_v1_preset(doc)

    from ruamel.yaml import YAML

    d = YAML(typ="safe", pure=True).load(text) or {}
    if d.get("_format") != "panoptica-native":
        raise InputValidationError(
            f"{path}: not a native or v1-preset panoptica config."
        )
    it = d.get("expected_input", "MATCHED_INSTANCE")
    return Evaluator(
        expected_input=InputType[str(it)],
        instance_metrics=d.get("instance_metrics", ["DSC", "IOU", "ASSD", "RVD"]),
        global_metrics=d.get("global_metrics", ["DSC"]),
        matcher=d.get("matcher", "naive"),
        matching_metric=d.get("matching_metric", "IOU"),
        matching_threshold=float(d.get("matching_threshold", 0.5)),
        strict_threshold=bool(d.get("strict_threshold", False)),
        connectivity=d.get("connectivity"),
        device=d.get("device", "auto"),
        n_jobs=d.get("n_jobs"),
    )


def save_config(evaluator: Evaluator, path: str | Path) -> None:
    """Write a native YAML config that ``load_config`` round-trips."""
    from ruamel.yaml import YAML

    d = {
        "_format": "panoptica-native",
        "expected_input": evaluator.expected_input.name,
        "instance_metrics": list(evaluator.instance_metrics),
        "global_metrics": list(evaluator.global_metrics),
        "matcher": evaluator.matcher,
        "matching_metric": evaluator.matching_metric,
        "matching_threshold": evaluator.matching_threshold,
        "strict_threshold": evaluator.strict_threshold,
        "connectivity": evaluator.connectivity,
        "device": evaluator.device,
        "n_jobs": evaluator.n_jobs,
    }
    yaml = YAML()
    yaml.default_flow_style = False
    with open(path, "w") as f:
        yaml.dump(d, f)


if TYPE_CHECKING:
    from panoptica.api.evaluator import Evaluator  # noqa: F401
