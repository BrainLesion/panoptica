"""Label grouping and instance-id bookkeeping."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LabelGroup:
    """A set of semantic label values evaluated together as one group."""

    value_labels: tuple[int, ...]
    single_instance: bool = False


@dataclass(frozen=True)
class LabelPartGroup(LabelGroup):
    """A group whose labels are parts of a larger instance."""

    part_labels: tuple[int, ...] = ()


@dataclass(frozen=True)
class SegmentationClassGroups:
    """Named mapping of group-name -> LabelGroup."""

    groups: dict[str, LabelGroup] = field(default_factory=dict)


class InstanceLabelMap:
    """Bidirectional prediction<->reference label mapping built during matching."""

    def __init__(self) -> None:
        self.labelmap: dict[int, int] = {}

    def add_labelmap_entry(self, pred_labels: int | list[int], ref_label: int) -> None:
        raise NotImplementedError("filled in by matching stream against v1 semantics")
