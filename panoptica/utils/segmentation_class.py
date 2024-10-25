import numpy as np
from pathlib import Path
from panoptica.utils.config import SupportsConfig
from panoptica.utils.label_group import LabelGroup, _LabelGroupAny

NO_GROUP_KEY = "ungrouped"


class SegmentationClassGroups(SupportsConfig):
    """Represents a collection of segmentation class groups.

    This class manages groups of labels used in segmentation tasks, ensuring that each label is defined
    exactly once across all groups. It supports both list and dictionary formats for group initialization.

    Attributes:
        __group_dictionary (dict[str, LabelGroup]): A dictionary mapping group names to their respective LabelGroup instances.
        __labels (list[int]): A flat list of unique labels collected from all LabelGroups.

    Args:
        groups (list[LabelGroup] | dict[str, LabelGroup | tuple[list[int] | int, bool]]):
            A list of `LabelGroup` instances or a dictionary where keys are group names (str) and values are either
            `LabelGroup` instances or tuples containing a list of label values and a boolean.

    Raises:
        AssertionError: If the same label is assigned to multiple groups.
    """

    def __init__(
        self,
        groups: list[LabelGroup] | dict[str, LabelGroup | tuple[list[int] | int, bool]],
    ) -> None:
        self.__group_dictionary: dict[str, LabelGroup] = {}
        self.__labels: list[int] = []
        # maps name of group to the group itself

        if isinstance(groups, list):
            self.__group_dictionary = {
                f"group_{idx}": g for idx, g in enumerate(groups)
            }
        elif isinstance(groups, dict):
            # transform dict into list of LabelGroups
            for i, g in groups.items():
                name_lower = str(i).lower()
                if isinstance(g, LabelGroup):
                    self.__group_dictionary[name_lower] = g
                else:
                    self.__group_dictionary[name_lower] = LabelGroup(g[0], g[1])

        # needs to check that each label is accounted for exactly ONCE
        labels = [
            value_label
            for lg in self.__group_dictionary.values()
            for value_label in lg.value_labels
        ]
        duplicates = list_duplicates(labels)
        if len(duplicates) > 0:
            print(
                f"The same labels {duplicates} were assigned to two different labelgroups, got {str(self)}\nIntended? This will evaluate the duplicate labels in both groups"
            )
        self.__labels = labels

    def has_defined_labels_for(
        self, arr: np.ndarray | list[int], raise_error: bool = False
    ):
        """Checks if the labels in the provided array are defined in the segmentation class groups.

        Args:
            arr (np.ndarray | list[int]): The array of labels to check.
            raise_error (bool): If True, raises an error when an undefined label is found. Defaults to False.

        Returns:
            bool: True if all labels are defined; False otherwise.

        Raises:
            AssertionError: If an undefined label is found and raise_error is True.
        """
        if isinstance(arr, list):
            arr_labels = arr
        else:
            arr_labels = [i for i in np.unique(arr) if i != 0]
        for al in arr_labels:
            if al not in self.labels:
                if raise_error:
                    raise AssertionError(
                        f"Input array has labels undefined in the SegmentationClassGroups, got label {al} the groups are defined as {str(self)}"
                    )
                return False
        return True

    def __str__(self) -> str:
        text = "SegmentationClassGroups = "
        for i, lg in self.__group_dictionary.items():
            text += f"\n - {i} : {str(lg)}"
        return text

    def __contains__(self, item):
        return item in self.__group_dictionary

    def __getitem__(self, key):
        return self.__group_dictionary[key]

    def __iter__(self):
        yield from self.__group_dictionary

    def keys(self) -> list[str]:
        return list(self.__group_dictionary.keys())

    @property
    def labels(self):
        return self.__labels

    def items(self):
        for k in self:
            yield k, self[k]

    @classmethod
    def _yaml_repr(cls, node):
        return {"groups": node.__group_dictionary}


def list_duplicates(seq):
    """Identifies duplicates in a sequence.

    Args:
        seq (list): The input sequence to check for duplicates.

    Returns:
        list: A list of duplicates found in the input sequence.
    """
    seen = set()
    seen_add = seen.add
    # adds all elements it doesn't know yet to seen and all other to seen_twice
    seen_twice = set(x for x in seq if x in seen or seen_add(x))
    # turn the set into a list (as requested)
    return list(seen_twice)


class _NoSegmentationClassGroups(SegmentationClassGroups):
    """Represents a placeholder for segmentation class groups with no defined labels.

    This class indicates that no specific segmentation groups or labels are defined, and any label is valid.

    Attributes:
        __group_dictionary (dict[str, LabelGroup]): A dictionary with a single entry representing all labels as a group.
    """

    def __init__(self) -> None:
        self.__group_dictionary = {NO_GROUP_KEY: _LabelGroupAny()}

    def has_defined_labels_for(
        self, arr: np.ndarray | list[int], raise_error: bool = False
    ):
        return True

    def __str__(self) -> str:
        text = "NoSegmentationClassGroups"
        return text

    def __contains__(self, item):
        return item in self.__group_dictionary

    def __getitem__(self, key):
        return self.__group_dictionary[key]

    def __iter__(self):
        yield from self.__group_dictionary

    def keys(self) -> list[str]:
        return list(self.__group_dictionary.keys())

    @property
    def labels(self):
        raise Exception(
            "_NoSegmentationClassGroups has no explicit definition of labels"
        )

    @classmethod
    def _yaml_repr(cls, node):
        return {}
