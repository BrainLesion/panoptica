import numpy as np


# TODO also support LabelMergedGroup which takes multi labels and convert them into one before the evaluation
# Useful for BraTs with hierarchical labels (then define one generic Group class and then two more specific subgroups, one for hierarchical, the other for the current one)


class LabelGroup:
    """Defines a group of labels that semantically belong to each other. Only labels within a group will be matched with each other"""

    def __init__(
        self,
        value_labels: list[int] | int,
        single_instance: bool = False,
    ) -> None:
        """Defines a group of labels that semantically belong to each other

        Args:
            value_labels (list[int]): Actually labels in the prediction and reference mask in this group. Defines the labels that can be matched to each other
            single_instance (bool, optional): If true, will not use the matching_threshold as there is only one instance (large organ, ...). Defaults to False.
        """
        if isinstance(value_labels, int):
            value_labels = [value_labels]
        self.__value_labels = value_labels
        assert np.all([v > 0 for v in self.__value_labels]), f"Given value labels are not >0, got {value_labels}"
        self.__single_instance = single_instance
        if self.__single_instance:
            assert len(value_labels) == 1, f"single_instance set to True, but got more than one label for this group, got {value_labels}"

    @property
    def value_labels(self) -> list[int]:
        return self.__value_labels

    @property
    def single_instance(self) -> bool:
        return self.__single_instance

    def __call__(
        self,
        array: np.ndarray,
        set_to_binary: bool = False,
    ) -> np.ndarray:
        """Extracts the labels of this class

        Args:
            array (np.ndarray): Array to extract the segmentation group labels from
            set_to_binary (bool, optional): If true, will output a binary array. Defaults to False.

        Returns:
            np.ndarray: Array containing only the labels of this segmentation group
        """
        array = array.copy()
        array[np.isin(array, self.value_labels, invert=True)] = 0
        if set_to_binary:
            array[array != 0] = 1
        return array

    def __str__(self) -> str:
        return f"LabelGroup {self.value_labels}, single_instance={self.single_instance}"

    def __repr__(self) -> str:
        return str(self)


class SegmentationClassGroups:
    def __init__(
        self,
        groups: list[LabelGroup] | dict[str, LabelGroup | tuple[list[int] | int, bool]],
    ) -> None:
        self.__group_dictionary: dict[str, LabelGroup] = {}
        self.__labels: list[int] = []
        # maps name of group to the group itself

        if isinstance(groups, list):
            self.__group_dictionary = {f"group_{idx}": g for idx, g in enumerate(groups)}
        else:
            # transform dict into list of LabelGroups
            for i, g in groups.items():
                name_lower = str(i).lower()
                if isinstance(g, LabelGroup):
                    self.__group_dictionary[name_lower] = LabelGroup(g.value_labels, g.single_instance)
                else:
                    self.__group_dictionary[name_lower] = LabelGroup(g[0], g[1])

        # needs to check that each label is accounted for exactly ONCE
        labels = [value_label for lg in self.__group_dictionary.values() for value_label in lg.value_labels]
        duplicates = list_duplicates(labels)
        if len(duplicates) > 0:
            raise AssertionError(f"The same label was assigned to two different labelgroups, got {str(self)}")

        self.__labels = labels

    def has_defined_labels_for(self, arr: np.ndarray | list[int], raise_error: bool = False):
        if isinstance(arr, list):
            arr_labels = arr
        else:
            arr_labels = [i for i in np.unique(arr) if i != 0]
        for al in arr_labels:
            if al not in self.__labels:
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


def list_duplicates(seq):
    seen = set()
    seen_add = seen.add
    # adds all elements it doesn't know yet to seen and all other to seen_twice
    seen_twice = set(x for x in seq if x in seen or seen_add(x))
    # turn the set into a list (as requested)
    return list(seen_twice)


if __name__ == "__main__":
    group1 = LabelGroup([1, 2, 3, 4, 5], single_instance=False)

    print(group1)
    print(group1.value_labels)

    arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    group1_arr = group1(arr, True)
    print(group1_arr)

    classgroups = SegmentationClassGroups(
        groups={
            "vertebra": group1,
            "ivds": LabelGroup([100, 101, 102]),
        }
    )
    print(classgroups)

    print(classgroups.has_defined_labels_for([1, 2, 3]))

    for i in classgroups:
        print(i)
