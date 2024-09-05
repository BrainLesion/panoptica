import numpy as np
from panoptica.utils.config import SupportsConfig

#


class LabelGroup(SupportsConfig):
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

        value_labels = list(set(value_labels))

        assert (
            len(value_labels) >= 1
        ), f"You tried to define a LabelGroup without any specified labels, got {value_labels}"
        self.__value_labels = value_labels
        assert np.all(
            [v > 0 for v in self.__value_labels]
        ), f"Given value labels are not >0, got {value_labels}"
        self.__single_instance = single_instance
        if self.__single_instance:
            assert (
                len(value_labels) == 1
            ), f"single_instance set to True, but got more than one label for this group, got {value_labels}"

        LabelGroup._register_permanently()

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

    @classmethod
    def _yaml_repr(cls, node):
        return {
            "value_labels": node.value_labels,
            "single_instance": node.single_instance,
        }


class _LabelGroupAny(LabelGroup):
    def __init__(self) -> None:
        pass

    @property
    def value_labels(self) -> list[int]:
        raise AssertionError("LabelGroupAny has no value_labels, it is all labels")

    @property
    def single_instance(self) -> bool:
        return False

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
        return array

    def __str__(self) -> str:
        return f"LabelGroupAny"

    def __repr__(self) -> str:
        return str(self)

    @classmethod
    def _yaml_repr(cls, node):
        return {}
