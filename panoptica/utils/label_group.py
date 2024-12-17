import numpy as np
from panoptica.utils.config import SupportsConfig

#


class LabelGroup(SupportsConfig):
    """Defines a group of labels that semantically belong together for segmentation purposes.

    Groups of labels define label sets that can be matched with each other.
    For example, labels might represent different parts of a segmented object, and only those within the group are eligible for matching.

    Attributes:
        value_labels (list[int]): List of integer labels representing segmentation group labels.
        single_instance (bool): If True, the group represents a single instance without matching threshold consideration.
    """

    def __init__(
        self,
        value_labels: list[int] | int,
        single_instance: bool = False,
    ) -> None:
        """Initializes a LabelGroup with specified labels and single instance setting.

        Args:
            value_labels (list[int] | int): Labels in the prediction and reference mask for this group.
            single_instance (bool, optional): If True, ignores matching threshold as only one instance exists. Defaults to False.

        Raises:
            AssertionError: If `value_labels` is empty or if labels are not positive integers.
            AssertionError: If `single_instance` is True but more than one label is provided.
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
        """List of integer labels for this segmentation group."""
        return self.__value_labels

    @property
    def single_instance(self) -> bool:
        """Indicates if this group is treated as a single instance."""
        return self.__single_instance

    def extract_label(
        self,
        array: np.ndarray,
        set_to_binary: bool = False,
    ):
        """Extracts an array of the labels specific to this segmentation group.

        Args:
            array (np.ndarray): The array to filter for segmentation group labels.
            set_to_binary (bool, optional): If True, outputs a binary array. Defaults to False.

        Returns:
            np.ndarray: An array with only the labels of this segmentation group.
        """
        array = array.copy()
        array[np.isin(array, self.value_labels, invert=True)] = 0
        if set_to_binary:
            array[array != 0] = 1
        return array

    def __call__(
        self,
        array: np.ndarray,
    ) -> np.ndarray:
        """Extracts labels from an array for this group when the instance is called.

        Args:
            array (np.ndarray): Array to filter for segmentation group labels.

        Returns:
            np.ndarray: Array containing only the labels for this segmentation group.
        """
        return self.extract_label(array, set_to_binary=False)

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


class LabelMergeGroup(LabelGroup):
    """Defines a group of labels that will be merged into a single label when extracted.

    Inherits from LabelGroup and sets extracted labels to a binary format.
    This is useful for region-evaluation (e.g. BRATS), where you want to merge multiple labels into one before evaluation.

    Methods:
        __call__(array): Extracts the label group as a binary array.
    """

    def __init__(
        self, value_labels: list[int] | int, single_instance: bool = False
    ) -> None:
        super().__init__(value_labels, single_instance)

    def __call__(
        self,
        array: np.ndarray,
    ) -> np.ndarray:
        """Extracts the labels of this group as a binary array.

        Args:
            array (np.ndarray): Array to filter for segmentation group labels.

        Returns:
            np.ndarray: Binary array representing presence or absence of group labels.
        """
        return self.extract_label(array, set_to_binary=True)

    def __str__(self) -> str:
        return f"LabelMergeGroup {self.value_labels} -> ONE, single_instance={self.single_instance}"


class _LabelGroupAny(LabelGroup):
    """Represents a group that includes all labels in the array with no specific segmentation constraints.

    Used to represent a group that does not restrict labels.

    Methods:
        __call__(array, set_to_binary): Returns the unfiltered array.
    """

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
        """Returns the original array, unfiltered.

        Args:
            array (np.ndarray): The original array to return.
            set_to_binary (bool, optional): Ignored in this implementation.

        Returns:
            np.ndarray: The original, unmodified array.
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
