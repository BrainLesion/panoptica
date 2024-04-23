import numpy as np


class ClassGroup:
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
        self._value_labels = value_labels
        assert np.all([v > 0 for v in self._value_labels]), f"Given value labels are not >0, got {value_labels}"
        self._single_instance = single_instance
        if self._single_instance:
            assert len(value_labels) == 1, f"single_instance set to True, but got more than one label for this group, got {value_labels}"

    @property
    def value_labels(self):
        return self._value_labels

    @property
    def single_instance(self):
        return self._single_instance
