import numpy as np


# Many-to-One Mapping
class InstanceLabelMap(object):
    """Creates a mapping between prediction labels and reference labels in a many-to-one relationship.

    This class allows mapping multiple prediction labels to a single reference label.
    It includes methods for adding new mappings, checking containment, retrieving
    predictions mapped to a reference, and exporting the mapping as a dictionary.

    Attributes:
        labelmap (dict[int, int]): Dictionary storing the prediction-to-reference label mappings.

    Methods:
        add_labelmap_entry(pred_labels, ref_label): Adds a new entry mapping prediction labels to a reference label.
        get_pred_labels_matched_to_ref(ref_label): Retrieves prediction labels mapped to a given reference label.
        contains_pred(pred_label): Checks if a prediction label exists in the map.
        contains_ref(ref_label): Checks if a reference label exists in the map.
        contains_and(pred_label, ref_label): Checks if both a prediction and a reference label are in the map.
        contains_or(pred_label, ref_label): Checks if either a prediction or reference label is in the map.
        get_one_to_one_dictionary(): Returns the labelmap dictionary for a one-to-one view.
    """

    labelmap: dict[int, int]

    def __init__(self) -> None:
        self.labelmap = {}

    def add_labelmap_entry(self, pred_labels: list[int] | int, ref_label: int):
        """Adds an entry that maps prediction labels to a single reference label.

        Args:
            pred_labels (list[int] | int): List of prediction labels or a single prediction label.
            ref_label (int): Reference label to map to.

        Raises:
            AssertionError: If `ref_label` is not an integer.
            AssertionError: If any `pred_labels` are not integers.
            Exception: If a prediction label is already mapped to a different reference label.
        """
        if not isinstance(pred_labels, list):
            pred_labels = [pred_labels]
        assert isinstance(ref_label, int), "add_labelmap_entry: got no int as ref_label"
        assert np.all(
            [isinstance(r, int) for r in pred_labels]
        ), "add_labelmap_entry: got no int as pred_label"
        for p in pred_labels:
            if p in self.labelmap and self.labelmap[p] != ref_label:
                raise Exception(
                    f"You are mapping a prediction label to a reference label that was already assigned differently, got {self.__str__} and you tried {pred_labels}, {ref_label}"
                )
            self.labelmap[p] = ref_label

    def get_pred_labels_matched_to_ref(self, ref_label: int):
        """Retrieves all prediction labels that map to a specified reference label.

        Args:
            ref_label (int): The reference label to search.

        Returns:
            list[int]: List of prediction labels mapped to `ref_label`.
        """
        return [k for k, v in self.labelmap.items() if v == ref_label]

    def contains_pred(self, pred_label: int):
        """Checks if a prediction label exists in the map.

        Args:
            pred_label (int): The prediction label to search.

        Returns:
            bool: True if `pred_label` is in `labelmap`, otherwise False.
        """
        return pred_label in self.labelmap

    def contains_ref(self, ref_label: int):
        """Checks if a reference label exists in the map.

        Args:
            ref_label (int): The reference label to search.

        Returns:
            bool: True if `ref_label` is in `labelmap` values, otherwise False.
        """
        return ref_label in self.labelmap.values()

    def contains_and(
        self, pred_label: int | None = None, ref_label: int | None = None
    ) -> bool:
        """Checks if both a prediction and a reference label are in the map.

        Args:
            pred_label (int | None): The prediction label to check.
            ref_label (int | None): The reference label to check.

        Returns:
            bool: True if both `pred_label` and `ref_label` are in the map; otherwise, False.
        """
        pred_in = True if pred_label is None else pred_label in self.labelmap
        ref_in = True if ref_label is None else ref_label in self.labelmap.values()
        return pred_in and ref_in

    def contains_or(
        self, pred_label: int | None = None, ref_label: int | None = None
    ) -> bool:
        """Checks if either a prediction or reference label is in the map.

        Args:
            pred_label (int | None): The prediction label to check.
            ref_label (int | None): The reference label to check.

        Returns:
            bool: True if either `pred_label` or `ref_label` are in the map; otherwise, False.
        """
        pred_in = True if pred_label is None else pred_label in self.labelmap
        ref_in = True if ref_label is None else ref_label in self.labelmap.values()
        return pred_in or ref_in

    def get_one_to_one_dictionary(self):
        """Returns a copy of the labelmap dictionary for a one-to-one view.

        Returns:
            dict[int, int]: The prediction-to-reference label mapping.
        """
        return self.labelmap

    def __str__(self) -> str:
        return str(
            list(
                [
                    str(tuple(k for k in self.labelmap.keys() if self.labelmap[k] == v))
                    + " -> "
                    + str(v)
                    for v in set(self.labelmap.values())
                ]
            )
        )

    def __repr__(self) -> str:
        return str(self)

    # Make all variables read-only!
    def __setattr__(self, attr, value):
        """Overrides attribute setting to make attributes read-only after initialization.

        Args:
            attr (str): Attribute name.
            value (Any): Attribute value.

        Raises:
            Exception: If trying to alter an existing attribute.
        """
        if hasattr(self, attr):
            raise Exception("Attempting to alter read-only value")

        self.__dict__[attr] = value
