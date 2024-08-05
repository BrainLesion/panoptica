import numpy as np


# Many-to-One Mapping
class InstanceLabelMap(object):
    # Mapping ((prediction_label, ...), (reference_label, ...))
    labelmap: dict[int, int]

    def __init__(self) -> None:
        self.labelmap = {}

    def add_labelmap_entry(self, pred_labels: list[int] | int, ref_label: int):
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
        return [k for k, v in self.labelmap.items() if v == ref_label]

    def contains_pred(self, pred_label: int):
        return pred_label in self.labelmap

    def contains_ref(self, ref_label: int):
        return ref_label in self.labelmap.values()

    def contains_and(
        self, pred_label: int | None = None, ref_label: int | None = None
    ) -> bool:
        pred_in = True if pred_label is None else pred_label in self.labelmap
        ref_in = True if ref_label is None else ref_label in self.labelmap.values()
        return pred_in and ref_in

    def contains_or(
        self, pred_label: int | None = None, ref_label: int | None = None
    ) -> bool:
        pred_in = True if pred_label is None else pred_label in self.labelmap
        ref_in = True if ref_label is None else ref_label in self.labelmap.values()
        return pred_in or ref_in

    def get_one_to_one_dictionary(self):
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
        if hasattr(self, attr):
            raise Exception("Attempting to alter read-only value")

        self.__dict__[attr] = value
