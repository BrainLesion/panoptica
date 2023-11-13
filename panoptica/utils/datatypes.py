from abc import ABC

import numpy as np
from numpy import dtype

from panoptica.utils import _count_unique_without_zeros, _unique_without_zeros

uint_type: type = np.unsignedinteger
int_type: type = np.integer


class _ProcessingPair(ABC):
    """
    Represents a general processing pair consisting of a reference array and a prediction array. Type of array can be arbitrary (integer recommended)
    Every member is read-only!
    """

    prediction_arr: np.ndarray
    reference_arr: np.ndarray
    # unique labels without zero
    ref_labels: tuple[int]
    pred_labels: tuple[int]

    def __init__(self, prediction_arr: np.ndarray, reference_arr: np.ndarray, dtype: type | None) -> None:
        """Initializes a general Processing Pair

        Args:
            prediction_arr (np.ndarray): Numpy array containig the prediction labels
            reference_arr (np.ndarray): Numpy array containig the reference labels
            dtype (type | None): Datatype that is asserted. None for no assertion
        """
        _check_array_integrity(prediction_arr, reference_arr, dtype=dtype)
        self.prediction_arr = prediction_arr
        self.reference_arr = reference_arr
        self.dtype = dtype
        self.ref_labels: tuple[int] = tuple(_unique_without_zeros(reference_arr))  # type:ignore
        self.pred_labels: tuple[int] = tuple(_unique_without_zeros(prediction_arr))  # type:ignore

    # Make all variables read-only!
    def __setattr__(self, attr, value):
        if hasattr(self, attr):
            raise Exception("Attempting to alter read-only value")

        self.__dict__[attr] = value


class _ProcessingPairInstanced(_ProcessingPair):
    """
    A ProcessingPair that contains instances, additionally has number of instances available
    """

    n_prediction_instance: int
    n_reference_instance: int

    def __init__(
        self,
        prediction_arr: np.ndarray,
        reference_arr: np.ndarray,
        dtype: type | None,
        n_prediction_instance: int | None = None,
        n_reference_instance: int | None = None,
    ) -> None:
        # reduce to lowest uint
        super().__init__(prediction_arr, reference_arr, dtype)
        if n_prediction_instance is None:
            self.n_prediction_instance = _count_unique_without_zeros(prediction_arr)

        else:
            self.n_prediction_instance = n_prediction_instance
        if n_reference_instance is None:
            self.n_reference_instance = _count_unique_without_zeros(reference_arr)
        else:
            self.n_reference_instance = n_reference_instance

    def copy(self):
        """
        Creates an exact copy of this object
        """
        return type(self)(
            prediction_arr=self.prediction_arr,
            reference_arr=self.reference_arr,
            n_prediction_instance=self.n_prediction_instance,
            n_reference_instance=self.n_reference_instance,
        )  # type:ignore


def _check_array_integrity(prediction_arr: np.ndarray, reference_arr: np.ndarray, dtype: type | None = None):
    """
    Check the integrity of two numpy arrays.

    Parameters:
    - prediction_arr (np.ndarray): The array to be checked.
    - reference_arr (np.ndarray): The reference array for comparison.
    - dtype (type | None): The expected data type for both arrays. Defaults to None.

    Raises:
    - AssertionError: If prediction_arr or reference_arr are not numpy arrays.
    - AssertionError: If the shapes of prediction_arr and reference_arr do not match.
    - AssertionError: If the data types of prediction_arr and reference_arr do not match.
    - AssertionError: If dtype is provided and the data types of prediction_arr and/or reference_arr
                     do not match the specified dtype.

    Example:
    >>> _check_array_integrity(np.array([1, 2, 3]), np.array([4, 5, 6]), dtype=int)
    """
    assert isinstance(prediction_arr, np.ndarray) and isinstance(
        reference_arr, np.ndarray
    ), "prediction and/or reference are not numpy arrays"
    assert prediction_arr.shape == reference_arr.shape, f"shape mismatch, got {prediction_arr.shape},{reference_arr.shape}"
    assert prediction_arr.dtype == reference_arr.dtype, f"dtype mismatch, got {prediction_arr.dtype},{reference_arr.dtype}"
    if dtype is not None:
        assert (
            np.issubdtype(prediction_arr.dtype, dtype)
            and np.issubdtype(reference_arr.dtype, dtype)
            # prediction_arr.dtype == dtype and reference_arr.dtype == dtype
        ), f"prediction and/or reference are not dtype {dtype}, got {prediction_arr.dtype} and {reference_arr.dtype}"


class SemanticPair(_ProcessingPair):
    """A Processing pair that contains Semantic Labels"""

    def __init__(self, prediction_arr: np.ndarray, reference_arr: np.ndarray) -> None:
        super().__init__(prediction_arr, reference_arr, dtype=int_type)


class UnmatchedInstancePair(_ProcessingPairInstanced):
    """
    A Processing pair that contain Unmatched Instance Maps
    Can be of any unsigned (but matching) integer type
    """

    def __init__(
        self,
        prediction_arr: np.ndarray,
        reference_arr: np.ndarray,
        n_prediction_instance: int | None = None,
        n_reference_instance: int | None = None,
    ) -> None:
        super().__init__(
            prediction_arr,
            reference_arr,
            uint_type,
            n_prediction_instance,
            n_reference_instance,
        )  # type:ignore


class MatchedInstancePair(_ProcessingPairInstanced):
    """
    A Processing pair that contain Matched Instance Maps, i.e. each equal label in both maps are a match
    Can be of any unsigned (but matching) integer type
    """

    missed_reference_labels: list[int]
    missed_prediction_labels: list[int]
    n_matched_instances: int

    def __init__(
        self,
        prediction_arr: np.ndarray,
        reference_arr: np.ndarray,
        missed_reference_labels: list[int] | None = None,
        missed_prediction_labels: list[int] | None = None,
        n_matched_instances: int | None = None,
        n_prediction_instance: int | None = None,
        n_reference_instance: int | None = None,
    ) -> None:
        """Initializes a MatchedInstancePair

        Args:
            prediction_arr (np.ndarray): Numpy array containing the prediction matched instance labels
            reference_arr (np.ndarray): Numpy array containing the reference matched instance labels
            missed_reference_labels (list[int] | None, optional): List of unmatched reference labels. Defaults to None.
            missed_prediction_labels (list[int] | None, optional): List of unmatched prediction labels. Defaults to None.
            n_matched_instances (int | None, optional): Number of total matched instances, i.e. unique matched labels in both maps. Defaults to None.
            n_prediction_instance (int | None, optional): Number of prediction instances. Defaults to None.
            n_reference_instance (int | None, optional): Number of reference instances. Defaults to None.

            For each argument: If none, will calculate on initialization.
        """
        super().__init__(
            prediction_arr,
            reference_arr,
            uint_type,
            n_prediction_instance,
            n_reference_instance,
        )  # type:ignore
        if n_matched_instances is None:
            n_matched_instances = len([i for i in self.pred_labels if i in self.ref_labels])
        self.n_matched_instances = n_matched_instances

        if missed_reference_labels is None:
            missed_reference_labels = list([i for i in self.ref_labels if i not in self.pred_labels])
        self.missed_reference_labels = missed_reference_labels

        if missed_prediction_labels is None:
            missed_prediction_labels = list([i for i in self.pred_labels if i not in self.ref_labels])
        self.missed_prediction_labels = missed_prediction_labels

    def copy(self):
        """
        Creates an exact copy of this object
        """
        return type(self)(
            prediction_arr=self.prediction_arr,
            reference_arr=self.reference_arr,
            n_prediction_instance=self.n_prediction_instance,
            n_reference_instance=self.n_reference_instance,
            missed_reference_labels=self.missed_reference_labels,
            missed_prediction_labels=self.missed_prediction_labels,
            n_matched_instances=self.n_matched_instances,
        )


# Many-to-One Mapping
class InstanceLabelMap:
    # Mapping ((prediction_label, ...), (reference_label, ...))
    labelmap: dict[int, int]

    def __init__(self) -> None:
        self.labelmap = {}

    def add_labelmap_entry(self, pred_labels: list[int] | int, ref_label: int):
        if not isinstance(pred_labels, list):
            pred_labels = [pred_labels]
        for p in pred_labels:
            if p in self.labelmap and self.labelmap[p] != ref_label:
                raise Exception(
                    f"You are mapping a prediction label to a reference label that was already assigned differently, got {self.__str__} and you tried {pred_labels}, {ref_label}"
                )
            self.labelmap[p] = ref_label

    def get_one_to_one_dictionary(self):
        return self.labelmap

    def __str__(self) -> str:
        return str(
            list(
                [
                    str(tuple(k for k in self.labelmap.keys() if self.labelmap[k] == v)) + " -> " + str(v)
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


if __name__ == "__main__":
    n = np.zeros([50, 50], dtype=np.int32)
    a = SemanticPair(n, n)
    print(a)
    # print(a.prediction_arr)

    map = InstanceLabelMap()
    map.labelmap = {2: 3, 3: 3, 4: 6}
    print(map)
