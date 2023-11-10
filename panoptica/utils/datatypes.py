from typing import Any, Self
import numpy as np
from numpy import dtype
from abc import ABC
import warnings
from panoptica.utils import _count_unique_without_zeros, _unique_without_zeros

uint_type: type = np.unsignedinteger
int_type: type = np.integer


class _ProcessingPair(ABC):
    prediction_arr: np.ndarray
    reference_arr: np.ndarray
    # unique labels without zero
    ref_labels: tuple[int]
    pred_labels: tuple[int]

    def __init__(self, prediction_arr: np.ndarray, reference_arr: np.ndarray, dtype: type | None) -> None:
        _check_array_integrity(prediction_arr, reference_arr, dtype=dtype)
        self.prediction_arr = prediction_arr
        self.reference_arr = reference_arr
        self.ref_labels: tuple[int] = tuple(_unique_without_zeros(reference_arr))  # type:ignore
        self.pred_labels: tuple[int] = tuple(_unique_without_zeros(prediction_arr))  # type:ignore

    # Make all variables read-only!
    def __setattr__(self, attr, value):
        if hasattr(self, attr):
            raise Exception("Attempting to alter read-only value")

        self.__dict__[attr] = value


class _ProcessingPairInstanced(_ProcessingPair):
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
        return type(self)(
            prediction_arr=self.prediction_arr,
            reference_arr=self.reference_arr,
            n_prediction_instance=self.n_prediction_instance,
            n_reference_instance=self.n_reference_instance,
        )


def _check_array_integrity(prediction_arr: np.ndarray, reference_arr: np.ndarray, dtype: type | None = None):
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
    """A Processing pair of any dtype

    Args:
        ProcessingPair (_type_): _description_
    """

    def __init__(self, prediction_arr: np.ndarray, reference_arr: np.ndarray) -> None:
        super().__init__(prediction_arr, reference_arr, dtype=int_type)


class UnmatchedInstancePair(_ProcessingPairInstanced):
    """A Processing pair of any unsigned (but matching) integer type

    Args:
        ProcessingPairInstanced (_type_): _description_
    """

    def __init__(
        self,
        prediction_arr: np.ndarray,
        reference_arr: np.ndarray,
        n_prediction_instance: int | None = None,
        n_reference_instance: int | None = None,
    ) -> None:
        super().__init__(prediction_arr, reference_arr, uint_type, n_prediction_instance, n_reference_instance)  # type:ignore


class MatchedInstancePair(_ProcessingPairInstanced):
    """A Processing pair of any unsigned (but matching) integer type consisting of only matched instance labels, as well as a list of missed labels from both

    Args:
        ProcessingPairInstanced (_type_): _description_
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
        super().__init__(prediction_arr, reference_arr, uint_type, n_prediction_instance, n_reference_instance)  # type:ignore
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
        return type(self)(
            prediction_arr=self.prediction_arr,
            reference_arr=self.reference_arr,
            n_prediction_instance=self.n_prediction_instance,
            n_reference_instance=self.n_reference_instance,
            missed_reference_labels=self.missed_reference_labels,
            missed_prediction_labels=self.missed_prediction_labels,
            n_matched_instances=self.n_matched_instances,
        )


# Mapping ((prediction_label, ...), (reference_label, ...))
Instance_Label_Map = list[tuple[list[uint_type], list[uint_type]]]


if __name__ == "__main__":
    n = np.zeros([50, 50], dtype=np.int32)
    a = SemanticPair(n, n)
    print(a)
    # print(a.prediction_arr)
