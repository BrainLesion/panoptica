from abc import ABC

import numpy as np

from panoptica._functionals import _get_paired_crop
from panoptica.utils import _count_unique_without_zeros, _unique_without_zeros
from panoptica.utils.constants import _Enum_Compare
from dataclasses import dataclass
from panoptica.metrics import Metric

uint_type: type = np.unsignedinteger
int_type: type = np.integer


class _ProcessingPair(ABC):
    """
    Represents a general processing pair consisting of a reference array and a prediction array. Type of array can be arbitrary (integer recommended)
    Every member is read-only!
    """

    _prediction_arr: np.ndarray
    _reference_arr: np.ndarray
    # unique labels without zero
    _ref_labels: tuple[int, ...]
    _pred_labels: tuple[int, ...]
    n_dim: int

    def __init__(
        self, prediction_arr: np.ndarray, reference_arr: np.ndarray, dtype: type | None
    ) -> None:
        """Initializes a general Processing Pair

        Args:
            prediction_arr (np.ndarray): Numpy array containig the prediction labels
            reference_arr (np.ndarray): Numpy array containig the reference labels
            dtype (type | None): Datatype that is asserted. None for no assertion
        """
        _check_array_integrity(prediction_arr, reference_arr, dtype=dtype)
        self._prediction_arr = prediction_arr
        self._reference_arr = reference_arr
        self.dtype = dtype
        self.n_dim = reference_arr.ndim
        self._ref_labels: tuple[int, ...] = tuple(
            _unique_without_zeros(reference_arr)
        )  # type:ignore
        self._pred_labels: tuple[int, ...] = tuple(
            _unique_without_zeros(prediction_arr)
        )  # type:ignore
        self.crop: tuple[slice, ...] = None
        self.is_cropped: bool = False
        self.uncropped_shape: tuple[int, ...] = reference_arr.shape

    def crop_data(self, verbose: bool = False):
        if self.is_cropped:
            return
        if self.crop is None:
            self.uncropped_shape = self._prediction_arr.shape
            self.crop = _get_paired_crop(
                self._prediction_arr,
                self._reference_arr,
            )

        self._prediction_arr = self._prediction_arr[self.crop]
        self._reference_arr = self._reference_arr[self.crop]
        (
            print(
                f"-- Cropped from {self.uncropped_shape} to {self._prediction_arr.shape}"
            )
            if verbose
            else None
        )
        self.is_cropped = True

    def uncrop_data(self, verbose: bool = False):
        if self.is_cropped == False:
            return
        assert (
            self.uncropped_shape is not None
        ), "Calling uncrop_data() without having cropped first"
        prediction_arr = np.zeros(self.uncropped_shape)
        prediction_arr[self.crop] = self._prediction_arr
        self._prediction_arr = prediction_arr

        reference_arr = np.zeros(self.uncropped_shape)
        reference_arr[self.crop] = self._reference_arr
        (
            print(
                f"-- Uncropped from {self._reference_arr.shape} to {self.uncropped_shape}"
            )
            if verbose
            else None
        )
        self._reference_arr = reference_arr
        self.is_cropped = False

    def set_dtype(self, type):
        assert np.issubdtype(
            type, int_type
        ), "set_dtype: tried to set dtype to something other than integers"
        self._prediction_arr = self._prediction_arr.astype(type)
        self._reference_arr = self._reference_arr.astype(type)

    @property
    def prediction_arr(self):
        return self._prediction_arr

    @property
    def reference_arr(self):
        return self._reference_arr

    @property
    def pred_labels(self):
        return self._pred_labels

    @property
    def ref_labels(self):
        return self._ref_labels

    def copy(self):
        """
        Creates an exact copy of this object
        """
        return type(self)(
            prediction_arr=self._prediction_arr,
            reference_arr=self._reference_arr,
        )  # type:ignore

    # Make all variables read-only!
    # def __setattr__(self, attr, value):
    #    if hasattr(self, attr):
    #        raise Exception("Attempting to alter read-only value")


#
#        self.__dict__[attr] = value


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
            prediction_arr=self._prediction_arr,
            reference_arr=self._reference_arr,
            n_prediction_instance=self.n_prediction_instance,
            n_reference_instance=self.n_reference_instance,
        )  # type:ignore


def _check_array_integrity(
    prediction_arr: np.ndarray, reference_arr: np.ndarray, dtype: type | None = None
):
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
    assert (
        prediction_arr.shape == reference_arr.shape
    ), f"shape mismatch, got {prediction_arr.shape},{reference_arr.shape}"
    assert (
        prediction_arr.dtype == reference_arr.dtype
    ), f"dtype mismatch, got {prediction_arr.dtype},{reference_arr.dtype}"
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
    matched_instances: list[int]

    def __init__(
        self,
        prediction_arr: np.ndarray,
        reference_arr: np.ndarray,
        missed_reference_labels: list[int] | None = None,
        missed_prediction_labels: list[int] | None = None,
        matched_instances: list[int] | None = None,
        n_prediction_instance: int | None = None,
        n_reference_instance: int | None = None,
    ) -> None:
        """Initializes a MatchedInstancePair

        Args:
            prediction_arr (np.ndarray): Numpy array containing the prediction matched instance labels
            reference_arr (np.ndarray): Numpy array containing the reference matched instance labels
            missed_reference_labels (list[int] | None, optional): List of unmatched reference labels. Defaults to None.
            missed_prediction_labels (list[int] | None, optional): List of unmatched prediction labels. Defaults to None.
            matched_instances (int | None, optional): matched instances labels, i.e. unique matched labels in both maps. Defaults to None.
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
        if matched_instances is None:
            matched_instances = [i for i in self._pred_labels if i in self._ref_labels]
        self.matched_instances = matched_instances

        if missed_reference_labels is None:
            missed_reference_labels = list(
                [i for i in self._ref_labels if i not in self._pred_labels]
            )
        self.missed_reference_labels = missed_reference_labels

        if missed_prediction_labels is None:
            missed_prediction_labels = list(
                [i for i in self._pred_labels if i not in self._ref_labels]
            )
        self.missed_prediction_labels = missed_prediction_labels

    @property
    def n_matched_instances(self):
        return len(self.matched_instances)

    def copy(self):
        """
        Creates an exact copy of this object
        """
        return type(self)(
            prediction_arr=self._prediction_arr.copy(),
            reference_arr=self._reference_arr.copy(),
            n_prediction_instance=self.n_prediction_instance,
            n_reference_instance=self.n_reference_instance,
            missed_reference_labels=self.missed_reference_labels,
            missed_prediction_labels=self.missed_prediction_labels,
            matched_instances=self.matched_instances,
        )


@dataclass
class EvaluateInstancePair:
    reference_arr: np.ndarray
    prediction_arr: np.ndarray
    num_pred_instances: int
    num_ref_instances: int
    tp: int
    list_metrics: dict[Metric, list[float]]


class InputType(_Enum_Compare):
    SEMANTIC = SemanticPair
    UNMATCHED_INSTANCE = UnmatchedInstancePair
    MATCHED_INSTANCE = MatchedInstancePair

    def __call__(
        self, prediction_arr: np.ndarray, reference_arr: np.ndarray
    ) -> _ProcessingPair:
        return self.value(prediction_arr, reference_arr)


class IntermediateStepsData:
    def __init__(self, original_input: _ProcessingPair | None):
        self._original_input = original_input
        self._intermediatesteps: dict[str, _ProcessingPair] = {}

    def add_intermediate_arr_data(
        self, processing_pair: _ProcessingPair, inputtype: InputType
    ):
        type_name = inputtype.name
        self.add_intermediate_data(type_name, processing_pair)

    def add_intermediate_data(self, key, value):
        assert key not in self._intermediatesteps, f"key {key} already added"
        self._intermediatesteps[key] = value

    @property
    def original_prediction_arr(self):
        assert (
            self._original_input is not None
        ), "Original prediction_arr is None, there are no intermediate steps"
        return self._original_input.prediction_arr

    @property
    def original_reference_arr(self):
        assert (
            self._original_input is not None
        ), "Original reference_arr is None, there are no intermediate steps"
        return self._original_input.reference_arr

    def prediction_arr(self, inputtype: InputType):
        type_name = inputtype.name
        procpair = self[type_name]
        assert isinstance(
            procpair, _ProcessingPair
        ), f"step {type_name} is not a processing pair, error"
        return procpair.prediction_arr

    def reference_arr(self, inputtype: InputType):
        type_name = inputtype.name
        procpair = self[type_name]
        assert isinstance(
            procpair, _ProcessingPair
        ), f"step {type_name} is not a processing pair, error"
        return procpair.reference_arr

    def __getitem__(self, key):
        assert (
            key in self._intermediatesteps
        ), f"key {key} not in intermediate steps, maybe the step was skipped?"
        return self._intermediatesteps[key]
