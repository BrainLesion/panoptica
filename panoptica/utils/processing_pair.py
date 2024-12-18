from abc import ABC

import numpy as np

from panoptica._functionals import _get_paired_crop
from panoptica.utils import _count_unique_without_zeros, _unique_without_zeros
from panoptica.utils.constants import _Enum_Compare
from dataclasses import dataclass
from panoptica.metrics import Metric
from panoptica.utils.numpy_utils import _get_smallest_fitting_uint

uint_type: type = np.unsignedinteger
int_type: type = np.integer


class _ProcessingPair(ABC):
    """Represents a pair of processing arrays, typically prediction and reference arrays.

    This base class provides core functionality for processing and comparing prediction
    and reference data arrays. Each instance contains two arrays and supports cropping and
    data integrity checks.

    Attributes:
        n_dim (int): The number of dimensions in the reference array.
        crop (tuple[slice, ...] | None): The crop region applied to both arrays, if any.
        is_cropped (bool): Indicates whether the arrays have been cropped.
        uncropped_shape (tuple[int, ...]): The original shape of the arrays before cropping.
    """

    def __init__(self, prediction_arr: np.ndarray, reference_arr: np.ndarray) -> None:
        """Initializes the processing pair with prediction and reference arrays.

        Args:
            prediction_arr (np.ndarray): Numpy array of prediction labels.
            reference_arr (np.ndarray): Numpy array of reference labels.
            dtype (type | None): The expected datatype of arrays. If None, no datatype check is performed.
        """
        self.__prediction_arr: np.ndarray = prediction_arr
        self.__reference_arr: np.ndarray = reference_arr
        _check_array_integrity(
            self.__prediction_arr, self.__reference_arr, dtype=int_type
        )
        max_value = max(prediction_arr.max(), reference_arr.max())
        dtype = _get_smallest_fitting_uint(max_value)
        self.set_dtype(dtype)
        self.__dtype = dtype
        self.__n_dim: int = reference_arr.ndim
        self.__ref_labels: tuple[int, ...] = tuple(
            _unique_without_zeros(reference_arr)
        )  # type:ignore
        self.__pred_labels: tuple[int, ...] = tuple(
            _unique_without_zeros(prediction_arr)
        )  # type:ignore
        self.__crop: tuple[slice, ...] = None
        self.__is_cropped: bool = False
        self.__uncropped_shape: tuple[int, ...] = reference_arr.shape

    def crop_data(self, verbose: bool = False):
        """Crops prediction and reference arrays to non-zero regions.

        Args:
            verbose (bool, optional): If True, prints cropping details. Defaults to False.
        """
        if self.__is_cropped:
            return
        if self.__crop is None:
            self.__uncropped_shape = self.__prediction_arr.shape
            self.__crop = _get_paired_crop(
                self.__prediction_arr,
                self.__reference_arr,
            )

        self.__prediction_arr = self.__prediction_arr[self.__crop]
        self.__reference_arr = self.__reference_arr[self.__crop]
        (
            print(
                f"-- Cropped from {self.__uncropped_shape} to {self.__prediction_arr.shape}"
            )
            if verbose
            else None
        )
        self.__is_cropped = True

    def uncrop_data(self, verbose: bool = False):
        """Restores the arrays to their original, uncropped shape.

        Args:
            verbose (bool, optional): If True, prints uncropping details. Defaults to False.
        """
        if self.__is_cropped == False:
            return
        assert (
            self.__uncropped_shape is not None
        ), "Calling uncrop_data() without having cropped first"
        prediction_arr = np.zeros(self.__uncropped_shape)
        prediction_arr[self.__crop] = self.__prediction_arr
        self.__prediction_arr = prediction_arr

        reference_arr = np.zeros(self.__uncropped_shape)
        reference_arr[self.__crop] = self.__reference_arr
        (
            print(
                f"-- Uncropped from {self.__reference_arr.shape} to {self.__uncropped_shape}"
            )
            if verbose
            else None
        )
        self.__reference_arr = reference_arr
        self.__is_cropped = False

    def set_dtype(self, type):
        """Sets the data type for both prediction and reference arrays.

        Args:
            dtype (type): Expected integer type for the arrays.
        """
        assert np.issubdtype(
            type, int_type
        ), "set_dtype: tried to set dtype to something other than integers"
        self.__prediction_arr = self.__prediction_arr.astype(type)
        self.__reference_arr = self.__reference_arr.astype(type)

    @property
    def prediction_arr(self):
        return self.__prediction_arr

    @property
    def reference_arr(self):
        return self.__reference_arr

    @property
    def pred_labels(self):
        return self.__pred_labels

    @property
    def ref_labels(self):
        return self.__ref_labels

    @property
    def n_dim(self):
        return self.__n_dim

    def copy(self):
        """
        Creates an exact copy of this object
        """
        return type(self)(
            prediction_arr=self.__prediction_arr,
            reference_arr=self.__reference_arr,
        )  # type:ignore


class _ProcessingPairInstanced(_ProcessingPair):
    """Represents a processing pair with labeled instances, including unique label counts.

    This subclass tracks additional details about the number of unique instances in each array.

    Attributes:
        n_prediction_instance (int): Number of unique prediction instances.
        n_reference_instance (int): Number of unique reference instances.
    """

    n_prediction_instance: int
    n_reference_instance: int

    def __init__(
        self,
        prediction_arr: np.ndarray,
        reference_arr: np.ndarray,
        n_prediction_instance: int | None = None,
        n_reference_instance: int | None = None,
    ) -> None:
        """Initializes a processing pair for instances.

        Args:
            prediction_arr (np.ndarray): Array of predicted instance labels.
            reference_arr (np.ndarray): Array of reference instance labels.
            dtype (type | None): Expected data type of the arrays.
            n_prediction_instance (int | None, optional): Pre-calculated number of prediction instances.
            n_reference_instance (int | None, optional): Pre-calculated number of reference instances.
        """
        super().__init__(prediction_arr, reference_arr)
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


def _check_array_integrity(
    prediction_arr: np.ndarray, reference_arr: np.ndarray, dtype: type | None = None
):
    """Validates integrity between two arrays, checking shape, dtype, and consistency with `dtype`.

    Args:
        prediction_arr (np.ndarray): The array to be validated.
        reference_arr (np.ndarray): The reference array for comparison.
        dtype (type | None): Expected type of the arrays. If None, dtype validation is skipped.

    Raises:
        AssertionError: If validation fails in any of the following cases:
            - Arrays are not numpy arrays.
            - Shapes of both arrays are not identical.
            - Data types of both arrays do not match.
            - Dtype mismatch when specified.

    Example:
    >>> _check_array_integrity(np.array([1, 2, 3]), np.array([4, 5, 6]), dtype=int)
    """
    assert isinstance(prediction_arr, np.ndarray) and isinstance(
        reference_arr, np.ndarray
    ), "prediction and/or reference are not numpy arrays"
    assert (
        prediction_arr.shape == reference_arr.shape
    ), f"shape mismatch, got {prediction_arr.shape},{reference_arr.shape}"

    min_value = min(prediction_arr.min(), reference_arr.min())
    assert (
        min_value >= 0
    ), "There are negative values in the semantic maps. This is not allowed!"

    # if prediction_arr.dtype != reference_arr.dtype:
    #    print(f"Dtype is equal in prediction and reference, got {prediction_arr.dtype},{reference_arr.dtype}. Intended?")
    # assert prediction_arr.dtype == reference_arr.dtype, f"dtype mismatch, got {prediction_arr.dtype},{reference_arr.dtype}"
    if dtype is not None:
        assert (
            np.issubdtype(prediction_arr.dtype, dtype)
            and np.issubdtype(reference_arr.dtype, dtype)
            # prediction_arr.dtype == dtype and reference_arr.dtype == dtype
        ), f"prediction and/or reference are not dtype {dtype}, got {prediction_arr.dtype} and {reference_arr.dtype}"


class SemanticPair(_ProcessingPair):
    """Represents a semantic processing pair with integer-type arrays for label analysis.

    This class is tailored to scenarios where arrays contain semantic labels rather than instance IDs.
    """

    def __init__(self, prediction_arr: np.ndarray, reference_arr: np.ndarray) -> None:
        super().__init__(prediction_arr, reference_arr)


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
            n_prediction_instance,
            n_reference_instance,
        )  # type:ignore


class MatchedInstancePair(_ProcessingPairInstanced):
    """Represents a matched processing pair for instance maps, handling matched and unmatched labels.

    This class tracks both matched instances and any unmatched labels between prediction
    and reference arrays.

    Attributes:
        missed_reference_labels (list[int]): Reference labels with no matching prediction.
        missed_prediction_labels (list[int]): Prediction labels with no matching reference.
        matched_instances (list[int]): Labels matched between prediction and reference arrays.
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
            n_prediction_instance,
            n_reference_instance,
        )  # type:ignore
        if matched_instances is None:
            matched_instances = [i for i in self.pred_labels if i in self.ref_labels]
        self.matched_instances = matched_instances

        if missed_reference_labels is None:
            missed_reference_labels = list(
                [i for i in self.ref_labels if i not in self.pred_labels]
            )
        self.missed_reference_labels = missed_reference_labels

        if missed_prediction_labels is None:
            missed_prediction_labels = list(
                [i for i in self.pred_labels if i not in self.ref_labels]
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
            prediction_arr=self.prediction_arr.copy(),
            reference_arr=self.reference_arr.copy(),
            n_prediction_instance=self.n_prediction_instance,
            n_reference_instance=self.n_reference_instance,
            missed_reference_labels=self.missed_reference_labels,
            missed_prediction_labels=self.missed_prediction_labels,
            matched_instances=self.matched_instances,
        )


@dataclass
class EvaluateInstancePair:
    """Represents an evaluation of instance segmentation results, comparing reference and prediction data.

    This class is used to store and evaluate metrics for instance segmentation, tracking the number of instances
    and true positives (tp) alongside calculated metrics.

    Attributes:
        reference_arr (np.ndarray): Array containing reference instance labels.
        prediction_arr (np.ndarray): Array containing predicted instance labels.
        num_pred_instances (int): The number of unique instances in the prediction array.
        num_ref_instances (int): The number of unique instances in the reference array.
        tp (int): The number of true positive matches between predicted and reference instances.
        list_metrics (dict[Metric, list[float]]): Dictionary of metric calculations, where each key is a `Metric`
            object, and each value is a list of metric scores (floats).
    """

    reference_arr: np.ndarray
    prediction_arr: np.ndarray
    num_pred_instances: int
    num_ref_instances: int
    tp: int
    list_metrics: dict[Metric, list[float]]


class InputType(_Enum_Compare):
    """Defines the types of input processing pairs available for evaluation.

    This enumeration provides different processing classes for handling various instance segmentation scenarios,
    allowing flexible instantiation of processing pairs based on the desired comparison type.

    Attributes:
        SEMANTIC (SemanticPair): Processes semantic labels, intended for cases without instances.
        UNMATCHED_INSTANCE (UnmatchedInstancePair): Processes instance maps without requiring label matches.
        MATCHED_INSTANCE (MatchedInstancePair): Processes instance maps with label matching between prediction
            and reference.

    Methods:
        __call__(self, prediction_arr: np.ndarray, reference_arr: np.ndarray) -> _ProcessingPair:
            Creates a processing pair based on the specified `InputType`, using the provided prediction
            and reference arrays.

    Example:
        >>> input_type = InputType.MATCHED_INSTANCE
        >>> processing_pair = input_type(prediction_arr, reference_arr)
    """

    SEMANTIC = SemanticPair
    UNMATCHED_INSTANCE = UnmatchedInstancePair
    MATCHED_INSTANCE = MatchedInstancePair

    def __call__(
        self, prediction_arr: np.ndarray, reference_arr: np.ndarray
    ) -> _ProcessingPair:
        return self.value(prediction_arr, reference_arr)


class IntermediateStepsData:
    """Manages intermediate data steps for a processing pipeline, storing and retrieving processing states.

    This class enables step-by-step tracking of data transformations during processing.

    Attributes:
        original_input (_ProcessingPair | None): The original input data before processing steps.
        _intermediatesteps (dict[str, _ProcessingPair]): Dictionary of intermediate processing steps.
    """

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
