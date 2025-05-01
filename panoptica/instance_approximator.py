from abc import ABC, abstractmethod, ABCMeta

import numpy as np

from panoptica.utils.constants import CCABackend
from panoptica._functionals import _connected_components

# from panoptica.utils.numpy_utils import _get_smallest_fitting_uint
from panoptica.utils.processing_pair import (
    MatchedInstancePair,
    SemanticPair,
    UnmatchedInstancePair,
)
from panoptica.utils.config import SupportsConfig

# Add LabelGroup import
from panoptica.utils.label_group import LabelGroup, LabelPartGroup


class InstanceApproximator(SupportsConfig, metaclass=ABCMeta):
    """
    Abstract base class for instance approximation algorithms in panoptic segmentation evaluation.

    Attributes:
        None

    Methods:
        _approximate_instances(self, semantic_pair: SemanticPair, **kwargs) -> UnmatchedInstancePair | MatchedInstancePair:
            Abstract method to be implemented by subclasses for instance approximation.

        approximate_instances(self, semantic_pair: SemanticPair, **kwargs) -> UnmatchedInstancePair | MatchedInstancePair:
            Perform instance approximation on the given SemanticPair.

    Raises:
        AssertionError: If there are negative values in the semantic maps, which is not allowed.

    Example:
    >>> class CustomInstanceApproximator(InstanceApproximator):
    ...     def _approximate_instances(self, semantic_pair: SemanticPair, **kwargs) -> UnmatchedInstancePair | MatchedInstancePair:
    ...         # Implementation of instance approximation algorithm
    ...         pass
    ...
    >>> approximator = CustomInstanceApproximator()
    >>> semantic_pair = SemanticPair(...)
    >>> result = approximator.approximate_instances(semantic_pair)
    """

    @abstractmethod
    def _approximate_instances(
        self,
        semantic_pair: SemanticPair,
        label_group: LabelGroup | None = None,
        **kwargs,
    ) -> UnmatchedInstancePair | MatchedInstancePair:
        """
        Abstract method to be implemented by subclasses for instance approximation.

        Args:
            semantic_pair (SemanticPair): The semantic pair to be approximated.
            label_group (LabelGroup | None, optional): Information about the label group being processed. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            UnmatchedInstancePair | MatchedInstancePair: The result of the instance approximation.
        """
        pass

    def _yaml_repr(cls, node) -> dict:
        raise NotImplementedError(
            f"Tried to get yaml representation of abstract class {cls.__name__}"
        )
        return {}

    def approximate_instances(
        self,
        semantic_pair: SemanticPair,
        verbose: bool = False,
        label_group: LabelGroup | None = None,
        **kwargs,
    ) -> UnmatchedInstancePair | MatchedInstancePair:
        """
        Perform instance approximation on the given SemanticPair.

        Args:
            semantic_pair (SemanticPair): The semantic pair to be approximated.
            verbose (bool, optional): If True, prints detailed information. Defaults to False.
            label_group (LabelGroup | None, optional): Information about the label group being processed. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            UnmatchedInstancePair | MatchedInstancePair: The result of the instance approximation.

        Raises:
            AssertionError: If there are negative values in the semantic maps, which is not allowed.
        """
        # Check validity
        pred_labels, ref_labels = semantic_pair.pred_labels, semantic_pair.ref_labels
        pred_label_range = (
            (np.min(pred_labels), np.max(pred_labels))
            if len(pred_labels) > 0
            else (0, 0)
        )
        ref_label_range = (
            (np.min(ref_labels), np.max(ref_labels)) if len(ref_labels) > 0 else (0, 0)
        )
        min_value = min(np.min(pred_label_range[0]), np.min(ref_label_range[0]))
        assert (
            min_value >= 0
        ), "There are negative values in the semantic maps. This is not allowed!"

        # If label_group is LabelPartGroup, force OneHotConnectedComponentsInstanceApproximator
        if isinstance(label_group, LabelPartGroup):
            instance_pair = OneHotConnectedComponentsInstanceApproximator(
                cca_backend=CCABackend.cc3d
            )._approximate_instances(semantic_pair, label_group=label_group, **kwargs)
        else:
            # Call the instance approximation algorithm
            instance_pair = self._approximate_instances(
                semantic_pair, label_group=label_group, **kwargs
            )

        return instance_pair


class ConnectedComponentsInstanceApproximator(InstanceApproximator):
    """
    Instance approximator using connected components algorithm for panoptic segmentation evaluation.

    Attributes:
        cca_backend (CCABackend): The connected components algorithm backend.

    Methods:
        __init__(self, cca_backend: CCABackend) -> None:
            Initialize the ConnectedComponentsInstanceApproximator.
        _approximate_instances(self, semantic_pair: SemanticPair, **kwargs) -> UnmatchedInstancePair:
            Approximate instances using the connected components algorithm.

    Example:
    >>> cca_approximator = ConnectedComponentsInstanceApproximator(cca_backend=CCABackend.cc3d)
    >>> semantic_pair = SemanticPair(...)
    >>> result = cca_approximator.approximate_instances(semantic_pair)
    """

    def __init__(self, cca_backend: CCABackend | None = None) -> None:
        """
        Initialize the ConnectedComponentsInstanceApproximator.

        Args:
            cca_backend (CCABackend): The connected components algorithm backend. If None, will use cc3d for 3D and scipy for 1D and 2D inputs.
        """
        self.cca_backend = cca_backend

    def _approximate_instances(
        self, semantic_pair: SemanticPair, **kwargs
    ) -> UnmatchedInstancePair:
        """
        Approximate instances using the connected components algorithm.

        Args:
            semantic_pair (SemanticPair): The semantic pair to be approximated.
            **kwargs: Additional keyword arguments.

        Returns:
            UnmatchedInstancePair: The result of the instance approximation.
        """
        cca_backend = self.cca_backend
        if cca_backend is None:
            cca_backend = (
                CCABackend.cc3d if semantic_pair.n_dim >= 3 else CCABackend.scipy
            )
        assert cca_backend is not None

        empty_prediction = len(semantic_pair.pred_labels) == 0
        empty_reference = len(semantic_pair.ref_labels) == 0
        prediction_arr, n_prediction_instance = (
            _connected_components(semantic_pair.prediction_arr, cca_backend)
            if not empty_prediction
            else (semantic_pair.prediction_arr, 0)
        )
        reference_arr, n_reference_instance = (
            _connected_components(semantic_pair.reference_arr, cca_backend)
            if not empty_reference
            else (semantic_pair.reference_arr, 0)
        )

        return UnmatchedInstancePair(
            prediction_arr=prediction_arr,
            reference_arr=reference_arr,
            n_prediction_instance=n_prediction_instance,
            n_reference_instance=n_reference_instance,
        )
        
    @classmethod
    def _yaml_repr(cls, node) -> dict:
        return {"cca_backend": node.cca_backend}


class OneHotConnectedComponentsInstanceApproximator(InstanceApproximator):
    """
    Instance approximator that first applies one-hot encoding to the prediction and reference arrays,
    then runs connected components on each channel and merges the results.
    """

    def __init__(self, cca_backend: CCABackend | None = None) -> None:
        self.cca_backend = cca_backend

    def _one_hot(self, arr):
        n_classes = int(np.max(arr)) + 1
        arr_shape = arr.shape  # e.g., (H, W) or (D, H, W)
        one_hot = np.eye(n_classes)[arr.astype(int)]  # shape: (*arr_shape, C)
        # Move the class axis to the front: (C, *arr_shape)
        one_hot = np.moveaxis(one_hot, -1, 0)
        return one_hot, arr_shape

    def _approximate_instances(
        self, semantic_pair: SemanticPair, label_group: LabelGroup | None = None
    ) -> UnmatchedInstancePair:
        cca_backend = self.cca_backend
        if label_group is not None and isinstance(label_group, LabelPartGroup):
            cca_backend = CCABackend.cc3d
        elif cca_backend is None:
            cca_backend = (
                CCABackend.cc3d if semantic_pair.n_dim >= 3 else CCABackend.scipy
            )
        assert cca_backend is not None

        _, n_prediction_instance = _connected_components(
            semantic_pair.prediction_arr, cca_backend
        )
        _, n_reference_instance = _connected_components(
            semantic_pair.reference_arr, cca_backend
        )

        # One-hot encode
        prediction_arr, prediction_arr_shape = self._one_hot(
            semantic_pair.prediction_arr
        )
        reference_arr, reference_arr_shape = self._one_hot(semantic_pair.reference_arr)
        
        print(
            f"Prediction shape: {prediction_arr_shape}, Reference shape: {reference_arr_shape}"
        )

        for i in range(prediction_arr.shape[0]):
            print(f"Processing channel {i} of {prediction_arr.shape[0]}")
            # If channel 0, inverse it before connected components
            if i == 0:
                prediction_arr[i] = 1 - prediction_arr[i]
                reference_arr[i] = 1 - reference_arr[i]
            prediction_arr[i], _ = _connected_components(prediction_arr[i], cca_backend)
            reference_arr[i], _ = _connected_components(reference_arr[i], cca_backend)

            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(prediction_arr[i], cmap='gray')
            ax[0].set_title(f"Prediction Channel {i}")
            ax[1].imshow(reference_arr[i], cmap='gray')
            ax[1].set_title(f"Reference Channel {i}")
            plt.suptitle(f"Connected Components for Channel {i}")
            plt.show()

        # flatten to meet UnmatchedInstancePair requirements
        prediction_arr = prediction_arr.flatten()
        reference_arr = reference_arr.flatten()

        # Ensure arrays are integer type
        prediction_arr = prediction_arr.astype(np.int64)
        reference_arr = reference_arr.astype(np.int64)

        return UnmatchedInstancePair(
            prediction_arr=prediction_arr,
            reference_arr=reference_arr,
            n_prediction_instance=n_prediction_instance,
            n_reference_instance=n_reference_instance,
        )

    @classmethod
    def _yaml_repr(cls, node) -> dict:
        return {"cca_backend": node.cca_backend}