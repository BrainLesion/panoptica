from abc import ABC, abstractmethod, ABCMeta

import numpy as np

from panoptica.utils.constants import CCABackend
from panoptica._functionals import _connected_components
from panoptica.utils.numpy_utils import _get_smallest_fitting_uint
from panoptica.utils.processing_pair import (
    MatchedInstancePair,
    SemanticPair,
    UnmatchedInstancePair,
)
from panoptica.utils.config import SupportsConfig


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
        self, semantic_pair: SemanticPair, **kwargs
    ) -> UnmatchedInstancePair | MatchedInstancePair:
        """
        Abstract method to be implemented by subclasses for instance approximation.

        Args:
            semantic_pair (SemanticPair): The semantic pair to be approximated.
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
        self, semantic_pair: SemanticPair, verbose: bool = False, **kwargs
    ) -> UnmatchedInstancePair | MatchedInstancePair:
        """
        Perform instance approximation on the given SemanticPair.

        Args:
            semantic_pair (SemanticPair): The semantic pair to be approximated.
            **kwargs: Additional keyword arguments.

        Returns:
            UnmatchedInstancePair | MatchedInstancePair: The result of the instance approximation.

        Raises:
            AssertionError: If there are negative values in the semantic maps, which is not allowed.
        """
        # Check validity
        pred_labels, ref_labels = semantic_pair._pred_labels, semantic_pair._ref_labels
        pred_label_range = (
            (np.min(pred_labels), np.max(pred_labels))
            if len(pred_labels) > 0
            else (0, 0)
        )
        ref_label_range = (
            (np.min(ref_labels), np.max(ref_labels)) if len(ref_labels) > 0 else (0, 0)
        )
        #
        min_value = min(np.min(pred_label_range[0]), np.min(ref_label_range[0]))
        assert (
            min_value >= 0
        ), "There are negative values in the semantic maps. This is not allowed!"
        # Set dtype to smalles fitting uint
        max_value = max(np.max(pred_label_range[1]), np.max(ref_label_range[1]))
        dtype = _get_smallest_fitting_uint(max_value)
        semantic_pair.set_dtype(dtype)
        print(f"-- Set dtype to {dtype}") if verbose else None

        # Call algorithm
        instance_pair = self._approximate_instances(semantic_pair, **kwargs)
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

        empty_prediction = len(semantic_pair._pred_labels) == 0
        empty_reference = len(semantic_pair._ref_labels) == 0
        prediction_arr, n_prediction_instance = (
            _connected_components(semantic_pair._prediction_arr, cca_backend)
            if not empty_prediction
            else (semantic_pair._prediction_arr, 0)
        )
        reference_arr, n_reference_instance = (
            _connected_components(semantic_pair._reference_arr, cca_backend)
            if not empty_reference
            else (semantic_pair._reference_arr, 0)
        )

        dtype = _get_smallest_fitting_uint(
            max(prediction_arr.max(), reference_arr.max())
        )

        return UnmatchedInstancePair(
            prediction_arr=prediction_arr.astype(dtype),
            reference_arr=reference_arr.astype(dtype),
            n_prediction_instance=n_prediction_instance,
            n_reference_instance=n_reference_instance,
        )

    @classmethod
    def _yaml_repr(cls, node) -> dict:
        return {"cca_backend": node.cca_backend}
