from abc import abstractmethod, ABC
from panoptica.utils.datatypes import SemanticPair, UnmatchedInstancePair, MatchedInstancePair
from panoptica._functionals import _connected_components, CCABackend
from panoptica.utils.numpy_utils import _get_smallest_fitting_uint
import numpy as np


class InstanceApproximator(ABC):
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
    def _approximate_instances(self, semantic_pair: SemanticPair, **kwargs) -> UnmatchedInstancePair | MatchedInstancePair:
        """
        Abstract method to be implemented by subclasses for instance approximation.

        Args:
            semantic_pair (SemanticPair): The semantic pair to be approximated.
            **kwargs: Additional keyword arguments.

        Returns:
            UnmatchedInstancePair | MatchedInstancePair: The result of the instance approximation.
        """
        pass

    def approximate_instances(self, semantic_pair: SemanticPair, **kwargs) -> UnmatchedInstancePair | MatchedInstancePair:
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
        # Call algorithm
        instance_pair = self._approximate_instances(semantic_pair, **kwargs)
        # Check validity
        min_value = min(np.min(instance_pair.pred_labels), np.min(instance_pair.ref_labels))
        assert min_value >= 0, "There are negative values in the semantic maps. This is not allowed!"
        # Set dtype to smalles fitting uint
        max_value = max(np.max(instance_pair.pred_labels), np.max(instance_pair.ref_labels))
        dtype = _get_smallest_fitting_uint(max_value)
        instance_pair.prediction_arr.astype(dtype)
        instance_pair.reference_arr.astype(dtype)
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

    def __init__(self, cca_backend: CCABackend) -> None:
        """
        Initialize the ConnectedComponentsInstanceApproximator.

        Args:
            cca_backend (CCABackend): The connected components algorithm backend.
        """
        self.cca_backend = cca_backend

    def _approximate_instances(self, semantic_pair: SemanticPair, **kwargs) -> UnmatchedInstancePair:
        """
        Approximate instances using the connected components algorithm.

        Args:
            semantic_pair (SemanticPair): The semantic pair to be approximated.
            **kwargs: Additional keyword arguments.

        Returns:
            UnmatchedInstancePair: The result of the instance approximation.
        """
        prediction_arr, n_prediction_instance = _connected_components(semantic_pair.prediction_arr, self.cca_backend)
        reference_arr, n_reference_instance = _connected_components(semantic_pair.reference_arr, self.cca_backend)
        return UnmatchedInstancePair(
            prediction_arr=prediction_arr,
            reference_arr=reference_arr,
            n_prediction_instance=n_prediction_instance,
            n_reference_instance=n_reference_instance,
        )
