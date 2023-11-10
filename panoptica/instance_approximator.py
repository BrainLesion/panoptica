from abc import abstractmethod, ABC
from panoptica.utils.datatypes import SemanticPair, UnmatchedInstancePair, MatchedInstancePair
from panoptica._functionals import _connected_components, CCABackend
from panoptica.utils.numpy_utils import _get_smallest_fitting_uint
import numpy as np


class InstanceApproximator(ABC):
    @abstractmethod
    def _approximate_instances(self, semantic_pair: SemanticPair, **kwargs) -> UnmatchedInstancePair | MatchedInstancePair:
        pass

    def approximate_instances(self, semantic_pair: SemanticPair, **kwargs) -> UnmatchedInstancePair | MatchedInstancePair:
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
    def __init__(self, cca_backend: CCABackend) -> None:
        self.cca_backend = cca_backend

    def _approximate_instances(self, semantic_pair: SemanticPair, **kwargs) -> UnmatchedInstancePair:
        prediction_arr, n_prediction_instance = _connected_components(semantic_pair.prediction_arr, self.cca_backend)
        reference_arr, n_reference_instance = _connected_components(semantic_pair.reference_arr, self.cca_backend)
        return UnmatchedInstancePair(
            prediction_arr=prediction_arr,
            reference_arr=reference_arr,
            n_prediction_instance=n_prediction_instance,
            n_reference_instance=n_reference_instance,
        )
