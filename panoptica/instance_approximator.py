from abc import abstractmethod, ABC
from utils.datatypes import SemanticPair, UnmatchedInstancePair, MatchedInstancePair
from utils.connected_component_backends import CCABackend
import numpy as np


class InstanceApproximator(ABC):
    @abstractmethod
    def _approximate_instances(self, semantic_pair: SemanticPair, **kwargs) -> UnmatchedInstancePair | MatchedInstancePair:
        pass

    def approximate_instances(self, semantic_pair: SemanticPair, **kwargs) -> UnmatchedInstancePair | MatchedInstancePair:
        # TODO call _approx
        max_value = max(np.max(prediction_arr), np.max(reference_arr))
        # reduce to smallest uint


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


def _connected_components(
    array: np.ndarray,
    cca_backend: CCABackend,
) -> tuple[np.ndarray, int]:
    if cca_backend == CCABackend.cc3d:
        import cc3d

        cc_arr, n_instances = cc3d.connected_components(array, return_N=True)
    elif cca_backend == CCABackend.scipy:
        from scipy.ndimage import label

        cc_arr, n_instances = label(array)
    else:
        raise NotImplementedError(cca_backend)

    return cc_arr, n_instances
