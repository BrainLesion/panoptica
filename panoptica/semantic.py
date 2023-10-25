from typing import Tuple

import numpy as np

import cc3d
from scipy import ndimage

from .evaluator import Evaluator


class SemanticSegmentationEvaluator(Evaluator):
    def __init__(self, cca_backend: str):
        self.cca_backend = cca_backend

    def evaluate(
        self,
        reference_mask: np.ndarray,
        prediction_mask: np.ndarray,
        iou_threshold: float,
    ):
        ref_labels, num_ref_instances = self._label_instances(
            mask=reference_mask,
        )

        pred_labels, num_pred_instances = self._label_instances(
            mask=prediction_mask,
        )

        self._handle_edge_cases(
            num_ref_instances=num_ref_instances,
            num_pred_instances=num_pred_instances,
        )

        

    def _label_instances(
        self,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, int]:
        """
        Label connected components in a segmentation mask.

        Args:
            mask (np.ndarray): segmentation mask (2D or 3D array).
            cca_backend (str): Backend for connected components labeling. Should be "cc3d" or "scipy".

        Returns:
            Tuple[np.ndarray, int]:
                - Labeled mask with instances
                - Number of instances found
        """
        if self.cca_backend == "cc3d":
            labeled, num_instances = cc3d.connected_components(mask, return_N=True)
        elif self.cca_backend == "scipy":
            labeled, num_instances = ndimage.label(mask)
        else:
            raise NotImplementedError(f"Unsupported cca_backend: {self.cca_backend}")
        return labeled, num_instances
