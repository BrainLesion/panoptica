from __future__ import annotations

from multiprocessing import Pool
from typing import Tuple

import cc3d
import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from panoptica.evaluator import Evaluator
from panoptica.result import PanopticaResult
from panoptica.semantic_evaluation.connected_component_backends import CCABackend
from panoptica.timing import measure_time


class SemanticSegmentationEvaluator(Evaluator):
    """
    Evaluator for semantic segmentation results.

    This class extends the Evaluator class and provides methods for evaluating semantic segmentation masks
    using metrics such as Intersection over Union (IoU) and Dice coefficient.

    Args:
        cca_backend (CCABackend): The backend for connected components labeling (Enum: CCABackend.cc3d or CCABackend.scipy).

    Methods:
        evaluate(reference_mask, prediction_mask, iou_threshold): Evaluate the segmentation masks.
        _label_instances(mask): Label connected components in a segmentation mask.
        _compute_instance_iou(ref_labels, pred_labels, ref_instance_idx, pred_instance_idx): Compute IoU for instances.
        _compute_instance_dice_coefficient(ref_labels, pred_labels, ref_instance_idx, pred_instance_idx): Compute Dice coefficient for instances.
    """

    def __init__(self, cca_backend: CCABackend):
        self.cca_backend = cca_backend
        # TODO consider initializing evaluator with metrics it should compute

    @measure_time
    def evaluate(
        self,
        reference_mask: np.ndarray,
        prediction_mask: np.ndarray,
        iou_threshold: float,
    ) -> PanopticaResult:
        """
        Evaluate the intersection over union (IoU) and Dice coefficient for semantic segmentation masks.

        Args:
            reference_mask (np.ndarray): The reference segmentation mask.
            prediction_mask (np.ndarray): The predicted segmentation mask.
            iou_threshold (float): The IoU threshold for considering a match.

        Returns:
            PanopticaResult: A named tuple containing evaluation results.
        """
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

        # Create a pool of worker processes to parallelize the computation
        with Pool() as pool:
            # Generate all possible pairs of instance indices for IoU computation
            instance_pairs = [
                (ref_labels, pred_labels, ref_idx, pred_idx)
                for ref_idx in range(1, num_ref_instances + 1)
                for pred_idx in range(1, num_pred_instances + 1)
            ]

            # Calculate IoU for all instance pairs in parallel using starmap
            iou_values = pool.starmap(self._compute_instance_iou, instance_pairs)

        # Reshape the resulting IoU values into a matrix
        iou_matrix = np.array(iou_values).reshape(
            (num_ref_instances, num_pred_instances)
        )

        # Use linear_sum_assignment to find the best matches
        ref_indices, pred_indices = linear_sum_assignment(-iou_matrix)

        # Initialize variables for True Positives (tp) and False Positives (fp)
        tp, dice_list, iou_list = 0, [], []

        # Loop through matched instances to compute PQ components
        for ref_idx, pred_idx in zip(ref_indices, pred_indices):
            iou = iou_matrix[ref_idx][pred_idx]
            if iou >= iou_threshold:
                # Match found, increment true positive count and collect IoU and Dice values
                tp += 1
                iou_list.append(iou)

                # Compute Dice for matched instances
                dice = self._compute_instance_dice_coefficient(
                    ref_labels=ref_labels,
                    pred_labels=pred_labels,
                    ref_instance_idx=ref_idx + 1,
                    pred_instance_idx=pred_idx + 1,
                )
                dice_list.append(dice)

        # Create and return the PanopticaResult object with computed metrics
        return PanopticaResult(
            num_ref_instances=num_ref_instances,
            num_pred_instances=num_pred_instances,
            tp=tp,
            dice_list=dice_list,
            iou_list=iou_list,
        )

    def _label_instances(
        self,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, int]:
        """
        Label connected components in a segmentation mask.

        Args:
            mask (np.ndarray): Segmentation mask (2D or 3D array).

        Returns:
            Tuple[np.ndarray, int]:
                - Labeled mask with instances
                - Number of instances found
        """
        if self.cca_backend == CCABackend.cc3d:
            labeled, num_instances = cc3d.connected_components(mask, return_N=True)
        elif self.cca_backend == CCABackend.scipy:
            labeled, num_instances = ndimage.label(mask)

        return labeled, num_instances

    def _compute_instance_iou(
        self,
        ref_labels: np.ndarray,
        pred_labels: np.ndarray,
        ref_instance_idx: int,
        pred_instance_idx: int,
    ) -> float:
        """
        Compute Intersection over Union (IoU) between a specific pair of reference and prediction instances.

        Args:
            ref_labels (np.ndarray): Reference instance labels.
            pred_labels (np.ndarray): Prediction instance labels.
            ref_instance_idx (int): Index of the reference instance.
            pred_instance_idx (int): Index of the prediction instance.

        Returns:
            float: IoU between the specified instances.
        """
        ref_instance_mask = ref_labels == ref_instance_idx
        pred_instance_mask = pred_labels == pred_instance_idx

        return self._compute_iou(
            reference=ref_instance_mask,
            prediction=pred_instance_mask,
        )

    def _compute_instance_dice_coefficient(
        self,
        ref_labels: np.ndarray,
        pred_labels: np.ndarray,
        ref_instance_idx: int,
        pred_instance_idx: int,
    ) -> float:
        """
        Compute the Dice coefficient between a specific pair of instances.

        The Dice coefficient measures the similarity or overlap between two binary masks representing instances.
        It is defined as:

        Dice = (2 * intersection) / (ref_area + pred_area)

        Args:
            ref_labels (np.ndarray): Reference instance labels.
            pred_labels (np.ndarray): Prediction instance labels.
            ref_instance_idx (int): Index of the reference instance.
            pred_instance_idx (int): Index of the prediction instance.

        Returns:
            float: Dice coefficient between the specified instances. A value between 0 and 1, where higher values
            indicate better overlap and similarity between instances.
        """
        ref_instance_mask = ref_labels == ref_instance_idx
        pred_instance_mask = pred_labels == pred_instance_idx

        return self._compute_dice_coefficient(
            reference=ref_instance_mask,
            prediction=pred_instance_mask,
        )
