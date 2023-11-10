import numpy as np
from multiprocessing import Pool
from utils.metrics import _compute_instance_iou
from utils.constants import CCABackend


def _calc_iou_matrix(prediction_arr: np.ndarray, reference_arr: np.ndarray, ref_labels, pred_labels):
    num_ref_instances = len(ref_labels)
    num_pred_instances = len(pred_labels)

    # Create a pool of worker processes to parallelize the computation
    with Pool() as pool:
        # Generate all possible pairs of instance indices for IoU computation
        instance_pairs = [
            (reference_arr, prediction_arr, ref_idx, pred_idx)
            for ref_idx in range(1, num_ref_instances + 1)
            for pred_idx in range(1, num_pred_instances + 1)
        ]

        # Calculate IoU for all instance pairs in parallel using starmap
        iou_values = pool.starmap(_compute_instance_iou, instance_pairs)

    # Reshape the resulting IoU values into a matrix
    iou_matrix = np.array(iou_values).reshape((num_ref_instances, num_pred_instances))
    return iou_matrix


def _map_labels(arr: np.ndarray, label_map: dict[np.integer, np.integer]) -> np.ndarray:
    """Maps labels in the given array according to the label_map dictionary.
    Args:
        label_map (dict): A dictionary that maps the original label values (str or int) to the new label values (int).

    Returns:
        np.ndarray: Returns a copy of the remapped array
    """
    data = arr.copy()
    for v in np.unique(data):
        if v in label_map:  # int needed to match non-integer data-types
            data[arr == v] = label_map[v]
    return data


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
