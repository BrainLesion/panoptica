import numpy as np
from panoptica.utils.metrics import _compute_instance_iou
from panoptica.utils.constants import CCABackend
from multiprocessing import Pool


def _calc_iou_matrix(prediction_arr: np.ndarray, reference_arr: np.ndarray, ref_labels: tuple[int, ...], pred_labels: tuple[int, ...]):
    """
    Calculate the Intersection over Union (IoU) matrix between reference and prediction arrays.

    Args:
        prediction_arr (np.ndarray): Numpy array containing the prediction labels.
        reference_arr (np.ndarray): Numpy array containing the reference labels.
        ref_labels (list[int]): List of unique reference labels.
        pred_labels (list[int]): List of unique prediction labels.

    Returns:
        np.ndarray: IoU matrix where each element represents the IoU between a reference and prediction instance.

    Example:
    >>> _calc_iou_matrix(np.array([1, 2, 3]), np.array([4, 5, 6]), [1, 2, 3], [4, 5, 6])
    array([[0. , 0. , 0. ],
           [0. , 0. , 0. ],
           [0. , 0. , 0. ]])
    """
    num_ref_instances = len(ref_labels)
    num_pred_instances = len(pred_labels)

    # Create a pool of worker processes to parallelize the computation
    with Pool() as pool:
        #    # Generate all possible pairs of instance indices for IoU computation
        instance_pairs = [(reference_arr, prediction_arr, ref_idx, pred_idx) for ref_idx in ref_labels for pred_idx in pred_labels]

        # Calculate IoU for all instance pairs in parallel using starmap
        iou_values = pool.starmap(_compute_instance_iou, instance_pairs)

    # Reshape the resulting IoU values into a matrix
    iou_matrix = np.array(iou_values).reshape((num_ref_instances, num_pred_instances))
    return iou_matrix


def _map_labels(arr: np.ndarray, label_map: dict[np.integer, np.integer]) -> np.ndarray:
    """
    Maps labels in the given array according to the label_map dictionary.

    Args:
        label_map (dict): A dictionary that maps the original label values (str or int) to the new label values (int).

    Returns:
        np.ndarray: Returns a copy of the remapped array
    """
    k = np.array(list(label_map.keys()))
    v = np.array(list(label_map.values()))

    mapping_ar = np.arange(arr.max() + 1, dtype=arr.dtype)
    mapping_ar[k] = v
    return mapping_ar[arr]


def _connected_components(
    array: np.ndarray,
    cca_backend: CCABackend,
) -> tuple[np.ndarray, int]:
    """
    Label connected components in a binary array using a specified connected components algorithm.

    Args:
        array (np.ndarray): Binary array containing connected components.
        cca_backend (CCABackend): Enum indicating the connected components algorithm backend (CCABackend.cc3d or CCABackend.scipy).

    Returns:
        tuple[np.ndarray, int]: A tuple containing the labeled array and the number of connected components.

    Raises:
        NotImplementedError: If the specified connected components algorithm backend is not implemented.

    Example:
    >>> _connected_components(np.array([[1, 0, 1], [0, 1, 1], [1, 0, 0]]), CCABackend.scipy)
    (array([[1, 0, 2], [0, 3, 3], [4, 0, 0]]), 4)
    """
    if cca_backend == CCABackend.cc3d:
        import cc3d

        cc_arr, n_instances = cc3d.connected_components(array, return_N=True)
    elif cca_backend == CCABackend.scipy:
        from scipy.ndimage import label

        cc_arr, n_instances = label(array)
    else:
        raise NotImplementedError(cca_backend)

    return cc_arr, n_instances
