import numpy as np
from scipy.ndimage import _ni_support, binary_erosion, generate_binary_structure
from scipy.ndimage._nd_image import euclidean_feature_transform


def _compute_instance_average_symmetric_surface_distance(
    ref_labels: np.ndarray,
    pred_labels: np.ndarray,
    ref_instance_idx: int | None = None,
    pred_instance_idx: int | None = None,
    voxelspacing=None,
    connectivity=1,
):
    if ref_instance_idx is None and pred_instance_idx is None:
        return _average_symmetric_surface_distance(
            reference=ref_labels,
            prediction=pred_labels,
            voxelspacing=voxelspacing,
            connectivity=connectivity,
        )
    ref_instance_mask = ref_labels == ref_instance_idx
    pred_instance_mask = pred_labels == pred_instance_idx
    return _average_symmetric_surface_distance(
        reference=ref_instance_mask,
        prediction=pred_instance_mask,
        voxelspacing=voxelspacing,
        connectivity=connectivity,
    )


def _average_symmetric_surface_distance(
    reference,
    prediction,
    voxelspacing=None,
    connectivity=1,
    *args,
) -> float:
    """ASSD is computed by computing the average of the bidrectionally computed ASD."""
    assd = np.mean(
        (
            _average_surface_distance(
                reference=prediction,
                prediction=reference,
                voxelspacing=voxelspacing,
                connectivity=connectivity,
            ),
            _average_surface_distance(
                reference=reference,
                prediction=prediction,
                voxelspacing=voxelspacing,
                connectivity=connectivity,
            ),
        )
    )
    return float(assd)


def _average_surface_distance(reference, prediction, voxelspacing=None, connectivity=1):
    sds = __surface_distances(reference, prediction, voxelspacing, connectivity)
    asd = sds.mean()
    return asd


def __surface_distances(reference, prediction, voxelspacing=None, connectivity=1):
    """
    The distances between the surface voxel of binary objects in result and their
    nearest partner surface voxel of a binary object in reference.
    """
    prediction = np.atleast_1d(prediction.astype(bool))
    reference = np.atleast_1d(reference.astype(bool))
    if voxelspacing is not None:
        # Protected access presented by Scipy
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, prediction.ndim)
        voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()

    # binary structure
    footprint = generate_binary_structure(prediction.ndim, connectivity)

    # test for emptiness
    # if 0 == np.count_nonzero(result):
    #    raise RuntimeError("The first supplied array does not contain any binary object.")
    # if 0 == np.count_nonzero(reference):
    #    raise RuntimeError("The second supplied array does not contain any binary object.")

    # extract only 1-pixel border line of objects
    result_border = prediction ^ binary_erosion(
        prediction, structure=footprint, iterations=1
    )
    reference_border = reference ^ binary_erosion(
        reference, structure=footprint, iterations=1
    )

    # compute average surface distance
    # Note: scipys distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    dt = _distance_transform_edt(~reference_border, sampling=None)
    sds = dt[result_border]

    return sds


def _distance_transform_edt(
    input_array: np.ndarray,
    sampling=None,
    return_distances=True,
    return_indices=False,
):
    # calculate the feature transform
    # input = np.atleast_1d(np.where(input, 1, 0).astype(np.int8))
    # if sampling is not None:
    #    sampling = _ni_support._normalize_sequence(sampling, input.ndim)
    #    sampling = np.asarray(sampling, dtype=np.float64)
    #    if not sampling.flags.contiguous:
    #        sampling = sampling.copy()

    ft = np.zeros((input_array.ndim,) + input_array.shape, dtype=np.int32)

    euclidean_feature_transform(input_array, sampling, ft)
    # if requested, calculate the distance transform
    if return_distances:
        dt = ft - np.indices(input_array.shape, dtype=ft.dtype)
        dt = dt.astype(np.float64)
        # if sampling is not None:
        #    for ii in range(len(sampling)):
        #        dt[ii, ...] *= sampling[ii]
        np.multiply(dt, dt, dt)

        dt = np.add.reduce(dt, axis=0)
        dt = np.sqrt(dt)

    # construct and return the result
    result = []
    if return_distances:
        result.append(dt)
    if return_indices:
        result.append(ft)

    if len(result) == 2:
        return tuple(result)
    elif len(result) == 1:
        return result[0]
    else:
        return None
