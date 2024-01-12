import numpy as np
from scipy.ndimage import _ni_support
from scipy.ndimage._nd_image import euclidean_feature_transform
from scipy.ndimage.morphology import binary_erosion, generate_binary_structure


def _average_symmetric_surface_distance(
    reference,
    prediction,
    voxelspacing=None,
    connectivity=1,
    *args,
) -> float:
    assd = np.mean(
        (
            _average_surface_distance(
                prediction, reference, voxelspacing, connectivity
            ),
            _average_surface_distance(
                reference, prediction, voxelspacing, connectivity
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
    input: np.ndarray,
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

    ft = np.zeros((input.ndim,) + input.shape, dtype=np.int32)

    euclidean_feature_transform(input, sampling, ft)
    # if requested, calculate the distance transform
    if return_distances:
        dt = ft - np.indices(input.shape, dtype=ft.dtype)
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
