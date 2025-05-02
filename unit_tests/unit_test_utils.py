import numpy as np


def case_simple_identical():
    # trivial 100% overlap
    prediction_arr = np.array(
        [
            [0, 1, 1, 1],
            [0, 1, 1, 1],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
        ]
    )
    return prediction_arr, prediction_arr.copy()


def case_simple_nooverlap():
    # binary opposites
    prediction_arr = np.array(
        [
            [0, 1, 0, 1],
            [0, 1, 0, 1],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
        ]
    )
    reference_arr = 1 - prediction_arr
    return prediction_arr, reference_arr


def case_simple_shifted():
    # binary opposites
    prediction_arr = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
        ]
    )
    reference_arr = np.array(
        [
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
        ]
    )
    return prediction_arr, reference_arr


def case_simple_overpredicted():
    # reference is real subset of prediction
    prediction_arr = np.array(
        [
            [0, 0, 1, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
            [0, 1, 1, 0],
        ]
    )
    reference_arr = np.array(
        [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ]
    )
    return prediction_arr, reference_arr


def case_simple_underpredicted():
    # prediction is real subset of reference
    prediction_arr = np.array(
        [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ]
    )
    reference_arr = np.array(
        [
            [0, 0, 1, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
            [0, 1, 1, 0],
        ]
    )
    return prediction_arr, reference_arr


def case_simple_overlap_but_large_discrepancy():
    # prediction is real subset of reference
    prediction_arr = np.array(
        [
            [1, 1, 1, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 0],
            [1, 0, 0, 0],
        ]
    )
    reference_arr = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ]
    )
    return prediction_arr, reference_arr


def case_multiple_overlapping_instances():
    # Create empty arrays
    prediction_arr = np.zeros((30, 30), dtype=np.uint8)
    reference_arr = np.zeros((30, 30), dtype=np.uint8)

    # Ground truth
    reference_arr[1:10, 1:10] = 1
    reference_arr[11:20, 11:20] = 2

    # Prediction
    prediction_arr[1:3, 1:3] = 1
    prediction_arr[7:13, 7:13] = 2
    return prediction_arr, reference_arr
