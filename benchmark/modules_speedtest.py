from auxiliary.nifti.io import read_nifti
from auxiliary.turbopath import turbopath
import os
import cpuinfo
import json


from panoptica import (
    ConnectedComponentsInstanceApproximator,
    NaiveThresholdMatching,
    SemanticPair,
    UnmatchedInstancePair,
    MatchedInstancePair,
)
from panoptica.instance_evaluator import evaluate_matched_instance
from time import perf_counter
import numpy as np
from tqdm import tqdm
import csv

import pandas as pd

directory = turbopath(__file__).parent.parent

ref_masks = read_nifti(directory + "/examples/spine_seg/semantic/ref.nii.gz")
pred_masks = read_nifti(directory + "/examples/spine_seg/semantic/pred.nii.gz")

platform_name = "ryzen9_new"

csv_out = directory + "/benchmark/" + platform_name + "/performance_"


def evaluate_nparray(array, return_as_dict=False, verbose=False):
    if verbose:
        print("evaluate:", array)
    assert len(array) > 0, f"Got empty list to evaluate! Received {str(array)}"
    array = [a for a in array if not np.isnan(a)]
    min_val = min(array)
    max_val = max(array)
    avg = sum(array) / len(array)
    np_arr = np.asarray(array)
    std = np.std(np_arr)
    var = np.var(np_arr)
    percentile_01 = np.percentile(np_arr, 1)
    percentile_05 = np.percentile(np_arr, 5)
    percentile_10 = np.percentile(np_arr, 10)
    percentile_25 = np.percentile(np_arr, 25)
    percentile_75 = np.percentile(np_arr, 75)
    percentile_90 = np.percentile(np_arr, 90)
    percentile_95 = np.percentile(np_arr, 95)
    percentile_99 = np.percentile(np_arr, 99)

    percentiles = {
        1: percentile_01,
        5: percentile_05,
        10: percentile_10,
        25: percentile_25,
        75: percentile_75,
        90: percentile_90,
        95: percentile_95,
        99: percentile_99,
    }

    dict = {
        "min": min_val,
        "max": max_val,
        "avg": avg,
        "var": var,
        "std": std,
        "percentiles": percentiles,
    }
    if verbose:
        print(dict)
    if return_as_dict:
        return dict
    return min_val, max_val, avg, var, std, percentiles


def test_input(processing_pair: SemanticPair):
    # Crops away unecessary space of zeroes
    processing_pair = processing_pair.copy()
    processing_pair.crop_data()
    #
    start1 = perf_counter()
    unmatched_instance_pair = instance_approximator.approximate_instances(
        semantic_pair=processing_pair
    )
    time1 = perf_counter() - start1
    #
    start2 = perf_counter()
    matched_instance_pair = instance_matcher.match_instances(
        unmatched_instance_pair=unmatched_instance_pair
    )
    time2 = perf_counter() - start2
    #
    start3 = perf_counter()
    result = evaluate_matched_instance(
        matched_instance_pair,
        decision_threshold=iou_threshold,
    )
    time3 = perf_counter() - start3
    return time1, time2, time3


instance_approximator = ConnectedComponentsInstanceApproximator()
instance_matcher = NaiveThresholdMatching()
iou_threshold = 0.5

n_iterations = 42
time_phase: tuple[list[float], ...] = ([], [], [])

processing_pair_3D_mri_spine = SemanticPair(pred_masks, ref_masks)

# simple
a = np.zeros([50, 50], dtype=np.uint16)
b = a.copy().astype(a.dtype)
a[20:40, 10:20] = 1
a[41:45, 10:20] = 3
b[20:35, 10:20] = 2
b[40:50, 5:30] = 4
processing_pair_2d_simple = SemanticPair(b, a)

# only diagonal large 2D
a = np.zeros([1000, 1000], dtype=np.uint16)
b = a.copy().astype(a.dtype)
np.fill_diagonal(a, np.arange(1, 1000))
np.fill_diagonal(b, np.arange(1000, 0, -1))
processing_pair_2d_diagonal = SemanticPair(b, a)

# Full 3D matrix
size = 100
a = np.arange(size * size * size).reshape(size, size, size).astype(np.uint16)
a_max = a.max()
n_instances = 40
a = a // (a_max // n_instances)
b = a.copy() * 2
b = b.astype(a.dtype)
# print(np.unique(a))
processing_pair_3d_dense = SemanticPair(b, a)


test_samples = {
    "2d_simple": processing_pair_2d_simple,
    # "2d_diagonal": processing_pair_2d_diagonal,
    "3d_spine": processing_pair_3D_mri_spine,
    "3d_dense": processing_pair_3d_dense,
}

timings = {k: {1: [], 2: [], 3: []} for k in test_samples.keys()}


if __name__ == "__main__":
    # CPU information
    cpu_dict = cpuinfo.get_cpu_info()

    cpu_json_path = directory + "/benchmark/data/" + platform_name + "/platform.json"
    os.makedirs(cpu_json_path.parent, exist_ok=True)

    with open(cpu_json_path, "w") as fp:
        json.dump(cpu_dict, fp, indent=4)

    report_df_list = []

    # start
    for i in range(n_iterations):
        print(f"Iteration {i}")
        # do each test, write csv accordingly
        for sample_name, input in tqdm(test_samples.items()):
            time1, time2, time3 = test_input(input)
            # print(sample_name, time1, time2, time3)
            timings[sample_name][1].append(time1)
            timings[sample_name][2].append(time2)
            timings[sample_name][3].append(time3)

    for sample_name, timing_dict in timings.items():
        print()
        print()
        print(sample_name)
        csv_name = csv_out + sample_name + ".csv"
        with open(csv_name, "w", newline="") as csvfile:
            spamwriter = csv.writer(
                csvfile, delimiter=";", quotechar="|", quoting=csv.QUOTE_MINIMAL
            )
            spamwriter.writerow(timing_dict[1])
            spamwriter.writerow(timing_dict[2])
            spamwriter.writerow(timing_dict[3])

        # to pandas
        data_dict = {
            "approximation": timing_dict[1],
            "matching": timing_dict[2],
            "evaluation": timing_dict[3],
            "condition": sample_name,
            "platform": platform_name,
        }

        df = pd.DataFrame.from_dict(data_dict)

        report_df_list.append(df)

        first_phase = evaluate_nparray(timing_dict[1], return_as_dict=True)
        second_phase = evaluate_nparray(timing_dict[2], return_as_dict=True)
        third_phase = evaluate_nparray(timing_dict[3], return_as_dict=True)

        print(
            "1st phase:",
            round(first_phase["avg"], ndigits=4),
            "+-",
            round(first_phase["std"], ndigits=4),
        )
        print(
            "2nd phase:",
            round(second_phase["avg"], ndigits=4),
            "+-",
            round(second_phase["std"], ndigits=4),
        )
        print(
            "3rd phase:",
            round(third_phase["avg"], ndigits=4),
            "+-",
            round(third_phase["std"], ndigits=4),
        )

    # Concatenate the list of DataFrames into a single DataFrame
    agg_df = pd.concat(report_df_list, ignore_index=True)

    # Display the result
    print(agg_df)

    csv_path = directory + "/benchmark/data/" + platform_name + "/dataframe.csv"
    os.makedirs(csv_path.parent, exist_ok=True)
    agg_df.to_csv(csv_path)
    print(f"saved dataframe into {csv_path}")
