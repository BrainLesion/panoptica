from medpy import metric
from BIDS import NII
from BIDS.core.np_utils import np_extract_label

# from panoptica.assd import my_assd
from repo.panoptica.utils.assd import my_assd
from time import perf_counter
import numpy as np
from multiprocessing import Pool

gt = "/media/hendrik/be5e95dd-27c8-4c31-adc5-7b75f8ebd5c5/data/hendrik/panoptica/data/gt/verse012_seg.nii.gz"

pred = "/media/hendrik/be5e95dd-27c8-4c31-adc5-7b75f8ebd5c5/data/hendrik/panoptica/data/submissions/christian_payer/docker_phase2/results/verse012_seg.nii.gz"


def extract_both(pred_arr, gt_arr, label: int):
    pred_l = np_extract_label(pred_arr, label, inplace=False)
    gt_l = np_extract_label(gt_arr, label, inplace=False)
    return pred_l, gt_l


pred_nii = NII.load(pred, seg=True)
pred_nii.map_labels_({l: idx + 1 for idx, l in enumerate(pred_nii.unique())}, verbose=False)
gt_nii = NII.load(gt, seg=True)
gt_nii.map_labels_({l: idx + 1 for idx, l in enumerate(gt_nii.unique())}, verbose=False)

pred_arr = pred_nii.get_seg_array()
gt_arr = gt_nii.get_seg_array()

iterations = 3

medpy_result = 1.5266468819541414

time_medpy = []
time_my = []

labels = pred_nii.unique()

for i in range(iterations):
    start = perf_counter()
    # label_list = [l for l in labels if l in gt_arr]
    pairs = (extract_both(pred_arr, gt_arr, l) for l in labels if l in gt_arr)
    # for l in label_list:
    #    pred_l = np_extract_label(pred_arr, l, inplace=False)
    #    gt_l = np_extract_label(gt_arr, l, inplace=False)
    #    result = metric.assd(result=pred_l, reference=gt_l)
    result = [metric.assd(p[0], p[1]) for p in pairs]
    time = perf_counter() - start
    time_medpy.append(time)
#
# mine is faster, speedup my_assd even more?
# TODO try this pooling with my vertebra segmentation, make all pairs for dice calculation
#
for i in range(iterations):
    start = perf_counter()
    with Pool() as pool:
        pairs = (extract_both(pred_arr, gt_arr, l) for l in labels if l in gt_arr)

        assd_values = pool.starmap(my_assd, pairs)
    # result2 = my_assd(result=pred_arr, reference=gt_arr)
    time = perf_counter() - start
    time_my.append(time)
    # assert result2 == medpy_result

print(np.average(time_medpy))
print(np.average(time_my))

print(result)
print()
print(assd_values)
