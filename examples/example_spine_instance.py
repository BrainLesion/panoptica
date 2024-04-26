import cProfile

from auxiliary.nifti.io import read_nifti
from auxiliary.turbopath import turbopath

from panoptica import MatchedInstancePair, Panoptic_Evaluator
from panoptica.metrics import Metric
from panoptica.utils.segmentation_class import LabelGroup, SegmentationClassGroups

directory = turbopath(__file__).parent

ref_masks = read_nifti(directory + "/spine_seg/matched_instance/ref.nii.gz")

pred_masks = read_nifti(directory + "/spine_seg/matched_instance/pred.nii.gz")

sample = MatchedInstancePair(prediction_arr=pred_masks, reference_arr=ref_masks)

import numpy as np

print(np.unique(pred_masks))

evaluator = Panoptic_Evaluator(
    expected_input=MatchedInstancePair,
    eval_metrics=[Metric.DSC, Metric.IOU],
    segmentation_class_groups=SegmentationClassGroups(
        {
            "vertebra": LabelGroup([i for i in range(1, 10)]),
            "ivd": LabelGroup([i for i in range(101, 109)]),
            "sacrum": (26, True),
            "endplate": LabelGroup([i for i in range(201, 209)]),
        }
    ),
    decision_metric=Metric.DSC,
    decision_threshold=0.5,
)


with cProfile.Profile() as pr:
    if __name__ == "__main__":
        results = evaluator.evaluate(sample, verbose=False)
        for groupname, (result, debug) in results.items():
            print()
            print("### Group", groupname)
            print(result)

    pr.dump_stats(directory + "/instance_example.log")
