import cProfile

from auxiliary.nifti.io import read_nifti
from auxiliary.turbopath import turbopath

from panoptica import UnmatchedInstancePair, Panoptica_Evaluator, NaiveThresholdMatching
from panoptica.metrics import Metric
from panoptica.utils.segmentation_class import LabelGroup, SegmentationClassGroups

directory = turbopath(__file__).parent

ref_masks = read_nifti(directory + "/spine_seg/matched_instance/ref.nii.gz")
pred_masks = read_nifti(directory + "/spine_seg/matched_instance/pred.nii.gz")

sample = UnmatchedInstancePair(prediction_arr=pred_masks, reference_arr=ref_masks)

# LabelGroup._register_permanently()

evaluator = Panoptica_Evaluator(
    expected_input=UnmatchedInstancePair,
    eval_metrics=[Metric.DSC, Metric.IOU],
    instance_matcher=NaiveThresholdMatching(),
    segmentation_class_groups=SegmentationClassGroups.load_from_config_name("SegmentationClassGroups_example_unmatchedinstancepair"),
    decision_metric=Metric.DSC,
    decision_threshold=0.5,
    log_times=True,
)


with cProfile.Profile() as pr:
    if __name__ == "__main__":
        results = evaluator.evaluate(sample, verbose=False)
        for groupname, (result, debug) in results.items():
            print()
            print("### Group", groupname)
            print(result)

    pr.dump_stats(directory + "/instance_example.log")
