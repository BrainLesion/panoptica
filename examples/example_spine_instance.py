import cProfile

from auxiliary.nifti.io import read_nifti
from pathlib import Path

from panoptica import Panoptica_Evaluator, InputType
from panoptica.metrics import Metric
from panoptica.utils.segmentation_class import LabelGroup, SegmentationClassGroups

directory = str(Path(__file__).absolute().parent)

reference_mask = read_nifti(directory + "/spine_seg/matched_instance/ref.nii.gz")
prediction_mask = read_nifti(directory + "/spine_seg/matched_instance/pred.nii.gz")

evaluator = Panoptica_Evaluator(
    expected_input=InputType.MATCHED_INSTANCE,
    instance_metrics=[Metric.DSC, Metric.IOU],
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
    log_times=True,
)


def main():
    with cProfile.Profile() as pr:
        results = evaluator.evaluate(prediction_mask, reference_mask, verbose=False)
        for groupname, result in results.items():
            print()
            print("### Group", groupname)
            print(result)

    pr.dump_stats(directory + "/instance_example.log")
    return results


if __name__ == "__main__":
    main()
