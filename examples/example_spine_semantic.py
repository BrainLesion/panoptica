import cProfile

from auxiliary.nifti.io import read_nifti
from auxiliary.turbopath import turbopath

from panoptica import (
    ConnectedComponentsInstanceApproximator,
    NaiveThresholdMatching,
    Panoptic_Evaluator,
    SemanticPair,
)

if __name__ == "__main__":
    directory = turbopath(__file__).parent

    ref_masks = read_nifti(directory + "/spine_seg/semantic_example/sub-0007_mod-T2w_seg-spine_msk.nii.gz")
    pred_masks = read_nifti(directory + "/spine_seg/semantic_example/sub-0007_mod-T2w_seg-spine_msk_new.nii.gz")

    sample = SemanticPair(pred_masks, ref_masks)

    evaluator = Panoptic_Evaluator(
        expected_input=SemanticPair,
        instance_approximator=ConnectedComponentsInstanceApproximator(),
        instance_matcher=NaiveThresholdMatching(),
        iou_threshold=0.5,
    )
    with cProfile.Profile() as pr:
        result, debug_data = evaluator.evaluate(sample)
        print(result)

        pr.dump_stats(directory + "/semantic_example.log")
