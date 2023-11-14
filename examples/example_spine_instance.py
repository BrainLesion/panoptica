import cProfile

from auxiliary.nifti.io import read_nifti
from auxiliary.turbopath import turbopath

from panoptica import MatchedInstancePair, NaiveThresholdMatching, Panoptic_Evaluator

directory = turbopath(__file__).parent

ref_masks = read_nifti(directory + "/spine_seg/instance_example/sub-0007_mod-T2w_seg-vert_msk.nii.gz")

pred_masks = read_nifti(directory + "/spine_seg/instance_example/sub-0007_mod-T2w_seg-vert_msk_new.nii.gz")

sample = MatchedInstancePair(prediction_arr=pred_masks, reference_arr=ref_masks)


evaluator = Panoptic_Evaluator(
    expected_input=MatchedInstancePair,
    instance_approximator=None,
    instance_matcher=NaiveThresholdMatching(),
    iou_threshold=0.5,
)
with cProfile.Profile() as pr:
    result, debug_data = evaluator.evaluate(sample)
    print(result)

pr.dump_stats(directory + "/instance_example.log")
