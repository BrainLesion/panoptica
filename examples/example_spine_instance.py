from auxiliary.nifti.io import read_nifti
from auxiliary.turbopath import turbopath

from panoptica import MatchedInstancePair, NaiveOneToOneMatching, Panoptic_Evaluator


directory = turbopath(__file__).parent

ref_masks = read_nifti(directory + "/spine_seg/instance_example/sub-0007_mod-T2w_seg-vert_msk.nii.gz")

pred_masks = read_nifti(directory + "/spine_seg/instance_example/sub-0007_mod-T2w_seg-vert_msk_new.nii.gz")

sample = MatchedInstancePair(prediction_arr=pred_masks, reference_arr=ref_masks)

evaluator = Panoptic_Evaluator(
    expected_input=MatchedInstancePair,
    instance_approximator=None,
    instance_matcher=NaiveOneToOneMatching(),
    iou_threshold=0.5,
)

result, debug_data = evaluator.evaluate(sample)
print(result)
