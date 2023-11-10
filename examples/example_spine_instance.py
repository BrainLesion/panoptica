from auxiliary.nifti.io import read_nifti

from panoptica import (
    UnmatchedInstancePair,
    Panoptic_Evaluator,
    ConnectedComponentsInstanceApproximator,
    CCABackend,
    NaiveOneToOneMatching,
)

ref_masks = read_nifti("examples/spine_seg/instance/sub-0007_mod-T2w_seg-vert_msk.nii.gz")
pred_masks = read_nifti("examples/spine_seg/instance/sub-0007_mod-T2w_seg-vert_msk_new.nii.gz")


sample = UnmatchedInstancePair(pred_masks, ref_masks)

evaluator = Panoptic_Evaluator(
    expected_input=UnmatchedInstancePair,
    instance_approximator=None,
    instance_matcher=NaiveOneToOneMatching(),
    iou_threshold=0.5,
)

result, debug_data = evaluator.evaluate(sample)
print(result)
