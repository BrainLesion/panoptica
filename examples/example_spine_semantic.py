from auxiliary.nifti.io import read_nifti

from panoptica import SemanticPair, Panoptic_Evaluator, ConnectedComponentsInstanceApproximator, CCABackend, NaiveOneToOneMatching

ref_masks = read_nifti("examples/spine_seg/semantic/sub-0007_mod-T2w_seg-spine_msk.nii.gz")
pred_masks = read_nifti("examples/spine_seg/semantic/sub-0007_mod-T2w_seg-spine_msk_new.nii.gz")


sample = SemanticPair(pred_masks, ref_masks)

evaluator = Panoptic_Evaluator(
    expected_input=SemanticPair,
    instance_approximator=ConnectedComponentsInstanceApproximator(cca_backend=CCABackend.cc3d),
    instance_matcher=NaiveOneToOneMatching(),
    iou_threshold=0.5,
)

result, debug_data = evaluator.evaluate(sample)
print(result)
