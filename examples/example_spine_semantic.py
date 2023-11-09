from auxiliary.nifti.io import read_nifti

from panoptica import CCABackend, SemanticSegmentationEvaluator

ref_masks = read_nifti(
    "examples/spine_seg/semantic_example/sub-0007_mod-T2w_seg-spine_msk.nii.gz"
)
pred_masks = read_nifti(
    "examples/spine_seg/semantic_example/sub-0007_mod-T2w_seg-spine_msk_new.nii.gz"
)


eva = SemanticSegmentationEvaluator(cca_backend=CCABackend.cc3d)
res = eva.evaluate(
    reference_mask=ref_masks,
    prediction_mask=pred_masks,
    iou_threshold=0.5,
)

print(res)
