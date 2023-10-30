from auxiliary.nifti.io import read_nifti

from panoptica.instance import InstanceSegmentationEvaluator

ref_masks = read_nifti(
    "examples/spine_seg/instance/sub-0007_mod-T2w_seg-vert_msk.nii.gz"
)
pred_masks = read_nifti(
    "examples/spine_seg/instance/sub-0007_mod-T2w_seg-vert_msk_new.nii.gz"
)


eva = InstanceSegmentationEvaluator()

res = eva.evaluate(
    reference_mask=ref_masks,
    prediction_mask=pred_masks,
    iou_threshold=0.5,
)

print(res)
