from auxiliary.nifti.io import read_nifti

from panoptica import CCABackend, SemanticSegmentationEvaluator

pred_masks = read_nifti(
    input_nifti_path="/home/florian/flow/cfos_analysis/data/ablation/2021-11-25_23-50-56_2021-10-25_19-38-31_tr_dice_bce_11/patchvolume_695_2.nii.gz"
)
ref_masks = read_nifti(
    input_nifti_path="/home/florian/flow/cfos_analysis/data/reference/patchvolume_695_2/patchvolume_695_2_binary.nii.gz",
)

eva = SemanticSegmentationEvaluator(cca_backend=CCABackend.cc3d)
res = eva.evaluate(
    reference_mask=ref_masks,
    prediction_mask=pred_masks,
    iou_threshold=0.5,
)

print(res)
