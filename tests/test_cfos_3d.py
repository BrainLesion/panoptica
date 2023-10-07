from auxiliary.nifti.io import read_nifti
from auxiliary.turbopath import turbopath

from panoptica.panoptic_quality import panoptic_quality


pred_masks = read_nifti(
    input_nifti_path=turbopath(
        "/home/florian/flow/cfos_analysis/data/ablation/2021-11-25_23-50-56_2021-10-25_19-38-31_tr_dice_bce_11/patchvolume_695_2.nii.gz"
    )
)
ref_masks = read_nifti(
    input_nifti_path=turbopath(
        "/home/florian/flow/cfos_analysis/data/reference/patchvolume_695_2/patchvolume_695_2_binary.nii.gz",
    )
)

pq, sq, rq, tp, fp, fn = panoptic_quality(
    ref_mask=ref_masks,
    pred_mask=pred_masks,
    iou_threshold=0.5,
    modus="cc",
)

print("Panoptic Quality (PQ):", pq)
print("Segmentation Quality (SQ):", sq)
print("Recognition Quality (RQ):", rq)
print("True Positives (tp):", tp)
print("False Positives (fp):", fp)
print("False Negatives (fn):", fn)
