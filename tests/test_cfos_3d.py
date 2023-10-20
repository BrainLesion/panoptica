from auxiliary.nifti.io import read_nifti
from panoptica.panoptica_evaluation import panoptica_evaluation


pred_masks = read_nifti(
    input_nifti_path="/home/florian/flow/cfos_analysis/data/ablation/2021-11-25_23-50-56_2021-10-25_19-38-31_tr_dice_bce_11/patchvolume_695_2.nii.gz"
)
ref_masks = read_nifti(
    input_nifti_path="/home/florian/flow/cfos_analysis/data/reference/patchvolume_695_2/patchvolume_695_2_binary.nii.gz",
)

# Call panoptica_quality to obtain the result
result = panoptica_evaluation(
    ref_mask=ref_masks,
    pred_mask=pred_masks,
    iou_threshold=0.5,
    modus="cc",
)

# Print the metrics
print("Panoptic Quality (PQ):", result.pq)
print("Segmentation Quality (SQ):", result.sq)
print("Recognition Quality (RQ):", result.rq)
print("True Positives (tp):", result.tp)
print("False Positives (fp):", result.fp)
print("False Negatives (fn):", result.fn)
print("instance_dice", result.i_dice)
print("number of instances in prediction:", result.num_pred_instances)
print("number of instances in reference:", result.num_ref_instances)
