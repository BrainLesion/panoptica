from auxiliary.nifti.io import read_nifti

from panoptica import (
    SemanticPair,
    Panoptic_Evaluator,
    ConnectedComponentsInstanceApproximator,
    CCABackend,
    NaiveThresholdMatching,
)

pred_masks = read_nifti(
    input_nifti_path="/home/florian/flow/cfos_analysis/data/ablation/2021-11-25_23-50-56_2021-10-25_19-38-31_tr_dice_bce_11/patchvolume_695_2.nii.gz"
)
ref_masks = read_nifti(
    input_nifti_path="/home/florian/flow/cfos_analysis/data/reference/patchvolume_695_2/patchvolume_695_2_binary.nii.gz",
)

sample = SemanticPair(pred_masks, ref_masks)

evaluator = Panoptic_Evaluator(
    expected_input=SemanticPair,
    instance_approximator=ConnectedComponentsInstanceApproximator(),
    instance_matcher=NaiveThresholdMatching(),
    match_threshold=0.5,
)

result, debug_data = evaluator.evaluate(sample)
print(result)
