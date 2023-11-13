from auxiliary.nifti.io import read_nifti
import cProfile


from panoptica import (
    SemanticPair,
    MatchedInstancePair,
    Panoptic_Evaluator,
    ConnectedComponentsInstanceApproximator,
    CCABackend,
    NaiveOneToOneMatching,
)

ref_masks = read_nifti("repo/examples/spine_seg/semantic_example/sub-0007_mod-T2w_seg-spine_msk.nii.gz")
pred_masks = read_nifti("repo/examples/spine_seg/semantic_example/sub-0007_mod-T2w_seg-spine_msk_new.nii.gz")


sample = SemanticPair(pred_masks, ref_masks)

evaluator = Panoptic_Evaluator(
    expected_input=SemanticPair,
    instance_approximator=ConnectedComponentsInstanceApproximator(cca_backend=CCABackend.cc3d),
    instance_matcher=NaiveOneToOneMatching(),
    iou_threshold=0.5,
)

from BIDS.logger.log_file import get_time, format_time_short

timestamp = format_time_short(get_time())
with cProfile.Profile() as pr:
    result, debug_data = evaluator.evaluate(sample)
    print(result)
pr.dump_stats(f"repo/examples/cprofile_{timestamp}_log.log")
