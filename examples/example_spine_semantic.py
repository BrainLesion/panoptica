import cProfile

from auxiliary.nifti.io import read_nifti
from auxiliary.turbopath import turbopath


from panoptica import (
    ConnectedComponentsInstanceApproximator,
    NaiveThresholdMatching,
    Panoptic_Evaluator,
    SemanticPair,
)
from panoptica.metrics import MatchingMetrics
from panoptica.utils import EdgeCaseHandler, EdgeCaseZeroTP, EdgeCaseResult

directory = turbopath(__file__).parent

ref_masks = read_nifti(directory + "/spine_seg/semantic/ref.nii.gz")
pred_masks = read_nifti(directory + "/spine_seg/semantic/pred.nii.gz")

sample = SemanticPair(pred_masks, ref_masks)

evaluator = Panoptic_Evaluator(
    expected_input=SemanticPair,
    instance_approximator=ConnectedComponentsInstanceApproximator(),
    instance_matcher=NaiveThresholdMatching(),
    matching_metric=MatchingMetrics.IOU,
    matching_threshold=0.5,
)
with cProfile.Profile() as pr:
    if __name__ == "__main__":
        result, debug_data = evaluator.evaluate(sample)
        print(result)

        pr.dump_stats(directory + "/semantic_example.log")
