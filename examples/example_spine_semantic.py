import cProfile

from auxiliary.nifti.io import read_nifti
from auxiliary.turbopath import turbopath

from panoptica import (
    ConnectedComponentsInstanceApproximator,
    NaiveThresholdMatching,
    Panoptica_Evaluator,
    InputType,
)

directory = turbopath(__file__).parent

reference_mask = read_nifti(directory + "/spine_seg/semantic/ref.nii.gz")
prediction_mask = read_nifti(directory + "/spine_seg/semantic/pred.nii.gz")


evaluator = Panoptica_Evaluator(
    expected_input=InputType.SEMANTIC,
    instance_approximator=ConnectedComponentsInstanceApproximator(),
    instance_matcher=NaiveThresholdMatching(),
    verbose=True,
)

with cProfile.Profile() as pr:
    if __name__ == "__main__":
        result, debug_data = evaluator.evaluate(prediction_mask, reference_mask)["ungrouped"]
        print(result)

        pr.dump_stats(directory + "/semantic_example.log")
