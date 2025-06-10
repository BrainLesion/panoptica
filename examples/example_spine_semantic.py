import cProfile

from auxiliary.io import read_image
from pathlib import Path

from panoptica import (
    ConnectedComponentsInstanceApproximator,
    NaiveThresholdMatching,
    Panoptica_Evaluator,
    InputType,
)

directory = str(Path(__file__).absolute().parent)

reference_mask = read_image(directory + "/spine_seg/semantic/ref.nii.gz")
prediction_mask = read_image(directory + "/spine_seg/semantic/pred.nii.gz")


evaluator = Panoptica_Evaluator(
    expected_input=InputType.SEMANTIC,
    instance_approximator=ConnectedComponentsInstanceApproximator(),
    instance_matcher=NaiveThresholdMatching(),
    verbose=True,
    log_times=True,
)


def main():
    with cProfile.Profile() as pr:
        result = evaluator.evaluate(prediction_mask, reference_mask)["ungrouped"]

        # To print the results, just call print
        print(result)

        intermediate_steps_data = result.intermediate_steps_data
        assert intermediate_steps_data is not None
        # To get the different intermediate arrays, just use the second returned object
        intermediate_steps_data.original_prediction_arr  # Input prediction array, untouched
        intermediate_steps_data.original_reference_arr  # Input reference array, untouched

        intermediate_steps_data.prediction_arr(
            InputType.MATCHED_INSTANCE
        )  # Prediction array after instances have been matched
        intermediate_steps_data.reference_arr(
            InputType.MATCHED_INSTANCE
        )  # Reference array after instances have been matched

    pr.dump_stats(directory + "/semantic_example.log")
    return result, intermediate_steps_data


if __name__ == "__main__":
    main()
