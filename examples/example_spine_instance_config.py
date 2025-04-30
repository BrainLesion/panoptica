import cProfile

from auxiliary.nifti.io import read_nifti
from pathlib import Path

from panoptica import Panoptica_Evaluator

directory = str(Path(__file__).absolute().parent)

reference_mask = read_nifti(directory + "/spine_seg/matched_instance/ref.nii.gz")
prediction_mask = read_nifti(directory + "/spine_seg/matched_instance/pred.nii.gz")

evaluator = Panoptica_Evaluator.load_from_config(
    "panoptica_evaluator_unmatched_instance"
)


def main():
    with cProfile.Profile() as pr:
        results = evaluator.evaluate(prediction_mask, reference_mask, verbose=False)
        for groupname, result in results.items():
            print()
            print("### Group", groupname)
            print(result)

    pr.dump_stats(directory + "/instance_example.log")
    return results


if __name__ == "__main__":
    main()
