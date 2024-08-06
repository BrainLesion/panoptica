import cProfile

from auxiliary.nifti.io import read_nifti
from auxiliary.turbopath import turbopath

from panoptica import Panoptica_Evaluator

directory = turbopath(__file__).parent

reference_mask = read_nifti(directory + "/spine_seg/matched_instance/ref.nii.gz")
prediction_mask = read_nifti(directory + "/spine_seg/matched_instance/pred.nii.gz")

evaluator = Panoptica_Evaluator.load_from_config_name(
    "panoptica_evaluator_unmatched_instance"
)


with cProfile.Profile() as pr:
    if __name__ == "__main__":
        results = evaluator.evaluate(prediction_mask, reference_mask, verbose=False)
        for groupname, (result, debug) in results.items():
            print()
            print("### Group", groupname)
            print(result)

    pr.dump_stats(directory + "/instance_example.log")
