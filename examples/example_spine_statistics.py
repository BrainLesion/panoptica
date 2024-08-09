from auxiliary.nifti.io import read_nifti
from auxiliary.turbopath import turbopath

from panoptica import Panoptica_Evaluator, Panoptica_Aggregator

directory = turbopath(__file__).parent

reference_mask = read_nifti(directory + "/spine_seg/matched_instance/ref.nii.gz")
prediction_mask = read_nifti(directory + "/spine_seg/matched_instance/pred.nii.gz")

evaluator = Panoptica_Aggregator(
    Panoptica_Evaluator.load_from_config_name("panoptica_evaluator_unmatched_instance"),
)


if __name__ == "__main__":
    results = evaluator.evaluate(prediction_mask, reference_mask, verbose=False)
