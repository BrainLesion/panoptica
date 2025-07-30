import numpy as np
from panoptica import Panoptica_Evaluator
from panoptica.instance_approximator import ConnectedComponentsInstanceApproximator
from panoptica.instance_matcher import MaxBipartiteMatching
from panoptica.utils.processing_pair import InputType
from panoptica.utils.segmentation_class import SegmentationClassGroups
from panoptica.utils.label_group import LabelPartGroup, LabelMergeGroup

import nibabel as nib

ref_mask = nib.load("/home/localssk23/backup/soumya/BraTS-2023-Metrics/dataset/ASNR-MICCAI-BraTS2023-MET-Challenge-TestingData/BraTS-MET-00001-000/BraTS-MET-00001-000-seg.nii.gz").get_fdata()
pred_mask = nib.load("/home/localssk23/backup/soumya/BraTS-2023-Metrics/dataset/BratSMets_PredictedSegs/PredictedSegs/NVAUTO/BraTS-MET-00001-000.nii.gz").get_fdata()

groups = {
    "NETC": (1, False),
    "ET": (3, False),
    "SNFH": (2, False),
    "TC": LabelPartGroup([1], [3], False),
    "WT": LabelMergeGroup([1, 2, 3], False),
}
class_groups = SegmentationClassGroups(groups)

pred_mask = pred_mask.astype(np.int32)
ref_mask = ref_mask.astype(np.int32)

evaluator = Panoptica_Evaluator(
expected_input=InputType.SEMANTIC,
instance_approximator=ConnectedComponentsInstanceApproximator(),
instance_matcher=MaxBipartiteMatching(),
segmentation_class_groups=class_groups,
)

results = evaluator.evaluate(pred_mask, ref_mask, verbose=False)

print(results.items())

for class_name, result in results.items():
    if class_name == "tc" or class_name == "wt":

            print(f"\n--- {class_name} ---")
            result_dict = result.to_dict()
            # Filter the dictionary to show only the most important metrics
            important_metrics = [
                "tp",
                "fp",
                "fn",
                "sq",
                "rq",
                "pq",
                "sq_dsc",
                "global_bin_dsc",
            ]
            filtered_dict = {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in result_dict.items()
                if k in important_metrics
            }
            print(filtered_dict)

from pathlib import Path
from tqdm import tqdm

teams = ["NVAUTO", "S_Y", "blackbean"]
dataset_path = "/home/localssk23/backup/soumya/BraTS-2023-Metrics/dataset"
test_path = Path(dataset_path) / "ASNR-MICCAI-BraTS2023-MET-Challenge-TestingData"
pred_path = Path(dataset_path) / "BratSMets_PredictedSegs" / "PredictedSegs"


case_dirs = sorted(test_path.glob("BraTS-MET-*-000"))
num_cases = 2

if num_cases is not None:
    case_dirs = case_dirs[:num_cases]

for case_dir in tqdm(case_dirs):
    case_name = case_dir.name
    gt_file = case_dir / f"{case_name}-seg.nii.gz"
    
    for team in teams:
        pred_file = pred_path / team / f"{case_name}.nii.gz"

        result = evaluator.evaluate(pred_file, gt_file, verbose=False)

        for class_name, result in result.items():
            if class_name == "tc" or class_name == "wt":
                print(f"\n--- {class_name} ---")
                result_dict = result.to_dict()
                # Filter the dictionary to show only the most important metrics
                important_metrics = [
                    "tp",
                    "fp",
                    "fn",
                    "sq",
                    "rq",
                    "pq",
                    "sq_dsc",
                    "global_bin_dsc",
                ]
                filtered_dict = {
                    k: round(v, 4) if isinstance(v, float) else v
                    for k, v in result_dict.items()
                    if k in important_metrics
                }
                print(filtered_dict)