# Call 'python -m unittest' on this folder
# coverage run -m unittest
# coverage report
# coverage html
import os
import unittest
from panoptica.cli import cli_main
from unit_test_utils import case_multiple_overlapping_instances, case_simple_identical, case_simple_overlap_but_large_discrepancy
from pathlib import Path
import numpy as np

test_npy_pred_file = Path(__file__).parent.joinpath("test-prediction.npy")
test_npy_ref_file = Path(__file__).parent.joinpath("test-reference.npy")


class Test_CLI(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"
        return super().setUp()

    def test_cli_main(self):
        # Test with valid inputs
        pred_arr, ref_arr = case_multiple_overlapping_instances()

        np.save(test_npy_pred_file, pred_arr)
        np.save(test_npy_ref_file, ref_arr)

        cli_main(
            reference=str(test_npy_ref_file),
            prediction=str(test_npy_pred_file),
            config="panoptica/configs/panoptica_evaluator_unmatched_instance.yaml",
        )
