# Call 'python -m unittest' on this folder
# coverage run -m unittest
# coverage report
# coverage html
import os
import unittest
import numpy as np
from pathlib import Path
import SimpleITK as sitk
from unittest import mock
from unit_test_utils import patch_property
from unittest.mock import MagicMock, PropertyMock
import nibabel as nib

from panoptica.utils.input_check_and_conversion.sanity_checker import sanity_checker, INPUTDTYPE, _InputDataTypeChecker

test_npy_file = Path(__file__).parent.joinpath("test.npy")
test_nii_file = Path(__file__).parent.joinpath("test.nii.gz")


class Test_Input_Sanity_Checker_Numpy(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"
        return super().setUp()

    def test_sanity_checker(self):
        # Create two identical numpy arrays
        arr1 = np.random.rand(10, 10)
        arr2 = np.copy(arr1)

        with self.assertWarns(UserWarning):
            # Test the sanity checker
            prediction_arr, reference_arr = sanity_checker(arr1, arr2)
            # self.assertTrue(result)
        with self.assertWarns(UserWarning):
            # Modify one array and test again
            arr2[0, 0] += 1e-6
            prediction_arr, reference_arr = sanity_checker(arr1, arr2)

        arr1 = np.zeros((10, 10), dtype=np.uint8)
        arr2 = np.zeros((10, 10), dtype=np.uint8)
        arr2[0, 0] = 1
        # Test the sanity checker
        prediction_arr, reference_arr = sanity_checker(arr1, arr2)

    def test_sanity_checker_shapemismatch(self):
        # Create two identical numpy arrays
        arr1 = np.random.rand(10, 10)
        arr2 = np.random.rand(11, 10)

        # Test the sanity checker
        with self.assertRaises(ValueError):
            prediction_arr, reference_arr = sanity_checker(arr1, arr2)

        with self.assertRaises(ValueError):
            prediction_arr, reference_arr = sanity_checker(arr2, arr1)

    def test_sanity_checker_as_file(self):
        # Create two identical numpy arrays
        arr1 = np.random.rand(10, 10)
        np.save(test_npy_file, arr1)

        # Test the sanity checker with Path
        prediction_arr, reference_arr = sanity_checker(test_npy_file, test_npy_file)
        # Test the sanity checker with str
        prediction_arr, reference_arr = sanity_checker(str(test_npy_file), str(test_npy_file))
        os.remove(test_npy_file)


class Test_Input_Sanity_Checker_Sitk(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"
        return super().setUp()

    def test_sanity_checker(self):
        # Create two identical numpy arrays
        arr1 = np.random.rand(10, 10)
        arr2 = np.copy(arr1)

        with self.assertWarns(UserWarning):
            # Test the sanity checker
            prediction_arr, reference_arr = sanity_checker(sitk.GetImageFromArray(arr1), sitk.GetImageFromArray(arr2))

        arr1 = np.zeros((10, 10), dtype=np.uint8)
        arr2 = np.zeros((10, 10), dtype=np.uint8)
        arr2[0, 0] = 1
        # Test the sanity checker
        prediction_arr, reference_arr = sanity_checker(sitk.GetImageFromArray(arr1), sitk.GetImageFromArray(arr2))

    def test_sanity_checker_shapemismatch(self):
        # Create two identical numpy arrays
        arr1 = np.random.rand(10, 10)
        arr2 = np.random.rand(11, 10)

        # Test the sanity checker
        with self.assertRaises(ValueError):
            prediction_arr, reference_arr = sanity_checker(sitk.GetImageFromArray(arr1), sitk.GetImageFromArray(arr2))

        with self.assertRaises(ValueError):
            prediction_arr, reference_arr = sanity_checker(sitk.GetImageFromArray(arr1), sitk.GetImageFromArray(arr2))

    def test_sanity_checker_as_file(self):
        # Create two identical numpy arrays
        arr1 = np.zeros((10, 10), dtype=np.uint8)

        sitkimage = sitk.GetImageFromArray(arr1)
        sitk.WriteImage(sitkimage, test_nii_file)

        # Test the sanity checker with Path
        prediction_arr, reference_arr = sanity_checker(test_nii_file, test_nii_file)
        # Test the sanity checker with str
        prediction_arr, reference_arr = sanity_checker(str(test_nii_file), str(test_nii_file))
        os.remove(test_nii_file)

    def test_sanity_checker_without_package(self):
        patchers = [
            patch_property(
                "panoptica.utils.input_check_and_conversion.sanity_checker._InputDataTypeChecker.missing_packages", ["SimpleITK"]
            ),
            patch_property("panoptica.utils.input_check_and_conversion.sanity_checker._InputDataTypeChecker.requirements_fulfilled", False),
        ]
        # Create two identical numpy arrays
        arr1 = np.zeros((10, 10), dtype=np.uint8)

        print(INPUTDTYPE.SITK.value.missing_packages)
        print(INPUTDTYPE.SITK.value.requirements_fulfilled)

        sitkimage = sitk.GetImageFromArray(arr1)
        sitk.WriteImage(sitkimage, test_nii_file)

        with self.assertRaises(ImportError):
            # Test the sanity checker with Path
            prediction_arr, reference_arr = sanity_checker(test_nii_file, test_nii_file)
        with self.assertRaises(ImportError):
            # Test the sanity checker with str
            prediction_arr, reference_arr = sanity_checker(str(test_nii_file), str(test_nii_file))
        os.remove(test_nii_file)

        # Stop all patches
        for p in patchers:
            p.stop()


class Test_Input_Sanity_Checker_Nibabel(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"
        return super().setUp()

    def test_sanity_checker(self):
        # Create two identical numpy arrays
        arr1 = np.random.rand(10, 10)
        arr2 = np.copy(arr1)

        with self.assertWarns(UserWarning):
            # Test the sanity checker
            prediction_arr, reference_arr = sanity_checker(nib.Nifti1Image(arr1, affine=np.eye(4)), nib.Nifti1Image(arr2, affine=np.eye(4)))

        arr1 = np.zeros((10, 10), dtype=np.uint8)
        arr2 = np.zeros((10, 10), dtype=np.uint8)
        arr2[0, 0] = 1
        # Test the sanity checker
        prediction_arr, reference_arr = sanity_checker(nib.Nifti1Image(arr1, affine=np.eye(4)), nib.Nifti1Image(arr2, affine=np.eye(4)))

    def test_sanity_checker_shapemismatch(self):
        # Create two identical numpy arrays
        arr1 = np.random.rand(10, 10)
        arr2 = np.random.rand(11, 10)

        # Test the sanity checker
        with self.assertRaises(ValueError):
            prediction_arr, reference_arr = sanity_checker(sitk.GetImageFromArray(arr1), sitk.GetImageFromArray(arr2))

        with self.assertRaises(ValueError):
            prediction_arr, reference_arr = sanity_checker(sitk.GetImageFromArray(arr1), sitk.GetImageFromArray(arr2))

    def test_sanity_checker_as_file(self):
        # Create two identical numpy arrays
        arr1 = np.zeros((10, 10), dtype=np.uint8)

        sitkimage = sitk.GetImageFromArray(arr1)
        sitk.WriteImage(sitkimage, test_nii_file)

        # Test the sanity checker with Path
        prediction_arr, reference_arr = sanity_checker(test_nii_file, test_nii_file)
        # Test the sanity checker with str
        prediction_arr, reference_arr = sanity_checker(str(test_nii_file), str(test_nii_file))
        os.remove(test_nii_file)

    def test_sanity_checker_without_package(self):
        patchers = [
            patch_property(
                "panoptica.utils.input_check_and_conversion.sanity_checker._InputDataTypeChecker.missing_packages", ["SimpleITK"]
            ),
            patch_property("panoptica.utils.input_check_and_conversion.sanity_checker._InputDataTypeChecker.requirements_fulfilled", False),
        ]
        # Create two identical numpy arrays
        arr1 = np.zeros((10, 10), dtype=np.uint8)

        print(INPUTDTYPE.SITK.value.missing_packages)
        print(INPUTDTYPE.SITK.value.requirements_fulfilled)

        sitkimage = sitk.GetImageFromArray(arr1)
        sitk.WriteImage(sitkimage, test_nii_file)

        with self.assertRaises(ImportError):
            # Test the sanity checker with Path
            prediction_arr, reference_arr = sanity_checker(test_nii_file, test_nii_file)
        with self.assertRaises(ImportError):
            # Test the sanity checker with str
            prediction_arr, reference_arr = sanity_checker(str(test_nii_file), str(test_nii_file))
        os.remove(test_nii_file)

        # Stop all patches
        for p in patchers:
            p.stop()
