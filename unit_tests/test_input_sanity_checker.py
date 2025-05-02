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
import nibabel as nib
import torch

from panoptica.utils.input_check_and_conversion.sanity_checker import (
    sanity_check_and_convert_to_array,
    INPUTDTYPE,
    _InputDataTypeChecker,
    print_available_package_to_input_handlers,
)

test_npy_file = Path(__file__).parent.joinpath("test.npy")
test_torch_file = Path(__file__).parent.joinpath("test.pt")
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
            (prediction_arr, reference_arr), checker = (
                sanity_check_and_convert_to_array(arr1, arr2)
            )
            self.assertEqual(checker, INPUTDTYPE.NUMPY)
            # self.assertTrue(result)
        with self.assertWarns(UserWarning):
            # Modify one array and test again
            arr2[0, 0] += 1e-6
            (prediction_arr, reference_arr), checker = (
                sanity_check_and_convert_to_array(arr1, arr2)
            )
            self.assertEqual(checker, INPUTDTYPE.NUMPY)

        arr1 = np.zeros((10, 10), dtype=np.uint8)
        arr2 = np.zeros((10, 10), dtype=np.uint8)
        arr2[0, 0] = 1
        # Test the sanity checker
        (prediction_arr, reference_arr), checker = sanity_check_and_convert_to_array(
            arr1, arr2
        )
        self.assertEqual(checker, INPUTDTYPE.NUMPY)

    def test_sanity_checker_shapemismatch(self):
        # Create two identical numpy arrays
        arr1 = np.random.rand(10, 10)
        arr2 = np.random.rand(11, 10)

        # Test the sanity checker
        with self.assertRaises(ValueError):
            (prediction_arr, reference_arr), checker = (
                sanity_check_and_convert_to_array(arr1, arr2)
            )
            self.assertEqual(checker, INPUTDTYPE.NUMPY)

        with self.assertRaises(ValueError):
            (prediction_arr, reference_arr), checker = (
                sanity_check_and_convert_to_array(arr2, arr1)
            )
            self.assertEqual(checker, INPUTDTYPE.NUMPY)

    def test_sanity_checker_as_file(self):
        # Create two identical numpy arrays
        arr1 = np.random.rand(10, 10)
        np.save(test_npy_file, arr1)

        # Test the sanity checker with Path
        (prediction_arr, reference_arr), checker = sanity_check_and_convert_to_array(
            test_npy_file, test_npy_file
        )
        self.assertEqual(checker, INPUTDTYPE.NUMPY)
        # Test the sanity checker with str
        (prediction_arr, reference_arr), checker = sanity_check_and_convert_to_array(
            str(test_npy_file), str(test_npy_file)
        )
        self.assertEqual(checker, INPUTDTYPE.NUMPY)
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
            (prediction_arr, reference_arr), checker = (
                sanity_check_and_convert_to_array(
                    sitk.GetImageFromArray(arr1), sitk.GetImageFromArray(arr2)
                )
            )
            self.assertEqual(checker, INPUTDTYPE.SITK)

        arr1 = np.zeros((10, 10), dtype=np.uint8)
        arr2 = np.zeros((10, 10), dtype=np.uint8)
        arr2[0, 0] = 1
        # Test the sanity checker
        (prediction_arr, reference_arr), checker = sanity_check_and_convert_to_array(
            sitk.GetImageFromArray(arr1), sitk.GetImageFromArray(arr2)
        )
        self.assertEqual(checker, INPUTDTYPE.SITK)

    def test_sanity_checker_shapemismatch(self):
        # Create two identical numpy arrays
        arr1 = np.random.rand(10, 10)
        arr2 = np.random.rand(11, 10)

        # Test the sanity checker
        with self.assertRaises(ValueError):
            (prediction_arr, reference_arr), checker = (
                sanity_check_and_convert_to_array(
                    sitk.GetImageFromArray(arr1), sitk.GetImageFromArray(arr2)
                )
            )
            self.assertEqual(checker, INPUTDTYPE.SITK)

        with self.assertRaises(ValueError):
            (prediction_arr, reference_arr), checker = (
                sanity_check_and_convert_to_array(
                    sitk.GetImageFromArray(arr2), sitk.GetImageFromArray(arr1)
                )
            )
            self.assertEqual(checker, INPUTDTYPE.SITK)

    def test_sanity_checker_as_file(self):
        # Create two identical numpy arrays
        arr1 = np.zeros((10, 10), dtype=np.uint8)

        sitkimage = sitk.GetImageFromArray(arr1)
        sitk.WriteImage(sitkimage, test_nii_file)

        # Test the sanity checker with Path
        (prediction_arr, reference_arr), checker = sanity_check_and_convert_to_array(
            test_nii_file, test_nii_file
        )
        self.assertEqual(checker, INPUTDTYPE.SITK)
        # Test the sanity checker with str
        (prediction_arr, reference_arr), checker = sanity_check_and_convert_to_array(
            str(test_nii_file), str(test_nii_file)
        )
        self.assertEqual(checker, INPUTDTYPE.SITK)
        os.remove(test_nii_file)

    @mock.patch.object(
        INPUTDTYPE.SITK.value, "are_requirements_fulfilled", return_value=False
    )
    @mock.patch.object(
        INPUTDTYPE.NIBABEL.value, "are_requirements_fulfilled", return_value=False
    )
    def test_sanity_checker_without_package(self, *args):
        # Create two identical numpy arrays
        arr1 = np.zeros((10, 10), dtype=np.uint8)

        print(INPUTDTYPE.SITK.value.missing_packages)
        print(INPUTDTYPE.SITK.value.are_requirements_fulfilled())

        sitkimage = sitk.GetImageFromArray(arr1)
        sitk.WriteImage(sitkimage, test_nii_file)

        with self.assertRaises(ImportError):
            # Test the sanity checker with Path
            (prediction_arr, reference_arr), checker = (
                sanity_check_and_convert_to_array(test_nii_file, test_nii_file)
            )
            print(checker)
        with self.assertRaises(ImportError):
            # Test the sanity checker with str
            (prediction_arr, reference_arr), checker = (
                sanity_check_and_convert_to_array(
                    str(test_nii_file), str(test_nii_file)
                )
            )
            print(checker)
        os.remove(test_nii_file)


class Test_Input_Sanity_Checker_Nibabel(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"
        return super().setUp()

    def test_sanity_checker(self):
        ## Manually override the *property method resolution* at the instance level
        arr1 = np.random.rand(10, 10)
        arr2 = np.copy(arr1)

        print_available_package_to_input_handlers()

        with self.assertWarns(UserWarning):
            # Test the sanity checker
            (prediction_arr, reference_arr), checker = (
                sanity_check_and_convert_to_array(
                    nib.Nifti1Image(arr1, affine=np.eye(4)),
                    nib.Nifti1Image(arr2, affine=np.eye(4)),
                )
            )
            self.assertEqual(checker, INPUTDTYPE.NIBABEL)

        arr1 = np.zeros((10, 10), dtype=np.uint8)
        arr2 = np.zeros((10, 10), dtype=np.uint8)
        arr2[0, 0] = 1
        # Test the sanity checker
        (prediction_arr, reference_arr), checker = sanity_check_and_convert_to_array(
            nib.Nifti1Image(arr1, affine=np.eye(4)),
            nib.Nifti1Image(arr2, affine=np.eye(4)),
        )
        self.assertEqual(checker, INPUTDTYPE.NIBABEL)

    def test_sanity_checker_shapemismatch(self):
        # Create two identical numpy arrays
        arr1 = np.random.rand(10, 10)
        arr2 = np.random.rand(11, 10)

        # Test the sanity checker
        with self.assertRaises(ValueError):
            (prediction_arr, reference_arr), checker = (
                sanity_check_and_convert_to_array(
                    nib.Nifti1Image(arr1, affine=np.eye(4)),
                    nib.Nifti1Image(arr2, affine=np.eye(4)),
                )
            )
            self.assertEqual(checker, INPUTDTYPE.NIBABEL)

        with self.assertRaises(ValueError):
            (prediction_arr, reference_arr), checker = (
                sanity_check_and_convert_to_array(
                    nib.Nifti1Image(arr2, affine=np.eye(4)),
                    nib.Nifti1Image(arr1, affine=np.eye(4)),
                )
            )
            self.assertEqual(checker, INPUTDTYPE.NIBABEL)

    def test_sanity_checker_as_file(self):
        # Create two identical numpy arrays
        arr1 = np.zeros((10, 10), dtype=np.uint8)

        nib_image = nib.Nifti1Image(arr1, affine=np.eye(4))
        nib_image.to_filename(test_nii_file)

        # Test the sanity checker with Path
        (prediction_arr, reference_arr), checker = sanity_check_and_convert_to_array(
            test_nii_file, test_nii_file
        )
        self.assertEqual(checker, INPUTDTYPE.SITK)
        # Test the sanity checker with str
        (prediction_arr, reference_arr), checker = sanity_check_and_convert_to_array(
            str(test_nii_file), str(test_nii_file)
        )
        self.assertEqual(checker, INPUTDTYPE.SITK)
        os.remove(test_nii_file)

    @mock.patch.object(
        INPUTDTYPE.SITK.value, "are_requirements_fulfilled", return_value=False
    )
    def test_sanity_checker_as_file_wo_sitk(self, *args):
        # Create two identical numpy arrays
        arr1 = np.zeros((10, 10), dtype=np.uint8)

        nib_image = nib.Nifti1Image(arr1, affine=np.eye(4))
        nib_image.to_filename(test_nii_file)

        # Test the sanity checker with Path
        (prediction_arr, reference_arr), checker = sanity_check_and_convert_to_array(
            test_nii_file, test_nii_file
        )
        self.assertEqual(checker, INPUTDTYPE.NIBABEL)
        # Test the sanity checker with str
        (prediction_arr, reference_arr), checker = sanity_check_and_convert_to_array(
            str(test_nii_file), str(test_nii_file)
        )
        self.assertEqual(checker, INPUTDTYPE.NIBABEL)
        os.remove(test_nii_file)

    @mock.patch.object(
        INPUTDTYPE.SITK.value, "are_requirements_fulfilled", return_value=False
    )
    @mock.patch.object(
        INPUTDTYPE.NIBABEL.value, "are_requirements_fulfilled", return_value=False
    )
    def test_sanity_checker_without_package(self, *args):
        # Create two identical numpy arrays
        arr1 = np.zeros((10, 10), dtype=np.uint8)

        print(INPUTDTYPE.NIBABEL.value.missing_packages)
        print(INPUTDTYPE.NIBABEL.value.are_requirements_fulfilled())

        nib_image = nib.Nifti1Image(arr1, affine=np.eye(4))
        nib_image.to_filename(test_nii_file)

        with self.assertRaises(ImportError):
            # Test the sanity checker with Path
            (prediction_arr, reference_arr), checker = (
                sanity_check_and_convert_to_array(test_nii_file, test_nii_file)
            )
        with self.assertRaises(ImportError):
            # Test the sanity checker with str
            (prediction_arr, reference_arr), checker = (
                sanity_check_and_convert_to_array(
                    str(test_nii_file), str(test_nii_file)
                )
            )
        os.remove(test_nii_file)


class Test_Input_Sanity_Checker_Torch(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"
        return super().setUp()

    def test_sanity_checker(self):
        # Create two identical numpy arrays
        arr1 = torch.rand(10, 10)
        arr2 = arr1.clone()

        with self.assertWarns(UserWarning):
            # Test the sanity checker
            (prediction_arr, reference_arr), checker = (
                sanity_check_and_convert_to_array(arr1, arr2)
            )
            self.assertEqual(checker, INPUTDTYPE.TORCH)
            # self.assertTrue(result)
        with self.assertWarns(UserWarning):
            # Modify one array and test again
            arr2[0, 0] += 1e-6
            (prediction_arr, reference_arr), checker = (
                sanity_check_and_convert_to_array(arr1, arr2)
            )
            self.assertEqual(checker, INPUTDTYPE.TORCH)

        arr1 = torch.zeros((10, 10), dtype=torch.uint8)
        arr2 = torch.zeros((10, 10), dtype=torch.uint8)
        arr2[0, 0] = 1
        # Test the sanity checker
        (prediction_arr, reference_arr), checker = sanity_check_and_convert_to_array(
            arr1, arr2
        )
        self.assertEqual(checker, INPUTDTYPE.TORCH)

    def test_sanity_checker_shapemismatch(self):
        # Create two identical numpy arrays
        arr1 = torch.rand(10, 10)
        arr2 = torch.rand(11, 10)

        # Test the sanity checker
        with self.assertRaises(ValueError):
            (prediction_arr, reference_arr), checker = (
                sanity_check_and_convert_to_array(arr1, arr2)
            )
            self.assertEqual(checker, INPUTDTYPE.TORCH)

        with self.assertRaises(ValueError):
            (prediction_arr, reference_arr), checker = (
                sanity_check_and_convert_to_array(arr2, arr1)
            )
            self.assertEqual(checker, INPUTDTYPE.TORCH)

    def test_sanity_checker_as_file(self):
        # Create two identical numpy arrays
        arr1 = torch.rand(10, 10)
        torch.save(arr1, test_torch_file)

        # Test the sanity checker with Path
        (prediction_arr, reference_arr), checker = sanity_check_and_convert_to_array(
            test_torch_file, test_torch_file
        )
        self.assertEqual(checker, INPUTDTYPE.TORCH)
        # Test the sanity checker with str
        (prediction_arr, reference_arr), checker = sanity_check_and_convert_to_array(
            str(test_torch_file), str(test_torch_file)
        )
        self.assertEqual(checker, INPUTDTYPE.TORCH)
        os.remove(test_torch_file)
