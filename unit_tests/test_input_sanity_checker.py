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
import nrrd

from panoptica.utils.input_check_and_conversion.sanity_checker import (
    sanity_check_and_convert_to_array,
    INPUTDTYPE,
    _InputDataTypeChecker,
    print_available_package_to_input_handlers,
)
from panoptica.utils.input_check_and_conversion.check_nrrd_image import (
    NRRDImage,
)

test_npy_file = Path(__file__).parent.joinpath("test.npy")
test_torch_file = Path(__file__).parent.joinpath("test.pt")
test_nii_file = Path(__file__).parent.joinpath("test.nii.gz")
test_nrrd_file = Path(__file__).parent.joinpath("test.nrrd")
#
test_abc_file = Path(__file__).parent.joinpath("test.abc.npy")


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

        with self.assertRaises(ValueError):
            (prediction_arr, reference_arr), checker = (
                sanity_check_and_convert_to_array(arr2, arr1)
            )

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

        with self.assertRaises(ValueError):
            (prediction_arr, reference_arr), checker = (
                sanity_check_and_convert_to_array(
                    sitk.GetImageFromArray(arr2), sitk.GetImageFromArray(arr1)
                )
            )

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

        with self.assertRaises(ValueError):
            (prediction_arr, reference_arr), checker = (
                sanity_check_and_convert_to_array(
                    nib.Nifti1Image(arr2, affine=np.eye(4)),
                    nib.Nifti1Image(arr1, affine=np.eye(4)),
                )
            )

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

        with self.assertRaises(ValueError):
            (prediction_arr, reference_arr), checker = (
                sanity_check_and_convert_to_array(arr2, arr1)
            )

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


class Test_Input_Sanity_Checker_Nrrd(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"
        return super().setUp()

    def test_sanity_checker(self):
        # Create two identical numpy arrays
        arr1 = np.random.rand(10, 10)
        arr2 = np.copy(arr1)

        nrrd1 = NRRDImage(
            arr1,
            header={
                "space origin": [0, 0],
                "space directions": [[1, 0], [1, 0]],
                "dimension": 2,
            },
        )
        nrrd2 = NRRDImage(
            arr2,
            header={
                "space origin": [0, 0],
                "space directions": [[1, 0], [1, 0]],
                "dimension": 2,
            },
        )

        with self.assertWarns(UserWarning):
            # Test the sanity checker
            (prediction_arr, reference_arr), checker = (
                sanity_check_and_convert_to_array(nrrd1, nrrd2)
            )
            self.assertEqual(checker, INPUTDTYPE.NRRD)
            # self.assertTrue(result)
        with self.assertWarns(UserWarning):
            # Modify one array and test again
            arr2[0, 0] += 1e-6
            nrrd2 = NRRDImage(
                arr2,
                header={
                    "space origin": [0, 0],
                    "space directions": [[1, 0], [1, 0]],
                    "dimension": 2,
                },
            )
            (prediction_arr, reference_arr), checker = (
                sanity_check_and_convert_to_array(nrrd1, nrrd2)
            )
            self.assertEqual(checker, INPUTDTYPE.NRRD)

    def test_sanity_checker_shapemismatch(self):
        # Create two identical numpy arrays
        arr1 = np.random.rand(10, 10)
        arr2 = np.random.rand(10, 11)

        nrrd1 = NRRDImage(
            arr1,
            header={
                "space origin": [0, 0],
                "space directions": [[1, 0], [1, 0]],
                "dimension": 2,
            },
        )
        nrrd2 = NRRDImage(
            arr2,
            header={
                "space origin": [0, 0],
                "space directions": [[1, 0], [1, 0]],
                "dimension": 2,
            },
        )

        # Test the sanity checker
        with self.assertRaises(ValueError):
            (prediction_arr, reference_arr), checker = (
                sanity_check_and_convert_to_array(nrrd1, nrrd2)
            )

        with self.assertRaises(ValueError):
            (prediction_arr, reference_arr), checker = (
                sanity_check_and_convert_to_array(nrrd2, nrrd1)
            )

    def test_sanity_checker_as_file(self):
        # Create two identical numpy arrays
        arr1 = np.random.rand(10, 10)
        affine = np.eye(3, 3)
        n = affine.shape[0] - 1
        space_directions = affine[:n, :n]
        space_origin = affine[:n, n]
        header = {
            "type": str(arr1.dtype),
            "dimension": n,
            "sizes": arr1.shape,  # (data.shape[1],data.shape[0],data.shape[2]),
            "space directions": space_directions.tolist(),
            "space origin": space_origin,
            "endian": "little",
            "encoding": "gzip",
        }
        nrrd.write(
            str(test_nrrd_file),
            arr1,
            header=header,
        )

        # Test the sanity checker with Path
        (prediction_arr, reference_arr), checker = sanity_check_and_convert_to_array(
            test_nrrd_file, test_nrrd_file
        )
        self.assertEqual(checker, INPUTDTYPE.NRRD)
        # Test the sanity checker with str
        (prediction_arr, reference_arr), checker = sanity_check_and_convert_to_array(
            str(test_nrrd_file), str(test_nrrd_file)
        )
        self.assertEqual(checker, INPUTDTYPE.NRRD)
        os.remove(test_nrrd_file)


class Test_Input_Sanity_Checker_Misc(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"
        return super().setUp()

    def test_sanity_checker_unsupported_datatype(self):
        # List
        with self.assertRaises(ValueError):
            (prediction_arr, reference_arr), checker = (
                sanity_check_and_convert_to_array([0, 1, 2], [0, 1, 2])
            )

        # Tuple
        with self.assertRaises(ValueError):
            (prediction_arr, reference_arr), checker = (
                sanity_check_and_convert_to_array((0, 1, 2), (0, 1, 2))
            )

        # Dict
        with self.assertRaises(ValueError):
            (prediction_arr, reference_arr), checker = (
                sanity_check_and_convert_to_array({0: 1, 1: 2}, {0: 1, 1: 2})
            )

        # str
        with self.assertRaises(ValueError):
            (prediction_arr, reference_arr), checker = (
                sanity_check_and_convert_to_array("(0, 1, 2)", "(0, 1, 2)")
            )

    def test_sanity_checker_unsupported_file_ending(self):
        # Create two identical numpy arrays
        arr1 = np.random.rand(10, 10)
        np.save(test_abc_file, arr1)

        with self.assertRaises(ValueError):
            # Test the sanity checker with Path
            (prediction_arr, reference_arr), checker = (
                sanity_check_and_convert_to_array(test_abc_file, test_abc_file)
            )
        with self.assertRaises(ValueError):
            # Test the sanity checker with str
            (prediction_arr, reference_arr), checker = (
                sanity_check_and_convert_to_array(
                    str(test_abc_file), str(test_abc_file)
                )
            )
        os.remove(test_abc_file)
