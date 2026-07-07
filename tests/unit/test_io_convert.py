"""Tests for panoptica.io.convert converters."""

from __future__ import annotations

import numpy as np
import pytest

from panoptica.core.errors import InputValidationError
from panoptica.io.convert import nibabel_in, nrrd_in, numpy_in, sitk_in


class TestNumpyConverter:
    """Tests for numpy converter."""

    def test_numpy_passthrough_valid_int(self) -> None:
        """Valid integer numpy array passes through."""
        arr = np.array([0, 1, 2, 3], dtype=np.uint8).reshape(2, 2)
        result_arr, spacing = numpy_in.convert_to_numpy(arr)

        assert result_arr is arr  # Passthrough
        assert spacing is None
        assert result_arr.dtype == np.uint8

    def test_numpy_passthrough_int64(self) -> None:
        """Integer numpy array with larger dtype."""
        arr = np.array([0, 100, 1000], dtype=np.int32)
        result_arr, spacing = numpy_in.convert_to_numpy(arr)

        assert result_arr is arr
        assert spacing is None
        assert result_arr.dtype == np.int32

    def test_numpy_rejects_float(self) -> None:
        """Float arrays are rejected."""
        arr = np.array([0.0, 1.5], dtype=np.float32)

        with pytest.raises(InputValidationError, match="integer dtype"):
            numpy_in.convert_to_numpy(arr)

    def test_numpy_rejects_non_array(self) -> None:
        """Non-array inputs are rejected."""
        with pytest.raises(InputValidationError, match="Expected numpy.ndarray"):
            numpy_in.convert_to_numpy([1, 2, 3])

    def test_numpy_rejects_none(self) -> None:
        """None is rejected."""
        with pytest.raises(InputValidationError, match="Expected numpy.ndarray"):
            numpy_in.convert_to_numpy(None)


class TestSITKConverter:
    """Tests for SimpleITK converter."""

    def test_sitk_conversion_available(self) -> None:
        """Test SITK converter when SimpleITK is available."""
        try:
            import SimpleITK as sitk
        except ImportError:
            pytest.skip("SimpleITK not installed")

        # Create a simple SITK image
        arr = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=np.uint8)
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing((0.5, 1.0, 1.5))

        result_arr, spacing = sitk_in.convert_to_numpy(img)

        # SITK returns (Z, Y, X) order, same as input
        assert result_arr.shape == (2, 2, 2)
        assert np.array_equal(result_arr, arr)
        assert spacing == (0.5, 1.0, 1.5)

    def test_sitk_rejects_non_image(self) -> None:
        """Non-SITK images are rejected."""
        try:
            import SimpleITK as sitk  # noqa: F401

            with pytest.raises(InputValidationError, match="Expected SimpleITK Image"):
                sitk_in.convert_to_numpy(np.array([1, 2, 3]))
        except ImportError:
            pytest.skip("SimpleITK not installed")

    def test_sitk_rejects_float_image(self) -> None:
        """Float SITK images are rejected."""
        try:
            import SimpleITK as sitk
        except ImportError:
            pytest.skip("SimpleITK not installed")

        arr = np.array([[[0.5, 1.5]]], dtype=np.float32)  # 3D array for SITK
        img = sitk.GetImageFromArray(arr)

        with pytest.raises(InputValidationError, match="integer dtype"):
            sitk_in.convert_to_numpy(img)


class TestNibabelConverter:
    """Tests for Nibabel converter."""

    def test_nibabel_conversion_available(self) -> None:
        """Test Nibabel converter when nibabel is available."""
        try:
            import nibabel as nib
        except ImportError:
            pytest.skip("nibabel not installed")

        # Create a simple Nifti image
        arr = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=np.uint8)
        img = nib.Nifti1Image(arr, np.eye(4))

        result_arr, spacing = nibabel_in.convert_to_numpy(img)

        assert result_arr.shape == arr.shape
        assert np.array_equal(result_arr, arr)
        # Nifti header.get_zooms() includes qfac as last element for 3D, so check only spatial dims
        assert spacing is not None
        assert len(spacing) == 3

    def test_nibabel_rejects_non_image(self) -> None:
        """Non-Nifti images are rejected."""
        try:
            import nibabel as nib  # noqa: F401

            with pytest.raises(InputValidationError, match="Expected Nibabel"):
                nibabel_in.convert_to_numpy(np.array([1, 2, 3]))
        except ImportError:
            pytest.skip("nibabel not installed")

    def test_nibabel_rejects_float_image(self) -> None:
        """Float Nifti images are rejected."""
        try:
            import nibabel as nib
        except ImportError:
            pytest.skip("nibabel not installed")

        arr = np.array([0.5, 1.5], dtype=np.float32)
        img = nib.Nifti1Image(arr, np.eye(4))

        with pytest.raises(InputValidationError, match="integer dtype"):
            nibabel_in.convert_to_numpy(img)


class TestNRRDConverter:
    """Tests for NRRD converter."""

    def test_nrrd_conversion_available(self) -> None:
        """Test NRRD converter when nrrd is available."""
        try:
            import nrrd  # noqa: F401
        except ImportError:
            pytest.skip("nrrd not installed")

        # Create mock NRRD-like object
        arr = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=np.uint8)
        header = {
            "space directions": np.array([[1.0, 0, 0], [0, 2.0, 0], [0, 0, 3.0]]),
            "space origin": np.array([0, 0, 0]),
            "dimension": 3,
        }

        # Create a simple wrapper object
        class NRRDImage:
            def __init__(self, data, header_dict):
                self.array = data
                self.header = header_dict

        img = NRRDImage(arr, header)
        result_arr, spacing = nrrd_in.convert_to_numpy(img)

        assert np.array_equal(result_arr, arr)
        assert spacing == pytest.approx((1.0, 2.0, 3.0))

    def test_nrrd_rejects_non_image(self) -> None:
        """Non-NRRD images are rejected."""
        try:
            import nrrd  # noqa: F401

            with pytest.raises(InputValidationError):
                nrrd_in.convert_to_numpy(np.array([1, 2, 3]))
        except ImportError:
            pytest.skip("nrrd not installed")

    def test_nrrd_rejects_float_array(self) -> None:
        """Float NRRD arrays are rejected."""
        try:
            import nrrd  # noqa: F401
        except ImportError:
            pytest.skip("nrrd not installed")

        arr = np.array([0.5, 1.5], dtype=np.float32)
        header = {
            "space directions": np.array([[1.0, 0], [0, 1.0]]),
            "space origin": np.array([0, 0]),
            "dimension": 2,
        }

        class NRRDImage:
            def __init__(self, data, header_dict):
                self.array = data
                self.header = header_dict

        img = NRRDImage(arr, header)

        with pytest.raises(InputValidationError, match="integer dtype"):
            nrrd_in.convert_to_numpy(img)


class TestSanityChecks:
    """Tests for sanity checks on converted arrays."""

    def test_valid_arrays_pass_sanity_check(self) -> None:
        """Valid arrays pass sanity checks."""
        from panoptica.io.sanity import sanity_check

        ref = np.array([[0, 1], [1, 2]], dtype=np.uint8)
        pred = np.array([[0, 1], [2, 1]], dtype=np.uint8)

        sanity_check(ref, pred)  # Should not raise

    def test_shape_mismatch(self) -> None:
        """Shape mismatch is caught."""
        from panoptica.io.sanity import sanity_check

        ref = np.array([[0, 1], [1, 2]], dtype=np.uint8)
        pred = np.array([0, 1, 2], dtype=np.uint8)

        with pytest.raises(InputValidationError, match="shape"):
            sanity_check(ref, pred)

    def test_negative_values_rejected(self) -> None:
        """Negative values are rejected."""
        from panoptica.io.sanity import sanity_check

        ref = np.array([[-1, 1], [1, 2]], dtype=np.int8)
        pred = np.array([[0, 1], [1, 2]], dtype=np.uint8)

        with pytest.raises(InputValidationError, match="negative"):
            sanity_check(ref, pred)

    def test_float_dtype_rejected(self) -> None:
        """Float dtype is rejected."""
        from panoptica.io.sanity import sanity_check

        ref = np.array([[0.0, 1.0], [1.0, 2.0]], dtype=np.float32)
        pred = np.array([[0, 1], [1, 2]], dtype=np.uint8)

        with pytest.raises(InputValidationError, match="integer dtype"):
            sanity_check(ref, pred)
