# Unit tests for CuPy connected components functionality
import os
import unittest
import numpy as np
from unittest.mock import patch, MagicMock

from panoptica.utils.constants import CCABackend
from panoptica._functionals import _connected_components
from panoptica import ConnectedComponentsInstanceApproximator
from panoptica.utils.processing_pair import SemanticPair


class Test_CuPy_Connected_Components(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"
        return super().setUp()

    def create_test_binary_array(self):
        """Create a simple test binary array with known connected components."""
        # Create a 3D array with 3 separate connected components
        arr = np.zeros((10, 10, 10), dtype=np.bool_)

        # Component 1: small cube in corner
        arr[1:3, 1:3, 1:3] = True

        # Component 2: larger block in middle
        arr[4:7, 4:7, 4:7] = True

        # Component 3: single isolated voxel
        arr[8, 8, 8] = True

        return arr

    def test_cupy_backend_enum_exists(self):
        """Test that CuPy backend is properly defined in the enum."""
        self.assertTrue(hasattr(CCABackend, "cupy"))
        self.assertEqual(CCABackend.cupy.name, "cupy")

    def test_cupy_not_available_error(self):
        """Test that proper error is raised when CuPy is not available."""
        test_array = self.create_test_binary_array()

        # Mock the import to fail
        with patch(
            "builtins.__import__", side_effect=ImportError("No module named 'cupy'")
        ):
            with self.assertRaises(ImportError) as context:
                _connected_components(test_array, CCABackend.cupy)

            self.assertIn("CuPy is not installed", str(context.exception))
            self.assertIn("pip install cupy-cuda", str(context.exception))

    @patch("cupy.asarray")
    @patch("cupy.asnumpy")
    @patch("cupyx.scipy.ndimage.label")
    def test_cupy_connected_components_function(
        self, mock_cp_label, mock_asnumpy, mock_asarray
    ):
        """Test the _connected_components function with CuPy backend."""
        test_array = self.create_test_binary_array()

        # Mock CuPy functions
        mock_gpu_array = MagicMock()
        mock_asarray.return_value = mock_gpu_array

        # Mock the label function to return expected results
        expected_labeled_array = np.ones_like(test_array, dtype=np.int32)
        expected_n_components = 3
        mock_cp_label.return_value = (mock_gpu_array, expected_n_components)
        mock_asnumpy.return_value = expected_labeled_array

        # Call the function
        result_array, result_n_components = _connected_components(
            test_array, CCABackend.cupy
        )

        # Verify the calls
        mock_asarray.assert_called_once_with(test_array)
        mock_cp_label.assert_called_once_with(mock_gpu_array)
        mock_asnumpy.assert_called_once_with(mock_gpu_array)

        # Verify the results
        np.testing.assert_array_equal(result_array, expected_labeled_array)
        self.assertEqual(result_n_components, expected_n_components)

    def test_cupy_backend_comparison_with_scipy(self):
        """Test that CuPy and SciPy backends produce similar results (when CuPy is available)."""
        test_array = self.create_test_binary_array()

        try:
            # Try to get results from both backends
            scipy_result, scipy_n = _connected_components(test_array, CCABackend.scipy)
            cupy_result, cupy_n = _connected_components(test_array, CCABackend.cupy)

            # Both should find the same number of components
            self.assertEqual(scipy_n, cupy_n)

            # The label values might be different, but the structure should be the same
            # Check that both arrays have the same shape and dtype
            self.assertEqual(scipy_result.shape, cupy_result.shape)

            # Check that both find the same connected regions (regardless of label values)
            scipy_unique = len(np.unique(scipy_result)) - 1  # subtract 1 for background
            cupy_unique = len(np.unique(cupy_result)) - 1  # subtract 1 for background
            self.assertEqual(scipy_unique, cupy_unique)

        except ImportError:
            # CuPy not available, skip this test
            self.skipTest("CuPy not available for comparison test")

    def test_instance_approximator_with_cupy_backend(self):
        """Test ConnectedComponentsInstanceApproximator with CuPy backend."""
        try:
            # Create test semantic arrays
            pred_arr = np.zeros((10, 10, 10), dtype=np.uint8)
            ref_arr = np.zeros((10, 10, 10), dtype=np.uint8)

            # Add some semantic labels
            pred_arr[2:5, 2:5, 2:5] = 1  # One region
            pred_arr[6:8, 6:8, 6:8] = 1  # Another region (same semantic class)

            ref_arr[1:4, 1:4, 1:4] = 1  # Overlapping region
            ref_arr[7:9, 7:9, 7:9] = 1  # Another overlapping region

            # Create semantic pair
            semantic_pair = SemanticPair(pred_arr, ref_arr)

            # Create approximator with CuPy backend
            approximator = ConnectedComponentsInstanceApproximator(
                cca_backend=CCABackend.cupy
            )

            # Test approximation
            result = approximator.approximate_instances(semantic_pair)

            # Verify that we get an UnmatchedInstancePair
            from panoptica.utils.processing_pair import UnmatchedInstancePair

            self.assertIsInstance(result, UnmatchedInstancePair)

            # Verify that instances were found
            self.assertGreater(result.n_prediction_instance, 0)
            self.assertGreater(result.n_reference_instance, 0)

        except ImportError:
            # CuPy not available, skip this test
            self.skipTest("CuPy not available for instance approximator test")

    def test_cupy_backend_config_serialization(self):
        """Test that CuPy backend can be serialized/deserialized in config."""
        from pathlib import Path

        test_file = Path(__file__).parent.joinpath("test_cupy.yaml")

        try:
            # Test CuPy backend serialization
            backend = CCABackend.cupy
            backend.save_to_config(test_file)

            loaded_backend = CCABackend.load_from_config(test_file)
            self.assertEqual(backend, loaded_backend)

            # Test with ConnectedComponentsInstanceApproximator
            approximator = ConnectedComponentsInstanceApproximator(
                cca_backend=CCABackend.cupy
            )
            approximator.save_to_config(test_file)

            loaded_approximator = (
                ConnectedComponentsInstanceApproximator.load_from_config(test_file)
            )
            self.assertEqual(loaded_approximator.cca_backend, CCABackend.cupy)

        finally:
            # Clean up
            if test_file.exists():
                os.remove(test_file)

    def test_various_array_shapes_with_cupy(self):
        """Test CuPy backend with different array shapes and dimensions."""
        test_shapes = [
            (50, 50),  # 2D
            (20, 20, 20),  # 3D
            (10, 10, 10, 10),  # 4D
        ]

        for shape in test_shapes:
            with self.subTest(shape=shape):
                try:
                    # Create test array
                    arr = np.zeros(shape, dtype=np.bool_)
                    # Add a small component
                    slices = tuple(slice(1, 3) for _ in range(len(shape)))
                    arr[slices] = True

                    # Test with CuPy
                    result_arr, n_components = _connected_components(
                        arr, CCABackend.cupy
                    )

                    # Should find at least one component
                    self.assertGreaterEqual(n_components, 1)
                    self.assertEqual(result_arr.shape, arr.shape)

                except ImportError:
                    # CuPy not available
                    self.skipTest(f"CuPy not available for shape {shape} test")

    def test_empty_array_with_cupy(self):
        """Test CuPy backend with empty arrays."""
        try:
            empty_arr = np.zeros((10, 10, 10), dtype=np.bool_)

            result_arr, n_components = _connected_components(empty_arr, CCABackend.cupy)

            # Should find no components
            self.assertEqual(n_components, 0)
            self.assertEqual(result_arr.shape, empty_arr.shape)
            # All values should be 0 (background)
            self.assertEqual(np.max(result_arr), 0)

        except ImportError:
            self.skipTest("CuPy not available for empty array test")

    def test_cupy_backend_with_large_array(self):
        """Test CuPy backend with larger arrays to verify GPU memory handling."""
        try:
            # Create a larger test array
            large_arr = np.zeros((100, 100, 50), dtype=np.bool_)

            # Add several components
            large_arr[10:20, 10:20, 10:20] = True  # Component 1
            large_arr[30:40, 30:40, 30:40] = True  # Component 2
            large_arr[60:70, 60:70, 10:20] = True  # Component 3
            large_arr[80:90, 10:20, 30:40] = True  # Component 4

            result_arr, n_components = _connected_components(large_arr, CCABackend.cupy)

            # Should find 4 components
            self.assertEqual(n_components, 4)
            self.assertEqual(result_arr.shape, large_arr.shape)

            # Verify that we have the right number of unique labels (including background)
            unique_labels = np.unique(result_arr)
            self.assertEqual(len(unique_labels), 5)  # 4 components + background (0)

        except ImportError:
            self.skipTest("CuPy not available for large array test")
        except Exception as e:
            # If GPU memory issues or other CUDA errors, skip
            if "CUDA" in str(e) or "memory" in str(e).lower():
                self.skipTest(f"GPU/CUDA issues: {e}")
            else:
                raise


if __name__ == "__main__":
    unittest.main()
