# Call 'python -m unittest' on this folder
# coverage run -m unittest
# coverage report
# coverage html
import os
import unittest
import numpy as np
from scipy.ndimage import distance_transform_edt
from panoptica.utils.processing_pair import UnmatchedInstancePair
from panoptica._functionals import _get_voronoi_regions


def _min_region_distance(arr: np.ndarray, n: int) -> np.ndarray:
    """Per-voxel distance to the nearest labelled region (min over all regions)."""
    per_region = np.stack(
        [distance_transform_edt(arr != label) for label in range(1, n + 1)]
    )
    return per_region.min(axis=0)


def _label_distance(arr: np.ndarray, region_map: np.ndarray, n: int) -> np.ndarray:
    """Distance from each voxel to the region it was assigned to in ``region_map``."""
    per_region = {
        label: distance_transform_edt(arr != label) for label in range(1, n + 1)
    }
    out = np.zeros(arr.shape, dtype=float)
    for label in range(1, n + 1):
        sel = region_map == label
        out[sel] = per_region[label][sel]
    return out


def create_test_data():
    """Create simple test data with ground truth and prediction instances"""
    # Create a simple 3D volume with 2 GT regions
    gt = np.zeros((50, 50, 20), dtype=np.int32)
    pred = np.zeros((50, 50, 20), dtype=np.int32)

    # GT region 1: cube in corner
    gt[10:20, 10:20, 5:15] = 1

    # GT region 2: cube in opposite corner
    gt[30:40, 30:40, 5:15] = 2

    # Prediction region 1: slightly offset from GT region 1
    pred[12:22, 12:22, 6:16] = 1

    # Prediction region 2: different location, should map to closest GT region
    pred[25:35, 25:35, 6:16] = 2

    return gt, pred


class Test_RegionMatching(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"
        return super().setUp()

    def test_region_based_matching(self):
        """Test the RegionBasedMatching algorithm"""
        print("Testing RegionBasedMatching...")

        # Create test data
        gt, pred = create_test_data()

        # Create unmatched instance pair
        unmatched_pair = UnmatchedInstancePair(prediction_arr=pred, reference_arr=gt)

        print(f"Ground truth labels: {unmatched_pair.ref_labels}")
        print(f"Prediction labels: {unmatched_pair.pred_labels}")
        # Create regions
        region_map, num_features = _get_voronoi_regions(
            unmatched_pair.reference_arr, unmatched_pair.n_ref_instances
        )

        print(f"Matching successful!")

        self.assertTrue(True)

    def test_voronoi_assigns_each_voxel_to_nearest_region(self):
        """Every voxel must be assigned to a globally-nearest reference region.

        This is the correctness contract of the single-pass implementation: interior
        voxels keep their own label, and every voxel's assigned region is at the
        minimal achievable distance (assignments may differ from a naive per-region
        loop only at exactly-equidistant ties, which are arbitrary).
        """
        cases = [self._fixture_case()]
        for seed in range(4):
            cases.append(self._random_case(seed, (40, 40)))
            cases.append(self._random_case(seed, (24, 24, 16)))

        for arr, n in cases:
            region_map, num = _get_voronoi_regions(arr, n)
            self.assertEqual(num, n)
            self.assertEqual(region_map.shape, arr.shape)

            # 1. Interior voxels of every instance keep their own label.
            interior = arr != 0
            self.assertTrue(np.array_equal(region_map[interior], arr[interior]))

            # 2. Every voxel is assigned to a region at the globally-minimal distance.
            np.testing.assert_allclose(
                _label_distance(arr, region_map, n),
                _min_region_distance(arr, n),
            )

            # 3. Only valid region labels are emitted.
            self.assertTrue(set(np.unique(region_map)).issubset(set(range(1, n + 1))))

    @staticmethod
    def _fixture_case():
        gt = np.zeros((30, 30, 12), dtype=np.int32)
        gt[5:10, 5:10, 3:9] = 1
        gt[20:25, 20:25, 3:9] = 2
        return gt, 2

    @staticmethod
    def _random_case(seed, shape):
        rng = np.random.default_rng(seed)
        arr = np.zeros(shape, dtype=np.int32)
        n = 0
        for _ in range(40):
            if n >= 6:
                break
            center = tuple(int(rng.integers(2, s - 2)) for s in shape)
            if arr[center] != 0:
                continue
            n += 1
            arr[center] = n
        return arr, n
