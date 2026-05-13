# Call 'python -m unittest' on this folder
# coverage run -m unittest
# coverage report
# coverage html
import os
import unittest

import numpy as np

from panoptica.metrics import Metric
from panoptica.panoptica_result import (
    MetricCouldNotBeComputedException,
    PanopticaResult,
)
from panoptica.utils.edge_case_handling import EdgeCaseHandler, EdgeCaseResult
from unit_tests.unit_test_utils import (
    case_simple_identical,
    case_simple_nooverlap,
    case_simple_overpredicted,
    case_simple_shifted,
    case_simple_underpredicted,
    case_simple_overlap_but_large_discrepancy,
    case_multiple_overlapping_instances,
)


class Test_RVD(unittest.TestCase):
    # case_simple_nooverlap
    # case_simple_nooverlap
    # case_simple_overpredicted
    # case_simple_underpredicted

    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"
        return super().setUp()

    def test_rvd_case_simple_identical(self):
        pred_arr, ref_arr = case_simple_identical()
        rvd = Metric.RVD(
            reference_arr=ref_arr,
            prediction_arr=pred_arr,
            ref_instance_idx=1,
            pred_instance_idx=1,
        )
        self.assertEqual(rvd, 0.0)

    def test_rvd_case_simple_identical_idx(self):
        pred_arr, ref_arr = case_simple_identical()
        rvd = Metric.RVD(reference_arr=ref_arr, prediction_arr=pred_arr)
        self.assertEqual(rvd, 0.0)

    def test_rvd_case_simple_nooverlap(self):
        pred_arr, ref_arr = case_simple_nooverlap()
        rvd = Metric.RVD(reference_arr=ref_arr, prediction_arr=pred_arr)
        self.assertEqual(rvd, -0.4)

    def test_rvd_case_simple_overpredicted(self):
        pred_arr, ref_arr = case_simple_overpredicted()
        rvd = Metric.RVD(reference_arr=ref_arr, prediction_arr=pred_arr)
        self.assertEqual(rvd, 1.5)

    def test_rvd_case_simple_underpredicted(self):
        pred_arr, ref_arr = case_simple_underpredicted()
        rvd = Metric.RVD(reference_arr=ref_arr, prediction_arr=pred_arr)
        self.assertEqual(rvd, -0.6)


class Test_DSC(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"
        return super().setUp()

    def test_dsc_case_simple_identical(self):
        pred_arr, ref_arr = case_simple_identical()
        dsc = Metric.DSC(reference_arr=ref_arr, prediction_arr=pred_arr)
        self.assertEqual(dsc, 1.0)

    def test_dsc_case_simple_identical_idx(self):
        pred_arr, ref_arr = case_simple_identical()
        dsc = Metric.DSC(
            reference_arr=ref_arr,
            prediction_arr=pred_arr,
            ref_instance_idx=1,
            pred_instance_idx=1,
        )
        self.assertEqual(dsc, 1.0)

    def test_dsc_case_simple_identical_wrong_idx(self):
        pred_arr, ref_arr = case_simple_identical()
        dsc = Metric.DSC(
            reference_arr=ref_arr,
            prediction_arr=pred_arr,
            ref_instance_idx=2,
            pred_instance_idx=2,
        )
        self.assertEqual(dsc, 0.0)

    def test_dsc_case_simple_nooverlap(self):
        pred_arr, ref_arr = case_simple_nooverlap()
        dsc = Metric.DSC(reference_arr=ref_arr, prediction_arr=pred_arr)
        self.assertEqual(dsc, 0.0)

    def test_dsc_case_simple_overpredicted(self):
        pred_arr, ref_arr = case_simple_overpredicted()
        dsc = Metric.DSC(reference_arr=ref_arr, prediction_arr=pred_arr)
        self.assertEqual(dsc, 0.5714285714285714)

    def test_dsc_case_simple_underpredicted(self):
        pred_arr, ref_arr = case_simple_underpredicted()
        dsc = Metric.DSC(reference_arr=ref_arr, prediction_arr=pred_arr)
        self.assertEqual(dsc, 0.5714285714285714)

    def test_dsc_case_multiple_overlapping_instances(self):
        pred_arr, ref_arr = case_multiple_overlapping_instances()
        dsc = Metric.DSC(reference_arr=ref_arr, prediction_arr=pred_arr)
        print(dsc)
        self.assertAlmostEqual(dsc, 0.106583072)

        dsc = Metric.DSC(
            reference_arr=ref_arr,
            prediction_arr=pred_arr,
            ref_instance_idx=1,
            pred_instance_idx=1,
        )
        print(dsc)
        self.assertAlmostEqual(dsc, 0.094117647)

        dsc = Metric.DSC(
            reference_arr=ref_arr,
            prediction_arr=pred_arr,
            ref_instance_idx=2,
            pred_instance_idx=2,
        )
        print(dsc)
        self.assertAlmostEqual(dsc, 0.068376068)

        dsc = Metric.DSC(
            reference_arr=ref_arr,
            prediction_arr=pred_arr,
            ref_instance_idx=1,
            pred_instance_idx=2,
        )
        print(dsc)
        self.assertAlmostEqual(dsc, 0.15384615)

        dsc = Metric.DSC(
            reference_arr=ref_arr,
            prediction_arr=pred_arr,
            ref_instance_idx=2,
            pred_instance_idx=1,
        )
        print(dsc)
        self.assertEqual(dsc, 0.0)


class Test_ASSD(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"
        return super().setUp()

    def test_st_case_simple_identical(self):
        pred_arr, ref_arr = case_simple_identical()
        st = Metric.ASSD(reference_arr=ref_arr, prediction_arr=pred_arr)
        self.assertEqual(st, 0.0)

    def test_st_case_simple_identical_idx(self):
        pred_arr, ref_arr = case_simple_identical()
        st = Metric.ASSD(
            reference_arr=ref_arr,
            prediction_arr=pred_arr,
            ref_instance_idx=1,
            pred_instance_idx=1,
        )
        self.assertEqual(st, 0.0)

    def test_st_case_simple_nooverlap(self):
        pred_arr, ref_arr = case_simple_nooverlap()
        st = Metric.ASSD(reference_arr=ref_arr, prediction_arr=pred_arr)
        self.assertEqual(st, 1.05)

    def test_st_case_simple_nooverlap_voxelspacing(self):
        pred_arr, ref_arr = case_simple_nooverlap()
        st = Metric.ASSD(
            reference_arr=ref_arr, prediction_arr=pred_arr, voxelspacing=(20.0, 20.0)
        )
        self.assertEqual(st, 21.0)

    def test_st_case_simple_overpredicted(self):
        pred_arr, ref_arr = case_simple_overpredicted()
        st = Metric.ASSD(reference_arr=ref_arr, prediction_arr=pred_arr)
        self.assertEqual(st, 0.625)

    def test_st_case_simple_underpredicted(self):
        pred_arr, ref_arr = case_simple_underpredicted()
        st = Metric.ASSD(reference_arr=ref_arr, prediction_arr=pred_arr)
        self.assertEqual(st, 0.625)


class Test_clDSC(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"
        return super().setUp()

    def test_st_case_simple_identical(self):
        pred_arr, ref_arr = case_simple_identical()
        st = Metric.clDSC(reference_arr=ref_arr, prediction_arr=pred_arr)
        self.assertEqual(st, 1.0)

    def test_st_case_simple_identical_idx(self):
        pred_arr, ref_arr = case_simple_identical()
        st = Metric.clDSC(
            reference_arr=ref_arr,
            prediction_arr=pred_arr,
            ref_instance_idx=1,
            pred_instance_idx=1,
        )
        self.assertEqual(st, 1.0)

    def test_st_case_simple_nooverlap(self):
        pred_arr, ref_arr = case_simple_nooverlap()
        st = Metric.clDSC(reference_arr=ref_arr, prediction_arr=pred_arr)
        self.assertEqual(np.isnan(st), True)

    def test_st_case_simple_overpredicted(self):
        pred_arr, ref_arr = case_simple_overpredicted()
        st = Metric.clDSC(reference_arr=ref_arr, prediction_arr=pred_arr)
        self.assertEqual(st, 1.0)

    def test_st_case_simple_underpredicted(self):
        pred_arr, ref_arr = case_simple_underpredicted()
        st = Metric.clDSC(reference_arr=ref_arr, prediction_arr=pred_arr)
        self.assertEqual(st, 1.0)


class Test_RVAE(unittest.TestCase):
    # case_simple_nooverlap
    # case_simple_nooverlap
    # case_simple_overpredicted
    # case_simple_underpredicted

    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"
        return super().setUp()

    def test_rvae_case_simple_identical(self):
        pred_arr, ref_arr = case_simple_identical()
        rvd = Metric.RVAE(
            reference_arr=ref_arr,
            prediction_arr=pred_arr,
            ref_instance_idx=1,
            pred_instance_idx=1,
        )
        self.assertEqual(rvd, 0.0)

    def test_rvae_case_simple_identical_idx(self):
        pred_arr, ref_arr = case_simple_identical()
        rvd = Metric.RVAE(reference_arr=ref_arr, prediction_arr=pred_arr)
        self.assertEqual(rvd, 0.0)

    def test_rvae_case_simple_nooverlap(self):
        pred_arr, ref_arr = case_simple_nooverlap()
        rvd = Metric.RVAE(reference_arr=ref_arr, prediction_arr=pred_arr)
        self.assertEqual(rvd, 0.4)

    def test_rvae_case_simple_overpredicted(self):
        pred_arr, ref_arr = case_simple_overpredicted()
        rvd = Metric.RVAE(reference_arr=ref_arr, prediction_arr=pred_arr)
        self.assertEqual(rvd, 1.5)

    def test_rvae_case_simple_underpredicted(self):
        pred_arr, ref_arr = case_simple_underpredicted()
        rvd = Metric.RVAE(reference_arr=ref_arr, prediction_arr=pred_arr)
        self.assertEqual(rvd, 0.6)


class Test_CEDI(unittest.TestCase):
    # case_simple_nooverlap
    # case_simple_nooverlap
    # case_simple_overpredicted
    # case_simple_underpredicted

    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"
        return super().setUp()

    def test_cedi_case_simple_identical(self):
        pred_arr, ref_arr = case_simple_identical()
        mv = Metric.CEDI(
            reference_arr=ref_arr,
            prediction_arr=pred_arr,
            ref_instance_idx=1,
            pred_instance_idx=1,
        )
        self.assertEqual(mv, 0.0)

    def test_cedi_case_simple_identical_idx(self):
        pred_arr, ref_arr = case_simple_identical()
        mv = Metric.CEDI(reference_arr=ref_arr, prediction_arr=pred_arr)
        self.assertEqual(mv, 0.0)

    def test_cedi_case_simple_shifted(self):
        pred_arr, ref_arr = case_simple_shifted()
        mv = Metric.CEDI(reference_arr=ref_arr, prediction_arr=pred_arr)
        self.assertEqual(mv, 1.0)

    def test_cedi_case_simple_overpredicted(self):
        pred_arr, ref_arr = case_simple_overpredicted()
        mv = Metric.CEDI(reference_arr=ref_arr, prediction_arr=pred_arr)
        self.assertAlmostEqual(mv, 0.22360679774997896)

    def test_cedi_case_simple_underpredicted(self):
        pred_arr, ref_arr = case_simple_underpredicted()
        mv = Metric.CEDI(reference_arr=ref_arr, prediction_arr=pred_arr)
        self.assertEqual(mv, 0.22360679774997896)


class Test_HD(unittest.TestCase):
    # case_simple_nooverlap
    # case_simple_nooverlap
    # case_simple_overpredicted
    # case_simple_underpredicted

    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"
        return super().setUp()

    def test_hd_case_simple_identical(self):
        pred_arr, ref_arr = case_simple_identical()
        mv = Metric.HD(
            reference_arr=ref_arr,
            prediction_arr=pred_arr,
            ref_instance_idx=1,
            pred_instance_idx=1,
        )
        self.assertEqual(mv, 0.0)
        #
        mv = Metric.HD95(
            reference_arr=ref_arr,
            prediction_arr=pred_arr,
            ref_instance_idx=1,
            pred_instance_idx=1,
        )
        self.assertEqual(mv, 0.0)

    def test_hd_case_simple_identical_idx(self):
        pred_arr, ref_arr = case_simple_identical()
        mv = Metric.HD(reference_arr=ref_arr, prediction_arr=pred_arr)
        self.assertEqual(mv, 0.0)
        #
        mv = Metric.HD95(reference_arr=ref_arr, prediction_arr=pred_arr)
        self.assertEqual(mv, 0.0)

    def test_hd_case_simple_shifted(self):
        pred_arr, ref_arr = case_simple_shifted()
        mv = Metric.HD(reference_arr=ref_arr, prediction_arr=pred_arr)
        self.assertEqual(mv, 1.0)
        #
        mv = Metric.HD95(reference_arr=ref_arr, prediction_arr=pred_arr)
        self.assertEqual(mv, 1.0)

    def test_hd_case_simple_nooverlap_voxelspacing(self):
        pred_arr, ref_arr = case_simple_nooverlap()
        st = Metric.HD(
            reference_arr=ref_arr, prediction_arr=pred_arr, voxelspacing=(20.0, 20.0)
        )
        self.assertEqual(st, 40.0)

    def test_hd_case_simple_overpredicted(self):
        pred_arr, ref_arr = case_simple_overpredicted()
        mv = Metric.HD(reference_arr=ref_arr, prediction_arr=pred_arr)
        self.assertEqual(mv, 1.0)
        #
        mv = Metric.HD95(reference_arr=ref_arr, prediction_arr=pred_arr)
        self.assertEqual(mv, 1.0)

    def test_hd_case_simple_underpredicted(self):
        pred_arr, ref_arr = case_simple_underpredicted()
        mv = Metric.HD(reference_arr=ref_arr, prediction_arr=pred_arr)
        self.assertEqual(mv, 1.0)
        #
        mv = Metric.HD95(reference_arr=ref_arr, prediction_arr=pred_arr)
        self.assertEqual(mv, 1.0)

    def test_hd_case_overlap_but_large_discrepancy(self):
        pred_arr, ref_arr = case_simple_overlap_but_large_discrepancy()
        mv = Metric.HD(reference_arr=ref_arr, prediction_arr=pred_arr)
        self.assertAlmostEqual(mv, 2.82842712474619)
        #
        mv = Metric.HD95(reference_arr=ref_arr, prediction_arr=pred_arr)
        self.assertAlmostEqual(mv, 2.473011636398349)


class Test_NSD(unittest.TestCase):
    # case_simple_nooverlap
    # case_simple_nooverlap
    # case_simple_overpredicted
    # case_simple_underpredicted

    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"
        return super().setUp()

    def test_nsd_case_simple_identical(self):
        pred_arr, ref_arr = case_simple_identical()
        mv = Metric.NSD(
            reference_arr=ref_arr,
            prediction_arr=pred_arr,
            ref_instance_idx=1,
            pred_instance_idx=1,
        )
        self.assertEqual(mv, 1.0)

    def test_nsd_case_simple_identical_idx(self):
        pred_arr, ref_arr = case_simple_identical()
        mv = Metric.NSD(reference_arr=ref_arr, prediction_arr=pred_arr)
        self.assertEqual(mv, 1.0)

    def test_nsd_case_simple_underpredicted_thresholds(self):
        pred_arr, ref_arr = case_simple_underpredicted()
        mv = Metric.NSD(reference_arr=ref_arr, prediction_arr=pred_arr, threshold=0.5)
        self.assertEqual(mv, 0.375)
        #
        mv = Metric.NSD(reference_arr=ref_arr, prediction_arr=pred_arr, threshold=1)
        self.assertEqual(mv, 1.0)


class Test_VOLUME(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"
        return super().setUp()

    def test_volume_unit_spacing_default(self):
        # voxelspacing=None falls back to unit spacing matching reference_arr.ndim
        pred_arr, ref_arr = case_simple_identical()
        ref_voxels = int(np.count_nonzero(ref_arr == 1))
        v = Metric.VOLUME(
            reference_arr=ref_arr,
            prediction_arr=pred_arr,
            ref_instance_idx=1,
            pred_instance_idx=1,
        )
        self.assertEqual(v, float(ref_voxels))

    def test_volume_3d_anisotropic_spacing(self):
        # Non-unit, non-uniform 3D voxelspacing -> volume = count * prod(spacing)
        ref_arr = np.zeros((10, 10, 10), dtype=np.uint8)
        ref_arr[2:5, 2:5, 2:5] = 1  # 3*3*3 = 27 voxels
        pred_arr = ref_arr.copy()
        spacing = (0.5, 2.0, 3.0)  # prod = 3.0
        v = Metric.VOLUME(
            reference_arr=ref_arr,
            prediction_arr=pred_arr,
            ref_instance_idx=1,
            pred_instance_idx=1,
            voxelspacing=spacing,
        )
        self.assertAlmostEqual(v, 27 * 3.0)

    def test_volume_dimension_mismatch_raises(self):
        ref_arr = np.zeros((4, 4), dtype=np.uint8)
        ref_arr[1:3, 1:3] = 1
        with self.assertRaises(ValueError):
            Metric.VOLUME(
                reference_arr=ref_arr,
                prediction_arr=ref_arr,
                voxelspacing=(1.0, 1.0, 1.0),
            )


# class Test_ST(unittest.TestCase):
#    def setUp(self) -> None:
#        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"
#        return super().setUp()

#    def test_st_case_simple_identical(self):
#
#        pred_arr, ref_arr = case_simple_identical()
#        st = Metric.ST(reference_arr=ref_arr, prediction_arr=pred_arr)
#        self.assertEqual(st, 0.0)

#    def test_st_case_simple_nooverlap(self):
#
#        pred_arr, ref_arr = case_simple_nooverlap()
#        st = Metric.ST(reference_arr=ref_arr, prediction_arr=pred_arr)
#        self.assertEqual(st, -0.4)
#
#    def test_st_case_simple_overpredicted(self):
#
#        pred_arr, ref_arr = case_simple_overpredicted()
#        st = Metric.ST(reference_arr=ref_arr, prediction_arr=pred_arr)
#        self.assertEqual(st, 0.5714285714285714)
#
#    def test_st_case_simple_underpredicted(self):
#
#        pred_arr, ref_arr = case_simple_underpredicted()
#        st = Metric.ST(reference_arr=ref_arr, prediction_arr=pred_arr)
#        self.assertEqual(st, 0.5714285714285714)
