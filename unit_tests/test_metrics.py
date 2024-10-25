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


def case_simple_identical():
    # trivial 100% overlap
    prediction_arr = np.array(
        [
            [0, 1, 1, 1],
            [0, 1, 1, 1],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
        ]
    )
    return prediction_arr, prediction_arr.copy()


def case_simple_nooverlap():
    # binary opposites
    prediction_arr = np.array(
        [
            [0, 1, 0, 1],
            [0, 1, 0, 1],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
        ]
    )
    reference_arr = 1 - prediction_arr
    return prediction_arr, reference_arr


def case_simple_overpredicted():
    # reference is real subset of prediction
    prediction_arr = np.array(
        [
            [0, 0, 1, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
            [0, 1, 1, 0],
        ]
    )
    reference_arr = np.array(
        [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ]
    )
    return prediction_arr, reference_arr


def case_simple_underpredicted():
    # prediction is real subset of reference
    prediction_arr = np.array(
        [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ]
    )
    reference_arr = np.array(
        [
            [0, 0, 1, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
            [0, 1, 1, 0],
        ]
    )
    return prediction_arr, reference_arr


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
        rvd = Metric.RVD(reference_arr=ref_arr, prediction_arr=pred_arr, ref_instance_idx=1, pred_instance_idx=1)
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
        dsc = Metric.DSC(reference_arr=ref_arr, prediction_arr=pred_arr, ref_instance_idx=1, pred_instance_idx=1)
        self.assertEqual(dsc, 1.0)

    def test_dsc_case_simple_identical_wrong_idx(self):

        pred_arr, ref_arr = case_simple_identical()
        dsc = Metric.DSC(reference_arr=ref_arr, prediction_arr=pred_arr, ref_instance_idx=2, pred_instance_idx=2)
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
        st = Metric.ASSD(reference_arr=ref_arr, prediction_arr=pred_arr, ref_instance_idx=1, pred_instance_idx=1)
        self.assertEqual(st, 0.0)

    def test_st_case_simple_nooverlap(self):
        pred_arr, ref_arr = case_simple_nooverlap()
        st = Metric.ASSD(reference_arr=ref_arr, prediction_arr=pred_arr)
        self.assertEqual(st, 1.05)

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
        st = Metric.clDSC(reference_arr=ref_arr, prediction_arr=pred_arr, ref_instance_idx=1, pred_instance_idx=1)
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
