# Call 'python -m unittest' on this folder
# coverage run -m unittest
# coverage report
# coverage html
import os
import unittest

from panoptica.metrics import (
    Metric,
    _Metric,
    Evaluation_List_Metric,
    MetricMode,
    MetricCouldNotBeComputedException,
)
from panoptica.utils.filepath import config_by_name
from panoptica.utils.segmentation_class import SegmentationClassGroups, LabelGroup
from panoptica.utils.constants import CCABackend
from panoptica.utils.edge_case_handling import (
    EdgeCaseResult,
    EdgeCaseZeroTP,
    MetricZeroTPEdgeCaseHandling,
    EdgeCaseHandler,
)
from panoptica import (
    ConnectedComponentsInstanceApproximator,
    NaiveThresholdMatching,
    MaxBipartiteMatching,
    Panoptica_Evaluator,
)
from pathlib import Path
import numpy as np
import random

test_file = Path(__file__).parent.joinpath("test.yaml")


class Test_Datatypes(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"
        return super().setUp()

    def test_enum_config(self):
        a = CCABackend.cc3d
        a.save_to_config(test_file)
        print(a)
        b = CCABackend.load_from_config(test_file)
        print(b)
        os.remove(test_file)

        self.assertEqual(a, b)

    def test_enum_config_all(self):
        for enum in [CCABackend, EdgeCaseZeroTP, EdgeCaseResult, Metric]:
            for a in enum:
                a.save_to_config(test_file)
                print(a)
                b = enum.load_from_config(test_file)
                print(b)
                os.remove(test_file)

                self.assertEqual(a, b)
                self.assertEqual(a.name, b.name)
                self.assertEqual(str(a), str(b))

                # check for value equality
                if a.value is None:
                    self.assertTrue(b.value is None)
                else:
                    # if it is _Metric object, just check name
                    if not isinstance(a.value, _Metric):
                        if np.isnan(a.value):
                            self.assertTrue(np.isnan(b.value))
                        else:
                            self.assertEqual(a.value, b.value)
                    else:
                        self.assertEqual(a.value.name, b.value.name)

    def test_SegmentationClassGroups_config(self):
        e = {
            "groups": {
                "vertebra": LabelGroup([i for i in range(1, 11)], False),
                "ivd": LabelGroup([i for i in range(101, 111)]),
                "sacrum": LabelGroup(26, True),
                "endplate": LabelGroup([i for i in range(201, 211)]),
            }
        }
        t = SegmentationClassGroups(**e)
        print(t)
        print()
        t.save_to_config(test_file)
        d: SegmentationClassGroups = SegmentationClassGroups.load_from_config(test_file)
        os.remove(test_file)

        for k, v in d.items():
            self.assertEqual(t[k].single_instance, v.single_instance)
            self.assertEqual(len(t[k].value_labels), len(v.value_labels))

    def test_SegmentationClassGroups_config_by_name(self):
        e = {
            "groups": {
                "vertebra": LabelGroup([i for i in range(1, 11)], False),
                "ivd": LabelGroup([i for i in range(101, 111)]),
                "sacrum": LabelGroup(26, True),
                "endplate": LabelGroup([i for i in range(201, 211)]),
            }
        }
        t = SegmentationClassGroups(**e)
        print(t)
        print()

        configname = "test_file.yaml"
        t.save_to_config(configname)
        d: SegmentationClassGroups = SegmentationClassGroups.load_from_config(configname)

        testfile_d = config_by_name(configname)
        print(testfile_d)
        os.remove(testfile_d)

        for k, v in d.items():
            self.assertEqual(t[k].single_instance, v.single_instance)
            self.assertEqual(len(t[k].value_labels), len(v.value_labels))

    def test_InstanceApproximator_config(self):
        for backend in [None, CCABackend.cc3d, CCABackend.scipy]:
            t = ConnectedComponentsInstanceApproximator(cca_backend=backend)
            print(t)
            print()
            t.save_to_config(test_file)
            d: ConnectedComponentsInstanceApproximator = ConnectedComponentsInstanceApproximator.load_from_config(test_file)
            os.remove(test_file)

            self.assertEqual(d.cca_backend, t.cca_backend)

    def test_NaiveThresholdMatching_config(self):
        for mm in [Metric.DSC, Metric.IOU, Metric.ASSD]:
            for mt in [0.1, 0.4, 0.5, 0.8, 1.0]:
                for amto in [False, True]:
                    t = NaiveThresholdMatching(
                        matching_metric=mm,
                        matching_threshold=mt,
                        allow_many_to_one=amto,
                    )
                    print(t)
                    print()
                    t.save_to_config(test_file)
                    d: NaiveThresholdMatching = NaiveThresholdMatching.load_from_config(test_file)
                    os.remove(test_file)

                    self.assertEqual(d._allow_many_to_one, t._allow_many_to_one)
                    self.assertEqual(d._matching_metric, t._matching_metric)
                    self.assertEqual(d._matching_threshold, t._matching_threshold)

    def test_MaxBipartiteMatching_config(self):
        for mm in [Metric.DSC, Metric.IOU, Metric.ASSD]:
            for mt in [0.1, 0.4, 0.5, 0.8, 1.0]:
                t = MaxBipartiteMatching(
                    matching_metric=mm,
                    matching_threshold=mt,
                )
                print(t)
                print()
                t.save_to_config(test_file)
                d: MaxBipartiteMatching = MaxBipartiteMatching.load_from_config(test_file)
                os.remove(test_file)

                self.assertEqual(d._matching_metric, t._matching_metric)
                self.assertEqual(d._matching_threshold, t._matching_threshold)

    def test_MetricZeroTPEdgeCaseHandling_config(self):
        for iter in range(10):
            args = [random.choice(list(EdgeCaseResult)) for i in range(5)]

            t = MetricZeroTPEdgeCaseHandling(*args)
            print(t)
            print()
            t.save_to_config(test_file)
            d: MetricZeroTPEdgeCaseHandling = MetricZeroTPEdgeCaseHandling.load_from_config(test_file)
            os.remove(test_file)

            for k, v in t._edgecase_dict.items():
                self.assertEqual(v, d._edgecase_dict[k])
            # self.assertEqual(d.cca_backend, t.cca_backend)

    def test_EdgeCaseHandler_config(self):
        t = EdgeCaseHandler()
        print(t)
        print()
        t.save_to_config(test_file)
        d: EdgeCaseHandler = EdgeCaseHandler.load_from_config(test_file)
        os.remove(test_file)

        self.assertEqual(t.handle_empty_list_std(), d.handle_empty_list_std())
        for k, v in t.listmetric_zeroTP_handling.items():
            # v is dict[Metric, MetricZeroTPEdgeCaseHandling]
            v2 = d.listmetric_zeroTP_handling[k]

            print(v)
            print(v2)

            self.assertEqual(v, v2)

    def test_Panoptica_Evaluator_config(self):
        t = Panoptica_Evaluator()
        print(t)
        print()
        t.save_to_config(test_file)
        d: Panoptica_Evaluator = Panoptica_Evaluator.load_from_config(test_file)
        os.remove(test_file)
