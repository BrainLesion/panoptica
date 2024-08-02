# Call 'python -m unittest' on this folder
# coverage run -m unittest
# coverage report
# coverage html
import os
import unittest

from panoptica.metrics import (
    Metric,
    Evaluation_List_Metric,
    MetricMode,
    MetricCouldNotBeComputedException,
)
from panoptica.utils.segmentation_class import SegmentationClassGroups, LabelGroup
from panoptica.utils.constants import CCABackend
from pathlib import Path

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
