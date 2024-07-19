# Call 'python -m unittest' on this folder
# coverage run -m unittest
# coverage report
# coverage html
import os
import unittest
import numpy as np

from panoptica.utils.segmentation_class import LabelGroup, SegmentationClassGroups


class Test_DefinitionOfSegmentationLabels(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"
        return super().setUp()

    def test_labelgroup(self):
        group1 = LabelGroup([1, 2, 3, 4, 5], single_instance=False)

        print(group1)
        arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        group1_arr = group1(arr, True)

        print(group1_arr)
        self.assertEqual(group1_arr.sum(), 5)

        group1_arr_ind = np.argwhere(group1_arr).flatten()
        print(group1_arr_ind)
        group1_labels = np.asarray(group1.value_labels)
        print(group1_labels)
        self.assertTrue(np.all(group1_arr_ind == group1_labels))

    def test_labelgroup_notpresent(self):
        group1 = LabelGroup([1, 2, 3, 4, 5], single_instance=False)

        print(group1)
        arr = np.array([0, 6, 7, 8, 0, 15, 6, 7, 8, 9, 10])
        group1_arr = group1(arr, True)

        print(group1_arr)
        self.assertEqual(group1_arr.sum(), 0)

        group1_arr_ind = np.argwhere(group1_arr).flatten()
        self.assertEqual(len(group1_arr_ind), 0)

    def test_wrong_labelgroup_definitions(self):

        with self.assertRaises(AssertionError):
            group1 = LabelGroup([1, 2, 3, 4, 5], single_instance=True)

        with self.assertRaises(AssertionError):
            group1 = LabelGroup([], single_instance=False)

        with self.assertRaises(AssertionError):
            group1 = LabelGroup([1, 0, -1, 5], single_instance=False)

    def test_segmentationclassgroup_easy(self):
        group1 = LabelGroup([1, 2, 3, 4, 5], single_instance=False)
        classgroups = SegmentationClassGroups(
            groups={
                "vertebra": group1,
                "ivds": LabelGroup([100, 101, 102]),
            }
        )

        print(classgroups)

        self.assertTrue(classgroups.has_defined_labels_for([1, 2, 3]))

        self.assertTrue(classgroups.has_defined_labels_for([1, 100, 3]))

        self.assertFalse(classgroups.has_defined_labels_for([1, 99, 3]))

        self.assertTrue("ivds" in classgroups)

        for i in classgroups:
            self.assertTrue(i in ["vertebra", "ivds"])

        for i, lg in classgroups.items():
            print(i, lg)
            self.assertTrue(isinstance(i, str))
            self.assertTrue(isinstance(lg, LabelGroup))

    def test_segmentationclassgroup_decarations(self):
        classgroups = SegmentationClassGroups(
            groups=[LabelGroup(i) for i in range(1, 5)]
        )

        keys = classgroups.keys()
        for i in range(1, 5):
            self.assertTrue(f"group_{i-1}" in keys, f"not {i} in {keys}")
