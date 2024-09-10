# Call 'python -m unittest' on this folder
# coverage run -m unittest
# coverage report
# coverage html
import os
import unittest
import numpy as np

from panoptica.utils.segmentation_class import LabelGroup, SegmentationClassGroups
from panoptica.utils import LabelMergeGroup


class Test_DefinitionOfSegmentationLabels(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"
        return super().setUp()

    def test_labelgroup(self):
        group1 = LabelGroup([1, 2, 3, 4, 5], single_instance=False)

        print(group1)
        arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        group1_arr = group1.extract_label(arr, True)

        print(group1_arr)
        self.assertEqual(group1_arr.sum(), 5)

        group1_arr_ind = np.argwhere(group1_arr).flatten()
        print(group1_arr_ind)
        group1_labels = np.asarray(group1.value_labels)
        print(group1_labels)
        self.assertTrue(np.all(group1_arr_ind == group1_labels))

    def test_labelgroup2(self):
        group1 = LabelGroup([1, 2, 3, 4, 5], single_instance=False)

        print(group1)
        arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        group1_arr = group1(arr)

        print(group1_arr)
        self.assertEqual(group1_arr.sum(), 15)

        group1_arr_ind = np.argwhere(group1_arr).flatten()
        print(group1_arr_ind)
        group1_labels = np.asarray(group1.value_labels)
        print(group1_labels)
        self.assertTrue(np.all(group1_arr_ind == group1_labels))

    def test_labelmergegroup(self):
        group1 = LabelMergeGroup([1, 2, 3, 4, 5], single_instance=False)

        print(group1)
        arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        group1_arr = group1(arr)

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
        group1_arr = group1(arr)

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

        group1 = LabelGroup([1, 1, 2, 3], single_instance=False)

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

    def test_wrong_classgroup_definitions(self):

        classgroups = SegmentationClassGroups(
            groups={
                "vertebra": LabelGroup([100, 101, 102]),
                "ivds": LabelGroup([100, 101, 102]),
            }
        )

        classgroups = SegmentationClassGroups(
            groups={
                "vertebra": LabelGroup([1, 2, 3]),
                "ivds": LabelGroup([3, 4, 5]),
            }
        )

        classgroups = SegmentationClassGroups(
            groups={
                "vertebra": LabelGroup([1, 2, 3]),
                "ivds": LabelGroup([4, 5, 6]),
            }
        )
        with self.assertRaises(AssertionError):
            classgroups.has_defined_labels_for([0], raise_error=True)
        with self.assertRaises(AssertionError):
            classgroups.has_defined_labels_for([7], raise_error=True)
        classgroups.has_defined_labels_for([7], raise_error=False)

    def test_segmentationclassgroup_declarations(self):
        classgroups = SegmentationClassGroups(
            groups=[LabelGroup(i) for i in range(1, 5)]
        )

        keys = classgroups.keys()
        for i in range(1, 5):
            self.assertTrue(f"group_{i-1}" in keys, f"not {i} in {keys}")

    def test_segmentationclassgroup_default(self):
        group1 = LabelGroup([1, 2, 3, 4, 5], single_instance=False)

        print(group1)
        print(group1.value_labels)

        arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        group1_arr = group1(arr)
        print(group1_arr)

        classgroups = SegmentationClassGroups(
            groups={
                "vertebra": group1,
                "ivds": LabelGroup([100, 101, 102]),
            }
        )
        print(classgroups)

        print(classgroups.has_defined_labels_for([1, 2, 3]))

        for i in classgroups:
            print(i)

    def test_segmentationclassgroup_regions(self):
        group1 = LabelMergeGroup([1, 2], single_instance=False)
        group2 = LabelMergeGroup([2, 3], single_instance=False)
        group3 = LabelMergeGroup([1, 3], single_instance=False)

        print(group1)
        print(group1.value_labels)

        arr = np.array([0, 1, 2, 3, 1, 2, 2, 7, 3, 3, 3, 10])
        group1_arr = group1(arr)
        print(group1_arr)

        classgroups = SegmentationClassGroups(
            groups={
                "border": group1,
                "inner": group2,
                "core": group3,
            }
        )
        print(classgroups)

        print(classgroups.has_defined_labels_for([1, 2, 3]))

        for group_name, label_group in classgroups.items():
            arr_grouped = label_group(arr)

            labels = label_group.value_labels
            labels = [l + 1 for l in labels]
            labels = sum(labels)

            self.assertEqual(np.sum(arr_grouped), labels, (group_name, labels, arr_grouped))
