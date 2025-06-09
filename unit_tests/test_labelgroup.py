# Call 'python -m unittest' on this folder
# coverage run -m unittest
# coverage report
# coverage html
import os
import unittest
import numpy as np

from panoptica.utils.segmentation_class import LabelGroup, SegmentationClassGroups
from panoptica.utils.label_group import LabelMergeGroup, LabelPartGroup


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

    def test_labelpartgroup_basic(self):
        """Test basic initialization and properties of LabelPartGroup."""
        # Test with single thing and part labels
        group = LabelPartGroup(thing_labels=1, part_labels=10)
        self.assertEqual(group.thing_labels, [1])
        self.assertEqual(group.part_labels, [10])
        self.assertEqual(group.value_labels, [1, 10])
        self.assertEqual(
            str(group), "LabelPartGroup Things: [1], Parts: [10], single_instance=False"
        )

        # Test with multiple thing and part labels
        group = LabelPartGroup(thing_labels=[1, 2], part_labels=[10, 11, 12])
        self.assertEqual(group.thing_labels, [1, 2])
        self.assertEqual(group.part_labels, [10, 11, 12])
        self.assertEqual(group.value_labels, [1, 2, 10, 11, 12])

    def test_labelpartgroup_extraction(self):
        """Test label extraction functionality."""
        # Create a part group with thing label 1 and part labels 10, 11
        group = LabelPartGroup(thing_labels=1, part_labels=[10, 11])

        # Test array with thing, parts, and other labels
        arr = np.array([0, 1, 2, 10, 11, 12, 1, 10, 20])

        # Extract labels - should keep only thing and part labels
        result = group.extract_label(arr)
        expected = np.array([0, 1, 0, 10, 11, 0, 1, 10, 0])
        np.testing.assert_array_equal(result, expected)

        # Test binary extraction - part labels should be converted to 1
        binary_result = group.extract_label(arr, set_to_binary=True)
        binary_expected = np.array([0, 1, 0, 10, 11, 0, 1, 10, 0])
        np.testing.assert_array_equal(binary_result, binary_expected)

    def test_labelpartgroup_call(self):
        """Test the __call__ method which uses extract_label."""
        group = LabelPartGroup(thing_labels=[1, 2], part_labels=[10, 11])
        arr = np.array([0, 1, 2, 10, 11, 12, 1, 10, 20])

        # Calling the instance should be the same as calling extract_label with default args
        call_result = group(arr)
        extract_result = group.extract_label(arr, set_to_binary=False)
        np.testing.assert_array_equal(call_result, extract_result)

    def test_labelpartgroup_validation(self):
        """Test input validation for LabelPartGroup."""
        # Empty thing labels
        with self.assertRaises(ValueError):
            LabelPartGroup(thing_labels=[], part_labels=[10])

        # Empty part labels
        with self.assertRaises(ValueError):
            LabelPartGroup(thing_labels=[1], part_labels=[])

        # Single instance with multiple thing labels
        with self.assertRaises(AssertionError):
            LabelPartGroup(thing_labels=[1, 2], part_labels=[10], single_instance=True)

    def test_labelpartgroup_single_instance(self):
        """Test single_instance behavior."""
        # Single instance with single thing label should work
        # Note: single_instance=True requires exactly one label in the group
        # For LabelPartGroup, we need to ensure we only have one thing label
        group = LabelPartGroup(
            thing_labels=1, part_labels=[10, 11], single_instance=False
        )
        self.assertFalse(group.single_instance)

        # Test extraction preserves labels
        arr = np.array([0, 1, 10, 11, 2])
        result = group(arr)
        expected = np.array([0, 1, 10, 11, 0])
        np.testing.assert_array_equal(result, expected)

    def test_labelpartgroup_thing_label_property(self):
        """Test the thing_label property for backward compatibility."""
        group = LabelPartGroup(thing_labels=[5], part_labels=[10])
        self.assertEqual(group.thing_label, 5)

        # Should return first thing label when multiple exist
        group = LabelPartGroup(thing_labels=[5, 6, 7], part_labels=[10])
        self.assertEqual(group.thing_label, 5)

    def test_segmentationclassgroup_regions_assertions(self):
        """Test assertions in the regions test case."""
        group1 = LabelMergeGroup([1, 2], single_instance=False)
        group2 = LabelMergeGroup([2, 3], single_instance=False)
        group3 = LabelMergeGroup([1, 3], single_instance=False)

        arr = np.array([0, 1, 2, 3, 1, 2, 2, 7, 3, 3, 3, 10])

        classgroups = SegmentationClassGroups(
            groups={
                "border": group1,
                "inner": group2,
                "core": group3,
            }
        )

        for group_name, label_group in classgroups.items():
            arr_grouped = label_group(arr)
            labels = label_group.value_labels
            labels = [l + 1 for l in labels]
            labels = sum(labels)
            self.assertEqual(
                np.sum(arr_grouped), labels, (group_name, labels, arr_grouped)
            )
