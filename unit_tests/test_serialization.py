import os
import unittest
from pathlib import Path

import numpy as np

from panoptica import InputType, Panoptica_Aggregator, Panoptica_Statistic
from panoptica.instance_approximator import ConnectedComponentsInstanceApproximator
from panoptica.instance_matcher import NaiveThresholdMatching
from panoptica.panoptica_aggregator import _write_content
from panoptica.panoptica_evaluator import Panoptica_Evaluator
from panoptica.utils.label_group import LabelGroup
from panoptica.utils.segmentation_class import SegmentationClassGroups
from panoptica.utils.serialization import (
    format_instance_subject_name,
    is_instance_row,
    parse_instance_subject_name,
    validate_group_name,
    validate_subject_name,
    format_threshold_key,
    format_autc_key,
    parse_threshold_key,
    parse_autc_key,
)


class Test_Serialization_RoundTrip(unittest.TestCase):
    def test_roundtrip_with_hyphenated_subject(self):
        # Subject names may legitimately contain '-'. The group name must not.
        formatted = format_instance_subject_name("patient-001", "liver", 3)
        parsed = parse_instance_subject_name(formatted)
        self.assertEqual(parsed, ("patient-001", "liver", 3))
        self.assertTrue(is_instance_row(formatted))

    def test_roundtrip_simple(self):
        formatted = format_instance_subject_name("subj", "group_0", 0)
        parsed = parse_instance_subject_name(formatted)
        self.assertEqual(parsed, ("subj", "group_0", 0))

    def test_master_subject_not_classified_as_instance(self):
        self.assertFalse(is_instance_row("simple_subject"))
        self.assertFalse(is_instance_row("patient-001"))


class Test_Serialization_Validation(unittest.TestCase):
    def test_group_name_with_hyphen_rejected(self):
        with self.assertRaises(ValueError):
            validate_group_name("left-lung")

    def test_group_name_with_inst_token_rejected(self):
        with self.assertRaises(ValueError):
            validate_group_name("foo_inst_bar")

    def test_group_name_valid(self):
        validate_group_name("liver")
        validate_group_name("group_0")
        validate_group_name("left_lung")

    def test_subject_name_collision_rejected(self):
        # Matches the instance-row shape: <...>-<...>_inst_<int>
        with self.assertRaises(ValueError):
            validate_subject_name("foo-bar_inst_3")

    def test_subject_name_valid(self):
        validate_subject_name("patient-001")
        validate_subject_name("subject_with_inst_in_name")
        validate_subject_name("plain")

    def test_format_rejects_bad_group(self):
        with self.assertRaises(ValueError):
            format_instance_subject_name("subj", "bad-group", 0)

    def test_format_rejects_bad_subject(self):
        with self.assertRaises(ValueError):
            format_instance_subject_name("foo-bar_inst_3", "liver", 0)


class Test_SegmentationClassGroups_Validation(unittest.TestCase):
    def test_dict_key_with_hyphen_rejected(self):
        with self.assertRaises(ValueError):
            SegmentationClassGroups({"left-lung": LabelGroup([1])})

    def test_dict_key_with_inst_token_rejected(self):
        with self.assertRaises(ValueError):
            SegmentationClassGroups({"foo_inst_bar": LabelGroup([1])})

    def test_auto_generated_names_pass(self):
        # list-form construction uses "group_{idx}" which must remain valid.
        SegmentationClassGroups([LabelGroup([1]), LabelGroup([2])])

    def test_valid_dict_keys_pass(self):
        SegmentationClassGroups({"liver": LabelGroup([1]), "spleen": LabelGroup([2])})


class Test_Evaluator_Validates_Subject(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"
        self.output_file = Path(__file__).parent.joinpath(
            "unittest_serialization_tmp.tsv"
        )
        self.buffer_file = Path(str(self.output_file) + ".tmp")
        return super().setUp()

    def tearDown(self) -> None:
        for p in (self.output_file, self.buffer_file):
            if p.exists():
                os.remove(str(p))
        return super().tearDown()

    def test_evaluate_rejects_collision_shaped_subject(self):
        a = np.zeros([20, 20], dtype=np.uint16)
        b = a.copy()
        a[5:10, 5:10] = 1
        b[5:10, 5:10] = 1

        evaluator = Panoptica_Evaluator(
            expected_input=InputType.SEMANTIC,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=NaiveThresholdMatching(),
        )
        aggregator = Panoptica_Aggregator(evaluator, output_file=self.output_file)

        with self.assertRaises(ValueError):
            aggregator.evaluate(b, a, "foo-bar_inst_3")


class Test_Write_Content_None_Normalization(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_file = Path(__file__).parent.joinpath(
            "unittest_none_normalization.tsv"
        )
        if self.tmp_file.exists():
            os.remove(str(self.tmp_file))
        return super().setUp()

    def tearDown(self) -> None:
        if self.tmp_file.exists():
            os.remove(str(self.tmp_file))
        return super().tearDown()

    def test_none_written_as_empty_and_read_back_as_none(self):
        header = ["subject_name", "liver-dice", "liver-tp"]
        row = ["subj_a", None, 5]
        _write_content(self.tmp_file, [header, row])

        stat = Panoptica_Statistic.from_file(self.tmp_file, verbose=False)
        # Missing value must round-trip to None, not blow up on float("None").
        self.assertEqual(stat.get("liver", "dice"), [None])
        self.assertEqual(stat.get("liver", "tp"), [5.0])


class Test_Serialization(unittest.TestCase):
    def test_format_threshold_key(self):
        self.assertEqual(format_threshold_key(0.5, "pq"), "t0.5_pq")
        self.assertEqual(format_threshold_key(1.0, "sq"), "t1_sq")
        self.assertEqual(format_threshold_key(0.00001, "dsc"), "t1e-05_dsc")

    def test_parse_threshold_key(self):
        # Standard
        self.assertEqual(parse_threshold_key("t0.5_pq"), (0.5, "pq"))
        self.assertEqual(parse_threshold_key("t1_sq"), (1.0, "sq"))

        # Scientific Notation
        self.assertEqual(parse_threshold_key("t1e-05_dsc"), (1e-05, "dsc"))
        self.assertEqual(parse_threshold_key("t5.5e+02_metric"), (550.0, "metric"))

        # Invalid / Not a threshold
        self.assertIsNone(parse_threshold_key("pq"))
        self.assertIsNone(parse_threshold_key("autc_pq"))
        self.assertIsNone(parse_threshold_key("t_invalid_pq"))

    def test_parse_autc_key(self):
        self.assertEqual(parse_autc_key("autc_pq"), "pq")
        self.assertIsNone(parse_autc_key("pq"))
        self.assertIsNone(parse_autc_key("t0.5_pq"))

    def test_round_trip_threshold_key(self):
        """Tests that parsing a formatted threshold key yields the exact original inputs."""
        test_cases = [
            (0.5, "pq"),
            (1.0, "sq"),
            (0.00001, "dsc"),
            (1e-5, "dsc"),
            (550.0, "metric"),
            (0.0, "tp"),
        ]

        for threshold, metric in test_cases:
            with self.subTest(threshold=threshold, metric=metric):
                formatted = format_threshold_key(threshold, metric)
                parsed = parse_threshold_key(formatted)
                self.assertIsNotNone(parsed)
                self.assertEqual(parsed, (threshold, metric))

    def test_round_trip_autc_key(self):
        """Tests that parsing a formatted AUTC key yields the exact original metric."""
        test_cases = ["pq", "sq", "dsc", "tp", "custom_metric_name"]

        for metric in test_cases:
            with self.subTest(metric=metric):
                formatted = format_autc_key(metric)
                parsed = parse_autc_key(formatted)

                self.assertEqual(parsed, metric)
