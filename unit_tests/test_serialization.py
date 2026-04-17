import unittest

from panoptica.utils.serialization import (
    format_threshold_key,
    format_autc_key,
    parse_threshold_key,
    parse_autc_key,
)


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
