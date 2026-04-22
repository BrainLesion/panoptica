# Call 'python -m unittest' on this folder
# coverage run -m unittest
# coverage report
# coverage html
import os
import sys
import unittest
from pathlib import Path

import numpy as np

from panoptica import InputType, Panoptica_Aggregator
from panoptica.instance_approximator import ConnectedComponentsInstanceApproximator
from panoptica.instance_matcher import NaiveThresholdMatching
from panoptica.panoptica_evaluator import Panoptica_Evaluator

output_test_dir = Path(__file__).parent.joinpath("unittest_tmp_file.tsv")


class Test_Example_Scripts(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"
        return super().setUp()

    def test_example_scripts_future(self):
        directory = Path(__file__).parent.parent.joinpath("examples")

        print(directory)
        if not directory.exists():
            self.skipTest(f"directory {directory} does not exist")

        sys.path.append(str(directory))

        from examples.example_spine_statistics import main

        main("future")

    def test_example_scripts_pool(self):
        directory = Path(__file__).parent.parent.joinpath("examples")

        print(directory)
        if not directory.exists():
            self.skipTest(f"directory {directory} does not exist")

        sys.path.append(str(directory))

        from examples.example_spine_statistics import main

        main("pool")

    def test_example_scripts_autc(self):
        directory = Path(__file__).parent.parent.joinpath("examples")

        if not directory.exists():
            self.skipTest(f"directory {directory} does not exist")

        sys.path.append(str(directory))

        from examples.example_spine_autc import main

        main()


class Test_Panoptica_Aggregator(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"
        return super().setUp()

    def test_simple_evaluation(self):
        a = np.zeros([50, 50], dtype=np.uint16)
        b = a.copy().astype(a.dtype)
        a[20:40, 10:20] = 1
        b[20:35, 10:20] = 2

        evaluator = Panoptica_Evaluator(
            expected_input=InputType.SEMANTIC,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=NaiveThresholdMatching(),
        )

        aggregator = Panoptica_Aggregator(evaluator, output_file=output_test_dir)

        aggregator.evaluate(b, a, "test")
        os.remove(str(output_test_dir))

    def test_simple_evaluation_then_statistic(self):
        a = np.zeros([50, 50], dtype=np.uint16)
        b = a.copy().astype(a.dtype)
        a[20:40, 10:20] = 1
        b[20:35, 10:20] = 2

        evaluator = Panoptica_Evaluator(
            expected_input=InputType.SEMANTIC,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=NaiveThresholdMatching(),
        )

        aggregator = Panoptica_Aggregator(evaluator, output_file=output_test_dir)

        aggregator.evaluate(b, a, "test")

        statistic_obj = aggregator.make_statistic()
        statistic_obj.print_summary()

        os.remove(str(output_test_dir))

    def test_aggregator_individual_instance_metrics(self):
        import csv

        a = np.zeros([50, 50], dtype=np.uint16)
        b = a.copy().astype(a.dtype)

        # Instance 1
        a[10:20, 10:20] = 1
        b[10:20, 10:20] = 1

        # Instance 2
        a[30:40, 30:40] = 2
        b[30:40, 30:40] = 2

        evaluator = Panoptica_Evaluator(
            expected_input=InputType.SEMANTIC,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=NaiveThresholdMatching(),
        )

        aggregator = Panoptica_Aggregator(
            evaluator,
            output_file=output_test_dir,
            output_individual_instance_metrics=True,
        )

        aggregator.evaluate(b, a, "test_subject")

        # Read the resulting TSV file to verify rows
        with open(str(output_test_dir), "r", encoding="utf8", newline="") as tsvfile:
            rd = csv.reader(tsvfile, delimiter="\t")
            rows = list(rd)

        # We expect: 1 Header + 1 Master row + 2 Instance rows = 4 rows
        self.assertEqual(len(rows), 4, f"Expected 4 rows, got {len(rows)}")

        header = rows[0]
        master_row = rows[1]
        inst_0_row = rows[2]
        inst_1_row = rows[3]

        # Verify subject names match formatting expectations
        self.assertEqual(header[0], "subject_name")
        self.assertEqual(master_row[0], "test_subject")
        self.assertTrue(inst_0_row[0].startswith("test_subject-"))
        self.assertTrue(inst_0_row[0].endswith("_inst_0"))
        self.assertTrue(inst_1_row[0].startswith("test_subject-"))
        self.assertTrue(inst_1_row[0].endswith("_inst_1"))

        # Grab indices to ensure global metrics are empty strings in instance rows
        # e.g., 'tp' should have a value in master but be empty in instance rows
        tp_index = None
        for i, col_name in enumerate(header):
            if col_name.endswith("-tp"):
                tp_index = i
                break

        self.assertIsNotNone(tp_index, "Could not find TP column in header")

        # Assert Master row has a TP count, but instances leave it blank
        self.assertNotEqual(master_row[tp_index], "")
        self.assertEqual(inst_0_row[tp_index], "")
        self.assertEqual(inst_1_row[tp_index], "")

        # Cleanup
        if os.path.exists(str(output_test_dir)):
            os.remove(str(output_test_dir))
