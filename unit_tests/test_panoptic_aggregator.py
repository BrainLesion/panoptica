# Call 'python -m unittest' on this folder
# coverage run -m unittest
# coverage report
# coverage html
import csv
import os
import unittest

import numpy as np

from panoptica import InputType, Panoptica_Aggregator, Panoptica_Statistic
from panoptica.instance_approximator import ConnectedComponentsInstanceApproximator
from panoptica.instance_matcher import MaximizeMergeMatching, NaiveThresholdMatching
from panoptica.metrics import Metric
from panoptica.panoptica_evaluator import Panoptica_Evaluator
from panoptica.panoptica_result import MetricCouldNotBeComputedException
from panoptica.utils.processing_pair import SemanticPair
from panoptica.utils.segmentation_class import SegmentationClassGroups
import sys
from pathlib import Path

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

    def test_aggregator_volume_voxelspacing(self):
        # Two equal-sized 10x10 instances => 100 voxels each.
        a = np.zeros([50, 50], dtype=np.uint16)
        b = a.copy()
        a[10:20, 10:20] = 1
        b[10:20, 10:20] = 1
        a[30:40, 30:40] = 2
        b[30:40, 30:40] = 2

        voxels_per_instance = 100

        output_file = Path(__file__).parent.joinpath("unittest_volume_voxsp.tsv")

        def _run_and_read(voxelspacing):
            if output_file.exists():
                os.remove(str(output_file))

            evaluator = Panoptica_Evaluator(
                expected_input=InputType.SEMANTIC,
                instance_approximator=ConnectedComponentsInstanceApproximator(),
                instance_matcher=NaiveThresholdMatching(),
                instance_metrics=[Metric.DSC, Metric.IOU],
            )
            aggregator = Panoptica_Aggregator(
                evaluator,
                output_file=output_file,
                output_individual_instance_metrics=True,
            )
            aggregator.evaluate(b, a, "vs_test", voxelspacing=voxelspacing)

            with open(str(output_file), "r", encoding="utf8", newline="") as f:
                rows = list(csv.reader(f, delimiter="\t"))
            return rows

        try:
            for voxelspacing in [(1.0, 1.0), (2.0, 3.0)]:
                expected_volume = voxels_per_instance * float(np.prod(voxelspacing))

                rows = _run_and_read(voxelspacing)
                # 1 header + 1 master row + 2 instance rows
                self.assertEqual(
                    len(rows), 4, f"Unexpected row count for {voxelspacing}"
                )

                header = rows[0]
                master_row, inst_0_row, inst_1_row = rows[1], rows[2], rows[3]

                vol_col = next(
                    (
                        i
                        for i, name in enumerate(header)
                        if name.endswith("-instance_volume_ref")
                    ),
                    -1,
                )
                self.assertGreaterEqual(
                    vol_col,
                    0,
                    f"Volume column not found in header for {voxelspacing}",
                )

                count_col = next(
                    (
                        i
                        for i, name in enumerate(header)
                        if name.endswith("-instance_voxel_count_ref")
                    ),
                    -1,
                )
                self.assertGreaterEqual(
                    count_col,
                    0,
                    f"Voxel-count column not found in header for {voxelspacing}",
                )

                # Master row holds the average across instances; per-instance
                # rows hold the individual volume in the same column.
                self.assertAlmostEqual(
                    float(master_row[vol_col]),
                    expected_volume,
                    msg=f"Master avg_volume mismatch for {voxelspacing}",
                )
                self.assertAlmostEqual(
                    float(inst_0_row[vol_col]),
                    expected_volume,
                    msg=f"Instance 0 volume mismatch for {voxelspacing}",
                )
                self.assertAlmostEqual(
                    float(inst_1_row[vol_col]),
                    expected_volume,
                    msg=f"Instance 1 volume mismatch for {voxelspacing}",
                )

                # Voxel counts are spacing-invariant.
                self.assertAlmostEqual(
                    float(master_row[count_col]),
                    float(voxels_per_instance),
                    msg=f"Master avg voxel count mismatch for {voxelspacing}",
                )
                self.assertAlmostEqual(
                    float(inst_0_row[count_col]),
                    float(voxels_per_instance),
                    msg=f"Instance 0 voxel count mismatch for {voxelspacing}",
                )
                self.assertAlmostEqual(
                    float(inst_1_row[count_col]),
                    float(voxels_per_instance),
                    msg=f"Instance 1 voxel count mismatch for {voxelspacing}",
                )
        finally:
            if output_file.exists():
                os.remove(str(output_file))

    def test_aggregator_volume_zero_tp(self):
        # Reference has one instance; prediction is empty -> tp=0.
        # With no matched references the per-instance lists are empty -> NaN default.
        ref = np.zeros([50, 50], dtype=np.uint16)
        pred = np.zeros_like(ref)
        ref[10:20, 10:20] = 1

        output_file = Path(__file__).parent.joinpath("unittest_volume_zerotp.tsv")
        if output_file.exists():
            os.remove(str(output_file))

        evaluator = Panoptica_Evaluator(
            expected_input=InputType.SEMANTIC,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=NaiveThresholdMatching(),
            instance_metrics=[Metric.DSC, Metric.IOU],
        )
        aggregator = Panoptica_Aggregator(
            evaluator,
            output_file=output_file,
            output_individual_instance_metrics=True,
        )
        try:
            aggregator.evaluate(pred, ref, "zero_tp_test")

            with open(str(output_file), "r", encoding="utf8", newline="") as f:
                rows = list(csv.reader(f, delimiter="\t"))

            # 1 header + 1 master row, no per-instance rows because tp=0.
            self.assertEqual(len(rows), 2)
            header = rows[0]
            master_row = rows[1]
            vol_col = next(
                (
                    i
                    for i, name in enumerate(header)
                    if name.endswith("-instance_volume_ref")
                ),
                -1,
            )
            count_col = next(
                (
                    i
                    for i, name in enumerate(header)
                    if name.endswith("-instance_voxel_count_ref")
                ),
                -1,
            )
            self.assertGreaterEqual(vol_col, 0)
            self.assertGreaterEqual(count_col, 0)
            # NaN default for tp=0
            self.assertEqual(master_row[vol_col].lower(), "nan")
            self.assertEqual(master_row[count_col].lower(), "nan")
        finally:
            if output_file.exists():
                os.remove(str(output_file))

    def test_aggregator_volume_partial_match(self):
        # Two reference instances; only one has a matching prediction.
        # The per-instance volume list should contain exactly one entry.
        ref = np.zeros([50, 50], dtype=np.uint16)
        pred = np.zeros_like(ref)
        ref[10:20, 10:20] = 1  # 100 voxels, matched
        ref[30:40, 30:40] = 2  # 100 voxels, unmatched
        pred[10:20, 10:20] = 1

        output_file = Path(__file__).parent.joinpath("unittest_volume_partial.tsv")
        if output_file.exists():
            os.remove(str(output_file))

        evaluator = Panoptica_Evaluator(
            expected_input=InputType.SEMANTIC,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=NaiveThresholdMatching(),
            instance_metrics=[Metric.DSC, Metric.IOU],
        )
        aggregator = Panoptica_Aggregator(
            evaluator,
            output_file=output_file,
            output_individual_instance_metrics=True,
        )
        try:
            aggregator.evaluate(pred, ref, "partial_test", voxelspacing=(2.0, 2.0))

            with open(str(output_file), "r", encoding="utf8", newline="") as f:
                rows = list(csv.reader(f, delimiter="\t"))

            # 1 header + 1 master + 1 per-instance (only the matched instance)
            self.assertEqual(len(rows), 3)
            header = rows[0]
            master_row, inst_row = rows[1], rows[2]
            vol_col = next(
                (
                    i
                    for i, name in enumerate(header)
                    if name.endswith("-instance_volume_ref")
                ),
                -1,
            )
            count_col = next(
                (
                    i
                    for i, name in enumerate(header)
                    if name.endswith("-instance_voxel_count_ref")
                ),
                -1,
            )
            self.assertGreaterEqual(vol_col, 0)
            self.assertGreaterEqual(count_col, 0)
            expected = 100 * 4.0  # 100 voxels * prod((2.0, 2.0))
            self.assertAlmostEqual(float(master_row[vol_col]), expected)
            self.assertAlmostEqual(float(inst_row[vol_col]), expected)
            self.assertAlmostEqual(float(master_row[count_col]), 100.0)
            self.assertAlmostEqual(float(inst_row[count_col]), 100.0)
        finally:
            if output_file.exists():
                os.remove(str(output_file))
