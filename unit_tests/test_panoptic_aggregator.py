# Call 'python -m unittest' on this folder
# coverage run -m unittest
# coverage report
# coverage html
import csv
import json
import os
import unittest
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from panoptica import InputType, Panoptica_Aggregator, Panoptica_Statistic
from panoptica.instance_approximator import ConnectedComponentsInstanceApproximator
from panoptica.instance_matcher import MaximizeMergeMatching, NaiveThresholdMatching
from panoptica.metrics import Metric
from panoptica.panoptica_evaluator import Panoptica_Evaluator
from panoptica.panoptica_result import MetricCouldNotBeComputedException
from panoptica.utils import NonDaemonicPool
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
        # Matched-average columns are NaN, but the unmatched ref is reported
        # as a single FN row.
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

            # 1 header + 1 master + 1 FN row for the unmatched reference.
            self.assertEqual(len(rows), 3)
            header = rows[0]
            master_row = rows[1]
            fn_row = rows[2]
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
            is_matched_col = next(
                (i for i, name in enumerate(header) if name.endswith("-is_matched")),
                -1,
            )
            self.assertGreaterEqual(vol_col, 0)
            self.assertGreaterEqual(count_col, 0)
            self.assertGreaterEqual(is_matched_col, 0)
            # Matched-average columns are NaN because no instance was matched;
            # per the TSV canonical encoding (file_backend_tsv._canonical_tsv_value),
            # NaN/Inf/None all write as an empty cell.
            self.assertEqual(master_row[vol_col], "")
            self.assertEqual(master_row[count_col], "")
            # FN row carries the geometry of the unmatched reference.
            self.assertEqual(int(float(fn_row[is_matched_col])), 0)
            self.assertAlmostEqual(float(fn_row[count_col]), 100.0)
            self.assertAlmostEqual(float(fn_row[vol_col]), 100.0)
        finally:
            if output_file.exists():
                os.remove(str(output_file))

    def test_aggregator_volume_partial_match(self):
        # Two reference instances; only one has a matching prediction.
        # Expect a matched row and an unmatched row, distinguished by is_matched.
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

            # 1 header + 1 master + 1 matched row + 1 unmatched row
            self.assertEqual(len(rows), 4)
            header = rows[0]
            master_row, matched_row, unmatched_row = rows[1], rows[2], rows[3]
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
            is_matched_col = next(
                (i for i, name in enumerate(header) if name.endswith("-is_matched")),
                -1,
            )
            dsc_col = next(
                (i for i, name in enumerate(header) if name.endswith("-sq_dsc")),
                -1,
            )
            self.assertGreaterEqual(vol_col, 0)
            self.assertGreaterEqual(count_col, 0)
            self.assertGreaterEqual(is_matched_col, 0)
            self.assertGreaterEqual(dsc_col, 0)
            expected = 100 * 4.0  # 100 voxels * prod((2.0, 2.0))
            # Master row's volume/count averages reflect only matched refs;
            # is_matched is a per-instance flag and is empty on the master row.
            self.assertAlmostEqual(float(master_row[vol_col]), expected)
            self.assertAlmostEqual(float(master_row[count_col]), 100.0)
            self.assertEqual(master_row[is_matched_col], "")
            # Matched row
            self.assertEqual(int(float(matched_row[is_matched_col])), 1)
            self.assertAlmostEqual(float(matched_row[vol_col]), expected)
            self.assertAlmostEqual(float(matched_row[count_col]), 100.0)
            self.assertNotEqual(matched_row[dsc_col], "")
            # Unmatched row
            self.assertEqual(int(float(unmatched_row[is_matched_col])), 0)
            self.assertAlmostEqual(float(unmatched_row[vol_col]), expected)
            self.assertAlmostEqual(float(unmatched_row[count_col]), 100.0)
            self.assertEqual(unmatched_row[dsc_col], "")
        finally:
            if output_file.exists():
                os.remove(str(output_file))

    def test_aggregator_decision_threshold_rejection_as_unmatched(self):
        # Reference and prediction overlap at IoU == 0.5: the matcher pairs them (matching_threshold 0.5), but decision_threshold 0.8 rejects the pair.
        # The ref must show up as an is_matched=0 row in the per-instance TSV, not silently disappear.
        ref = np.zeros([50, 50], dtype=np.uint16)
        pred = np.zeros_like(ref)
        ref[10:20, 10:20] = 1  # 100 voxels
        pred[10:20, 10:15] = 1  # 50 voxels, fully inside ref → IoU = 50/100 = 0.5

        output_file = Path(__file__).parent.joinpath(
            "unittest_decision_threshold_reject.tsv"
        )
        if output_file.exists():
            os.remove(str(output_file))

        evaluator = Panoptica_Evaluator(
            expected_input=InputType.SEMANTIC,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=NaiveThresholdMatching(),
            instance_metrics=[Metric.DSC, Metric.IOU],
            decision_metric=Metric.IOU,
            decision_threshold=0.8,
        )
        aggregator = Panoptica_Aggregator(
            evaluator,
            output_file=output_file,
            output_individual_instance_metrics=True,
        )
        try:
            aggregator.evaluate(pred, ref, "decision_reject_test")

            with open(str(output_file), "r", encoding="utf8", newline="") as f:
                rows = list(csv.reader(f, delimiter="\t"))

            # 1 header + 1 master + 1 unmatched row (no matched rows)
            self.assertEqual(len(rows), 3)
            header = rows[0]
            master_row, unmatched_row = rows[1], rows[2]
            is_matched_col = next(
                (i for i, name in enumerate(header) if name.endswith("-is_matched")),
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
            dsc_col = next(
                (i for i, name in enumerate(header) if name.endswith("-sq_dsc")),
                -1,
            )
            self.assertGreaterEqual(is_matched_col, 0)
            self.assertGreaterEqual(count_col, 0)
            self.assertGreaterEqual(dsc_col, 0)
            # Master row: is_matched empty, no matched instances contributed to it.
            self.assertEqual(master_row[is_matched_col], "")
            # Unmatched row: is_matched=0, voxel count matches the rejected ref,
            # per-instance DSC column empty.
            self.assertEqual(int(float(unmatched_row[is_matched_col])), 0)
            self.assertAlmostEqual(float(unmatched_row[count_col]), 100.0)
            self.assertEqual(unmatched_row[dsc_col], "")
        finally:
            if output_file.exists():
                os.remove(str(output_file))

    def test_aggregator_all_matched_no_unmatched_rows(self):
        # Regression guard: if every reference is matched, only matched rows are emitted.
        ref = np.zeros([50, 50], dtype=np.uint16)
        pred = np.zeros_like(ref)
        ref[10:20, 10:20] = 1
        ref[30:40, 30:40] = 2
        pred[10:20, 10:20] = 1
        pred[30:40, 30:40] = 2

        output_file = Path(__file__).parent.joinpath("unittest_all_matched.tsv")
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
            aggregator.evaluate(pred, ref, "all_matched_test")

            with open(str(output_file), "r", encoding="utf8", newline="") as f:
                rows = list(csv.reader(f, delimiter="\t"))

            # 1 header + 1 master + 2 matched rows, no FN rows
            self.assertEqual(len(rows), 4)
            header = rows[0]
            is_matched_col = next(
                (i for i, name in enumerate(header) if name.endswith("-is_matched")),
                -1,
            )
            self.assertGreaterEqual(is_matched_col, 0)
            self.assertEqual(int(float(rows[2][is_matched_col])), 1)
            self.assertEqual(int(float(rows[3][is_matched_col])), 1)
        finally:
            if output_file.exists():
                os.remove(str(output_file))


class Test_Panoptica_Aggregator_Init_Errors(unittest.TestCase):
    """`Panoptica_Aggregator.__init__` should not leave a buffer temp file
    on disk if it raises before reaching the `atexit.register` step."""

    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"

    def test_no_tempfile_leak_on_autc_misconfiguration(self):
        # is_autc=True without threshold_step_size raises early. Patch
        # NamedTemporaryFile to verify it's never reached — equivalently,
        # any temp file that *is* created during a failing init would leak.
        from unittest.mock import patch

        evaluator = Panoptica_Evaluator(
            expected_input=InputType.SEMANTIC,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=NaiveThresholdMatching(),
        )
        with patch(
            "panoptica.panoptica_aggregator.NamedTemporaryFile"
        ) as mock_tempfile:
            with self.assertRaises(ValueError):
                Panoptica_Aggregator(
                    evaluator,
                    output_file=Path(__file__).parent.joinpath(
                        "unittest_init_leak.jsonl"
                    ),
                    is_autc=True,
                    threshold_step_size=None,
                )
            mock_tempfile.assert_not_called()

    def test_autc_with_individual_instance_metrics_raises_at_init(self):
        # PanopticaAUTCResult.to_dict(output_individual_instance_metrics=True)
        # raises NotImplementedError. The aggregator must reject this combination
        # up front rather than crashing on the first evaluate() call.
        from unittest.mock import patch

        evaluator = Panoptica_Evaluator(
            expected_input=InputType.SEMANTIC,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=NaiveThresholdMatching(),
        )
        with patch(
            "panoptica.panoptica_aggregator.NamedTemporaryFile"
        ) as mock_tempfile:
            with self.assertRaisesRegex(
                ValueError,
                "output_individual_instance_metrics is not supported with is_autc=True",
            ):
                Panoptica_Aggregator(
                    evaluator,
                    output_file=Path(__file__).parent.joinpath(
                        "unittest_autc_instance_guard.jsonl"
                    ),
                    is_autc=True,
                    threshold_step_size=0.5,
                    output_individual_instance_metrics=True,
                )
            mock_tempfile.assert_not_called()

    def test_no_tempfile_leak_on_missing_output_directory(self):
        from unittest.mock import patch

        evaluator = Panoptica_Evaluator(
            expected_input=InputType.SEMANTIC,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=NaiveThresholdMatching(),
        )
        bad_path = Path("/nonexistent_dir_for_test_leak_47c92/out.jsonl")
        with patch(
            "panoptica.panoptica_aggregator.NamedTemporaryFile"
        ) as mock_tempfile:
            with self.assertRaises(FileNotFoundError):
                Panoptica_Aggregator(evaluator, output_file=bad_path)
            mock_tempfile.assert_not_called()


class Test_Panoptica_Aggregator_Init_Locking_And_Lazy_Load(unittest.TestCase):
    """`Panoptica_Aggregator.__init__` should hold ``filelock`` while it
    initialises the backend, and should skip the full existing-subjects
    scan when ``continue_file=False`` (since the result is discarded
    anyway)."""

    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"
        self.output_file = Path(__file__).parent.joinpath(
            "unittest_aggregator_init.jsonl"
        )
        if self.output_file.exists():
            os.remove(self.output_file)

    def tearDown(self) -> None:
        if self.output_file.exists():
            os.remove(self.output_file)

    def _make_evaluator(self) -> Panoptica_Evaluator:
        return Panoptica_Evaluator(
            expected_input=InputType.SEMANTIC,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=NaiveThresholdMatching(),
        )

    def _patch_recording_filelocks(self, order: list[str]):
        """Patch the aggregator's FileLock with a factory that records, per lock
        path, when each lock is entered/exited. Returns the patch context manager."""
        from unittest.mock import MagicMock, patch

        def _factory(lock_path, *args, **kwargs):
            name = Path(str(lock_path)).name
            m = MagicMock()
            m.__enter__.side_effect = lambda n=name: order.append(f"enter:{n}")
            m.__exit__.side_effect = lambda *a, n=name, **kw: order.append(f"exit:{n}")
            return m

        return patch(
            "panoptica.panoptica_aggregator.FileLock", side_effect=_factory
        )

    def test_prepare_for_append_called_under_filelock(self):
        from unittest.mock import patch

        evaluator = self._make_evaluator()

        # The output-file FileLock must be held across prepare_for_append (file init).
        order: list[str] = []
        out_lock = self.output_file.name + ".lock"

        def _record_prepare(*args, **kwargs):
            order.append("prepare_for_append")
            return []

        with self._patch_recording_filelocks(order):
            with patch(
                "panoptica.utils.file_backend_jsonl.JSONLBackend.prepare_for_append",
                side_effect=_record_prepare,
            ):
                Panoptica_Aggregator(evaluator, output_file=self.output_file)

        # output FileLock entered, prepare_for_append ran, then it exited.
        self.assertEqual(
            order[:3], [f"enter:{out_lock}", "prepare_for_append", f"exit:{out_lock}"]
        )

    def test_append_subject_called_under_filelock(self):
        from unittest.mock import patch

        evaluator = self._make_evaluator()

        # The output-file FileLock must also guard the per-subject append (write).
        order: list[str] = []
        out_lock = self.output_file.name + ".lock"

        def _record_append(*args, **kwargs):
            order.append("append_subject")

        a = np.zeros([50, 50], dtype=np.uint16)
        b = a.copy()
        a[10:20, 10:20] = 1
        b[10:20, 10:20] = 1

        # Construct the aggregator inside the patch so its instance FileLocks are the
        # recording mocks that evaluate() then reuses.
        with self._patch_recording_filelocks(order):
            with patch(
                "panoptica.utils.file_backend_jsonl.JSONLBackend.append_subject",
                side_effect=_record_append,
            ):
                aggregator = Panoptica_Aggregator(
                    evaluator, output_file=self.output_file
                )
                aggregator.evaluate(b, a, subject_name="subj_write")

        # append_subject ran wrapped by the output FileLock's enter/exit.
        self.assertIn("append_subject", order)
        i = order.index("append_subject")
        self.assertEqual(order[i - 1], f"enter:{out_lock}")
        self.assertEqual(order[i + 1], f"exit:{out_lock}")

    def test_continue_file_false_skips_existing_scan(self):
        # Seed a file with one subject so that — if collect_existing were
        # honoured wrongly — the call would return ["subj_seed"].
        a = np.zeros([50, 50], dtype=np.uint16)
        b = a.copy()
        a[10:20, 10:20] = 1
        b[10:20, 10:20] = 1

        evaluator = self._make_evaluator()
        Panoptica_Aggregator(
            evaluator, output_file=self.output_file, continue_file=True
        ).evaluate(b, a, "subj_seed")

        from unittest.mock import patch

        seen_kwargs: dict = {}

        original = __import__(
            "panoptica.utils.file_backend_jsonl",
            fromlist=["JSONLBackend"],
        ).JSONLBackend.prepare_for_append

        def _spy(self, class_group_names, evaluation_metrics, collect_existing=True):
            seen_kwargs["collect_existing"] = collect_existing
            return original(
                self,
                class_group_names,
                evaluation_metrics,
                collect_existing=collect_existing,
            )

        with patch(
            "panoptica.utils.file_backend_jsonl.JSONLBackend.prepare_for_append",
            _spy,
        ):
            Panoptica_Aggregator(
                evaluator, output_file=self.output_file, continue_file=False
            )

        self.assertEqual(seen_kwargs["collect_existing"], False)

    def test_continue_file_true_collects_existing(self):
        # Symmetric assertion: continue_file=True must forward collect_existing=True
        # (default) so the dedup buffer is seeded with prior subjects.
        a = np.zeros([50, 50], dtype=np.uint16)
        b = a.copy()
        a[10:20, 10:20] = 1
        b[10:20, 10:20] = 1

        evaluator = self._make_evaluator()
        Panoptica_Aggregator(
            evaluator, output_file=self.output_file, continue_file=True
        ).evaluate(b, a, "subj_seed")

        from unittest.mock import patch

        seen_kwargs: dict = {}
        original = __import__(
            "panoptica.utils.file_backend_jsonl",
            fromlist=["JSONLBackend"],
        ).JSONLBackend.prepare_for_append

        def _spy(self, class_group_names, evaluation_metrics, collect_existing=True):
            seen_kwargs["collect_existing"] = collect_existing
            return original(
                self,
                class_group_names,
                evaluation_metrics,
                collect_existing=collect_existing,
            )

        with patch(
            "panoptica.utils.file_backend_jsonl.JSONLBackend.prepare_for_append",
            _spy,
        ):
            Panoptica_Aggregator(
                evaluator, output_file=self.output_file, continue_file=True
            )

        self.assertEqual(seen_kwargs["collect_existing"], True)


class Test_Panoptica_Aggregator_Parallel_JSONL(unittest.TestCase):
    """Parallel evaluation against the JSONL backend. Mirrors the parallel
    paths exercised for TSV via `Test_Example_Scripts.test_example_scripts_pool`
    / `test_example_scripts_future` but uses synthetic 50×50 arrays so the
    suite stays fast and hermetic. Closes a coverage hole that opened when
    the default backend on this branch flipped from TSV to JSONL: without
    these, the `filelock`-guarded init and append paths only ran serially
    or under mocks in the JSONL configuration."""

    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"
        self.output_file = Path(__file__).parent.joinpath(
            "unittest_parallel_jsonl.jsonl"
        )
        if self.output_file.exists():
            os.remove(self.output_file)

    def tearDown(self) -> None:
        if self.output_file.exists():
            os.remove(self.output_file)

    def _make_inputs(self):
        a = np.zeros([50, 50], dtype=np.uint16)
        b = a.copy()
        a[20:40, 10:20] = 1
        b[20:35, 10:20] = 2
        return a, b

    def _make_aggregator(self) -> Panoptica_Aggregator:
        evaluator = Panoptica_Evaluator(
            expected_input=InputType.SEMANTIC,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=NaiveThresholdMatching(),
        )
        return Panoptica_Aggregator(evaluator, output_file=self.output_file)

    def _assert_jsonl_round_trip(self, expected_subjects: list[str]) -> None:
        # Each line must parse standalone — a torn / interleaved write
        # under concurrency would surface here as a JSONDecodeError.
        with open(self.output_file, "r", encoding="utf8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        written = [r["subject_name"] for r in records]
        # No losses, no duplicates, exactly the submitted set.
        self.assertEqual(sorted(written), sorted(expected_subjects))
        self.assertEqual(len(written), len(set(written)))

        # Statistic round-trip carries the same subjects.
        stat = Panoptica_Statistic.from_file(self.output_file, verbose=False)
        self.assertEqual(sorted(stat.subjectnames), sorted(expected_subjects))

        # Re-opening with continue_file=True picks the existing subjects up
        # via `prepare_for_append`'s collect_existing branch, so a duplicate
        # evaluate is a no-op (file size unchanged).
        evaluator = Panoptica_Evaluator(
            expected_input=InputType.SEMANTIC,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=NaiveThresholdMatching(),
        )
        agg2 = Panoptica_Aggregator(
            evaluator, output_file=self.output_file, continue_file=True
        )
        a, b = self._make_inputs()
        size_before = self.output_file.stat().st_size
        agg2.evaluate(b, a, expected_subjects[0])
        self.assertEqual(self.output_file.stat().st_size, size_before)

    def test_parallel_jsonl_with_process_pool_executor(self):
        # Mirrors examples/example_spine_statistics.py:56-69 (parallel_opt
        # "future"), against JSONL instead of TSV.
        aggregator = self._make_aggregator()
        a, b = self._make_inputs()
        subjects = [f"subj_future_{i}" for i in range(6)]

        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(aggregator.evaluate, b, a, sn) for sn in subjects
            ]
            for fut in futures:
                fut.result()

        self._assert_jsonl_round_trip(subjects)

    def test_parallel_jsonl_with_nondaemonic_pool(self):
        # Mirrors examples/example_spine_statistics.py:39-47 (parallel_opt
        # "pool"), against JSONL instead of TSV.
        aggregator = self._make_aggregator()
        a, b = self._make_inputs()
        subjects = [f"subj_pool_{i}" for i in range(6)]
        args = [(b, a, sn) for sn in subjects]

        with NonDaemonicPool() as pool:
            pool.starmap(aggregator.evaluate, args)

        self._assert_jsonl_round_trip(subjects)
