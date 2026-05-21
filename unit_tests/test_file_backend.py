import atexit
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from panoptica import InputType, Panoptica_Aggregator, Panoptica_Statistic
from panoptica.instance_approximator import ConnectedComponentsInstanceApproximator
from panoptica.instance_matcher import NaiveThresholdMatching
from panoptica.panoptica_evaluator import Panoptica_Evaluator
from panoptica.utils.file_backend import (
    JSONLBackend,
    TSVBackend,
    get_backend,
)
from panoptica.utils.file_backend_jsonl import _canonical_jsonl_value
from panoptica.utils.file_backend_tsv import _canonical_tsv_value

# Write artifacts to an isolated tempdir so a crash in tearDown can't leave
# .tsv/.jsonl debris in the source tree.
_TMP_DIR = Path(tempfile.mkdtemp(prefix="panoptica_test_file_backend_"))
atexit.register(shutil.rmtree, _TMP_DIR, ignore_errors=True)


def _make_simple_evaluator() -> Panoptica_Evaluator:
    return Panoptica_Evaluator(
        expected_input=InputType.SEMANTIC,
        instance_approximator=ConnectedComponentsInstanceApproximator(),
        instance_matcher=NaiveThresholdMatching(),
    )


def _two_instance_arrays() -> tuple[np.ndarray, np.ndarray]:
    a = np.zeros([50, 50], dtype=np.uint16)
    b = a.copy()
    a[10:20, 10:20] = 1
    b[10:20, 10:20] = 1
    a[30:40, 30:40] = 2
    b[30:40, 30:40] = 2
    return a, b


class Test_TSVBackend_Direct(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"
        self.path = _TMP_DIR.joinpath("unittest_backend_direct.tsv")
        if self.path.exists():
            os.remove(self.path)

    def tearDown(self) -> None:
        if self.path.exists():
            os.remove(self.path)

    def test_prepare_for_append_creates_file_with_header(self):
        backend = TSVBackend(self.path)
        existing = backend.prepare_for_append(["liver"], ["dice", "tp"])
        self.assertEqual(existing, [])
        self.assertTrue(self.path.exists())
        with open(self.path, "r", encoding="utf8") as f:
            self.assertEqual(f.read().strip(), "subject_name\tliver-dice\tliver-tp")

    def test_prepare_for_append_raises_on_header_mismatch(self):
        backend = TSVBackend(self.path)
        backend.prepare_for_append(["liver"], ["dice", "tp"])
        backend2 = TSVBackend(self.path)
        with self.assertRaisesRegex(ValueError, "Header does not match"):
            backend2.prepare_for_append(["liver"], ["dice", "iou"])

    def test_get_backend_dispatches_by_extension(self):
        self.assertIsInstance(get_backend(self.path), TSVBackend)
        self.assertIsInstance(
            get_backend(_TMP_DIR.joinpath("dummy.jsonl")), JSONLBackend
        )

    def test_get_backend_unknown_extension_raises(self):
        with self.assertRaises(ValueError):
            get_backend(_TMP_DIR.joinpath("dummy.xyz"))

    def test_get_backend_extension_case_insensitive(self):
        self.assertIsInstance(get_backend(_TMP_DIR.joinpath("dummy.TSV")), TSVBackend)
        self.assertIsInstance(
            get_backend(_TMP_DIR.joinpath("dummy.JSONL")), JSONLBackend
        )

    def test_write_full_then_load_raw_roundtrip(self):
        backend = TSVBackend(self.path)
        subj_names = ["subj_a", "subj_b"]
        value_dict = {
            "liver": {"dice": [0.9, 0.8], "tp": [5.0, None]},
            "spleen": {"dice": [None, 0.7], "tp": [3.0, 2.0]},
        }
        backend.write_full(subj_names, value_dict, ["liver", "spleen"], ["dice", "tp"])

        loaded_subj, loaded_dict = backend.load_raw(verbose=False)
        self.assertEqual(loaded_subj, subj_names)
        self.assertEqual(loaded_dict["liver"]["dice"], [0.9, 0.8])
        self.assertEqual(loaded_dict["liver"]["tp"], [5.0, None])
        self.assertEqual(loaded_dict["spleen"]["dice"], [None, 0.7])
        self.assertEqual(loaded_dict["spleen"]["tp"], [3.0, 2.0])

    def test_load_raw_header_only_returns_empty(self):
        # Header-only TSV
        backend = TSVBackend(self.path)
        backend.prepare_for_append(["liver"], ["dice", "tp"])
        loaded_subj, loaded_dict = backend.load_raw(verbose=False)
        self.assertEqual(loaded_subj, [])
        self.assertEqual(loaded_dict, {})

    def test_load_raw_zero_byte_file_returns_empty(self):
        self.path.touch()
        self.assertEqual(self.path.stat().st_size, 0)
        backend = TSVBackend(self.path)
        loaded_subj, loaded_dict = backend.load_raw(verbose=False)
        self.assertEqual(loaded_subj, [])
        self.assertEqual(loaded_dict, {})


class Test_JSONLBackend_Direct(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"
        self.path = _TMP_DIR.joinpath("unittest_backend_direct.jsonl")
        if self.path.exists():
            os.remove(self.path)

    def tearDown(self) -> None:
        if self.path.exists():
            os.remove(self.path)

    def test_prepare_for_append_creates_empty_file(self):
        backend = JSONLBackend(self.path)
        existing = backend.prepare_for_append(["liver"], ["dice", "tp"])
        self.assertEqual(existing, [])
        self.assertTrue(self.path.exists())
        self.assertEqual(self.path.stat().st_size, 0)

    def test_prepare_for_append_returns_existing_subjects(self):
        backend = JSONLBackend(self.path)
        backend.prepare_for_append(["liver"], ["dice", "tp"])
        with open(self.path, "a", encoding="utf8") as f:
            f.write(
                json.dumps(
                    {
                        "subject_name": "subj_a",
                        "groups": {"liver": {"dice": 0.9, "tp": 5.0}},
                    }
                )
                + "\n"
            )
            f.write(
                json.dumps(
                    {
                        "subject_name": "subj_b",
                        "groups": {
                            "liver": {
                                "dice": 0.8,
                                "tp": 4.0,
                                "reference_instances": [{"sq_dice": 0.95}],
                            }
                        },
                    }
                )
                + "\n"
            )
        backend2 = JSONLBackend(self.path)
        existing = backend2.prepare_for_append(["liver"], ["dice", "tp"])
        self.assertEqual(existing, ["subj_a", "subj_b", "subj_b-liver_inst_0"])

    def test_prepare_for_append_raises_on_schema_mismatch_groups(self):
        backend = JSONLBackend(self.path)
        backend.prepare_for_append(["liver"], ["dice", "tp"])
        with open(self.path, "a", encoding="utf8") as f:
            f.write(
                json.dumps(
                    {
                        "subject_name": "subj_a",
                        "groups": {"liver": {"dice": 0.9, "tp": 5.0}},
                    }
                )
                + "\n"
            )
        backend2 = JSONLBackend(self.path)
        with self.assertRaisesRegex(
            ValueError, "schema of existing file does not match"
        ):
            backend2.prepare_for_append(["liver", "spleen"], ["dice", "tp"])

    def test_prepare_for_append_raises_on_schema_mismatch_metrics(self):
        backend = JSONLBackend(self.path)
        backend.prepare_for_append(["liver"], ["dice", "tp"])
        with open(self.path, "a", encoding="utf8") as f:
            f.write(
                json.dumps(
                    {
                        "subject_name": "subj_a",
                        "groups": {"liver": {"dice": 0.9, "tp": 5.0}},
                    }
                )
                + "\n"
            )
        backend2 = JSONLBackend(self.path)
        with self.assertRaisesRegex(
            ValueError, "schema of existing file does not match"
        ):
            backend2.prepare_for_append(["liver"], ["dice", "iou"])

    def test_aggregator_writes_one_jsonl_line_per_subject(self):
        evaluator = _make_simple_evaluator()
        aggregator = Panoptica_Aggregator(evaluator, output_file=self.path)
        a, b = _two_instance_arrays()
        aggregator.evaluate(b, a, "test_subject_0")
        aggregator.evaluate(b, a, "test_subject_1")

        with open(self.path, "r", encoding="utf8") as f:
            lines = [line for line in f if line.strip()]
        self.assertEqual(len(lines), 2)
        rec0 = json.loads(lines[0])
        self.assertEqual(rec0["subject_name"], "test_subject_0")
        self.assertIn("ungrouped", rec0["groups"])
        self.assertIn("tp", rec0["groups"]["ungrouped"])
        # No instances key when output_individual_instance_metrics=False
        self.assertNotIn("reference_instances", rec0["groups"]["ungrouped"])

    def test_aggregator_writes_instances_when_enabled(self):
        evaluator = _make_simple_evaluator()
        aggregator = Panoptica_Aggregator(
            evaluator,
            output_file=self.path,
            output_individual_instance_metrics=True,
        )
        a, b = _two_instance_arrays()
        aggregator.evaluate(b, a, "test_subject")

        with open(self.path, "r", encoding="utf8") as f:
            line = f.readline().strip()
        rec = json.loads(line)
        # Two matched instances → instances list of length 2
        self.assertIn("reference_instances", rec["groups"]["ungrouped"])
        self.assertEqual(len(rec["groups"]["ungrouped"]["reference_instances"]), 2)

    def test_aggregator_records_unmatched_instance(self):
        evaluator = _make_simple_evaluator()
        aggregator = Panoptica_Aggregator(
            evaluator,
            output_file=self.path,
            output_individual_instance_metrics=True,
        )
        ref = np.zeros([50, 50], dtype=np.uint16)
        pred = np.zeros_like(ref)
        ref[10:20, 10:20] = 1
        ref[30:40, 30:40] = 2
        pred[10:20, 10:20] = 1
        aggregator.evaluate(pred, ref, "partial_subject")

        with open(self.path, "r", encoding="utf8") as f:
            rec = json.loads(f.readline().strip())
        instances = rec["groups"]["ungrouped"]["reference_instances"]
        self.assertEqual(len(instances), 2)
        matched_flags = [inst.get("is_matched") for inst in instances]
        self.assertIn(1.0, matched_flags)
        self.assertIn(0.0, matched_flags)

    def test_load_raw_flattens_instances_into_synthetic_subj_names(self):
        evaluator = _make_simple_evaluator()
        aggregator = Panoptica_Aggregator(
            evaluator,
            output_file=self.path,
            output_individual_instance_metrics=True,
        )
        a, b = _two_instance_arrays()
        aggregator.evaluate(b, a, "test_subject")

        backend = JSONLBackend(self.path)
        subj_names, value_dict = backend.load_raw(verbose=False)
        # Master row + 2 instance rows
        self.assertEqual(len(subj_names), 3)
        self.assertEqual(subj_names[0], "test_subject")
        self.assertEqual(subj_names[1], "test_subject-ungrouped_inst_0")
        self.assertEqual(subj_names[2], "test_subject-ungrouped_inst_1")
        # Master row has a tp count, instance rows have None (since 'tp' is a
        # summary metric, not an instance-level key)
        self.assertIsNotNone(value_dict["ungrouped"]["tp"][0])
        self.assertIsNone(value_dict["ungrouped"]["tp"][1])
        self.assertIsNone(value_dict["ungrouped"]["tp"][2])

    def test_load_raw_raises_when_record_missing_group(self):
        with open(self.path, "w", encoding="utf8") as f:
            f.write(
                json.dumps(
                    {
                        "subject_name": "subj_a",
                        "groups": {
                            "liver": {"dice": 0.9, "tp": 5.0},
                            "spleen": {"dice": 0.8, "tp": 4.0},
                        },
                    }
                )
                + "\n"
            )
            f.write(
                json.dumps(
                    {
                        "subject_name": "subj_b",
                        "groups": {"liver": {"dice": 0.7, "tp": 3.0}},
                    }
                )
                + "\n"
            )
        backend = JSONLBackend(self.path)
        with self.assertRaisesRegex(ValueError, r"record 1.*missing group.*spleen"):
            backend.load_raw(verbose=False)

    def test_load_raw_missing_value_round_trips_to_none(self):
        # Write a record with an explicit JSON null and verify it loads as None
        with open(self.path, "w", encoding="utf8") as f:
            f.write(
                json.dumps(
                    {
                        "subject_name": "subj_a",
                        "groups": {"ungrouped": {"dice": None, "tp": 5.0}},
                    }
                )
                + "\n"
            )
        backend = JSONLBackend(self.path)
        subj_names, value_dict = backend.load_raw(verbose=False)
        self.assertEqual(subj_names, ["subj_a"])
        self.assertEqual(value_dict["ungrouped"]["dice"], [None])
        self.assertEqual(value_dict["ungrouped"]["tp"], [5.0])


class Test_Roundtrip_TSV_JSONL(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"
        self.before_tsv = _TMP_DIR.joinpath("unittest_roundtrip_before.tsv")
        self.middle_jsonl = _TMP_DIR.joinpath("unittest_roundtrip_middle.jsonl")
        self.after_tsv = _TMP_DIR.joinpath("unittest_roundtrip_after.tsv")
        self.before_jsonl = _TMP_DIR.joinpath("unittest_roundtrip_before.jsonl")
        self.middle_tsv = _TMP_DIR.joinpath("unittest_roundtrip_middle.tsv")
        self.after_jsonl = _TMP_DIR.joinpath("unittest_roundtrip_after.jsonl")
        for p in (
            self.before_tsv,
            self.middle_jsonl,
            self.after_tsv,
            self.before_jsonl,
            self.middle_tsv,
            self.after_jsonl,
        ):
            if p.exists():
                os.remove(p)

    def tearDown(self) -> None:
        for p in (
            self.before_tsv,
            self.middle_jsonl,
            self.after_tsv,
            self.before_jsonl,
            self.middle_tsv,
            self.after_jsonl,
        ):
            if p.exists():
                os.remove(p)

    def test_tsv_jsonl_tsv_byte_identical(self):
        evaluator = _make_simple_evaluator()
        aggregator = Panoptica_Aggregator(
            evaluator,
            output_file=self.before_tsv,
            output_individual_instance_metrics=True,
        )
        a, b = _two_instance_arrays()
        aggregator.evaluate(b, a, "subj_a")

        stat = Panoptica_Statistic.from_file(self.before_tsv, verbose=False)
        stat.to_file(self.middle_jsonl)

        stat2 = Panoptica_Statistic.from_file(self.middle_jsonl, verbose=False)
        stat2.to_file(self.after_tsv)

        self.assertEqual(
            self.before_tsv.read_bytes(),
            self.after_tsv.read_bytes(),
            "TSV → JSONL → TSV roundtrip must be byte-identical",
        )

    def test_jsonl_tsv_jsonl_byte_identical(self):
        evaluator = _make_simple_evaluator()
        aggregator = Panoptica_Aggregator(
            evaluator,
            output_file=self.before_jsonl,
            output_individual_instance_metrics=True,
        )
        a, b = _two_instance_arrays()
        aggregator.evaluate(b, a, "subj_a")

        stat = Panoptica_Statistic.from_file(self.before_jsonl, verbose=False)
        stat.to_file(self.middle_tsv)

        stat2 = Panoptica_Statistic.from_file(self.middle_tsv, verbose=False)
        stat2.to_file(self.after_jsonl)

        self.assertEqual(
            self.before_jsonl.read_bytes(),
            self.after_jsonl.read_bytes(),
            "JSONL → TSV → JSONL roundtrip must be byte-identical",
        )

    def test_continue_file_jsonl_blocks_duplicate_subject(self):
        evaluator = _make_simple_evaluator()
        aggregator = Panoptica_Aggregator(
            evaluator, output_file=self.before_jsonl, continue_file=False
        )
        a, b = _two_instance_arrays()
        aggregator.evaluate(b, a, "subj_a")

        # New aggregator with continue_file=True picks up the existing subject
        aggregator2 = Panoptica_Aggregator(
            evaluator, output_file=self.before_jsonl, continue_file=True
        )
        # evaluate() on the same subject_name prints a warning and returns;
        # the file is unchanged.
        size_before = self.before_jsonl.stat().st_size
        aggregator2.evaluate(b, a, "subj_a")
        size_after = self.before_jsonl.stat().st_size
        self.assertEqual(size_before, size_after)


class Test_Statistic_File_Suffix_Defaulting(unittest.TestCase):
    """`Panoptica_Statistic.from_file` / `to_file` should auto-append
    ``.{file_type}`` (default ``jsonl``) when the given path has no suffix,
    mirroring `Panoptica_Aggregator.__init__`."""

    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"
        self.stem = _TMP_DIR.joinpath("unittest_suffix_default")
        self.candidates = [
            self.stem,
            self.stem.with_suffix(".jsonl"),
            self.stem.with_suffix(".tsv"),
        ]
        for p in self.candidates:
            if p.exists():
                os.remove(p)

    def tearDown(self) -> None:
        for p in self.candidates:
            if p.exists():
                os.remove(p)

    def _seed_jsonl(self) -> Panoptica_Statistic:
        evaluator = _make_simple_evaluator()
        aggregator = Panoptica_Aggregator(
            evaluator,
            output_file=self.stem.with_suffix(".jsonl"),
        )
        a, b = _two_instance_arrays()
        aggregator.evaluate(b, a, "subj_a")
        return Panoptica_Statistic.from_file(
            self.stem.with_suffix(".jsonl"), verbose=False
        )

    def test_from_file_appends_jsonl_by_default(self):
        self._seed_jsonl()
        # Pass the bare stem; should resolve to .jsonl
        stat = Panoptica_Statistic.from_file(self.stem, verbose=False)
        self.assertIn("subj_a", stat.subjectnames)

    def test_from_file_appends_explicit_file_type(self):
        # Write a TSV and load it via stem + file_type="tsv"
        stat = self._seed_jsonl()
        stat.to_file(self.stem.with_suffix(".tsv"))
        stat2 = Panoptica_Statistic.from_file(self.stem, verbose=False, file_type="tsv")
        self.assertIn("subj_a", stat2.subjectnames)

    def test_to_file_appends_jsonl_by_default(self):
        stat = self._seed_jsonl()
        stat.to_file(self.stem)
        self.assertTrue(self.stem.with_suffix(".jsonl").exists())
        self.assertFalse(self.stem.with_suffix(".tsv").exists())

    def test_explicit_suffix_beats_file_type_kwarg(self):
        # Explicit .tsv suffix on the path must win over file_type="jsonl"
        stat = self._seed_jsonl()
        stat.to_file(self.stem.with_suffix(".tsv"), file_type="jsonl")
        self.assertTrue(self.stem.with_suffix(".tsv").exists())


class Test_Canonical_Value_Numpy_Scalars(unittest.TestCase):
    """Both canonicalizers must accept NumPy scalar dtypes — not just
    built-in ``int``/``float``. Otherwise ``np.float32`` / ``np.int*``
    would either crash ``json.dumps`` (JSONL) or escape as the literal
    string ``"nan"`` / ``"inf"`` via ``csv.writer`` (TSV), breaking the
    byte-identical roundtrip."""

    def test_canonical_jsonl_value_numpy_scalars(self):
        self.assertIsNone(_canonical_jsonl_value(np.float32(np.nan)))
        self.assertIsNone(_canonical_jsonl_value(np.float64(np.inf)))
        self.assertEqual(_canonical_jsonl_value(np.int32(5)), 5.0)
        self.assertEqual(_canonical_jsonl_value(np.int64(7)), 7.0)
        self.assertEqual(_canonical_jsonl_value(np.float32(1.5)), 1.5)
        # And the canonicalised values must be JSON-serialisable.
        json.dumps(
            [
                _canonical_jsonl_value(np.float32(np.nan)),
                _canonical_jsonl_value(np.float64(np.inf)),
                _canonical_jsonl_value(np.int32(5)),
                _canonical_jsonl_value(np.int64(7)),
                _canonical_jsonl_value(np.float32(1.5)),
            ]
        )

    def test_canonical_tsv_value_numpy_scalars(self):
        self.assertEqual(_canonical_tsv_value(np.float32(np.nan)), "")
        self.assertEqual(_canonical_tsv_value(np.float64(np.inf)), "")
        self.assertEqual(_canonical_tsv_value(np.int32(5)), 5.0)
        self.assertEqual(_canonical_tsv_value(np.int64(7)), 7.0)
        self.assertEqual(_canonical_tsv_value(np.float32(1.5)), 1.5)

    def test_canonical_values_filter_negative_infinity(self):
        # Both +inf and -inf must canonicalise to the missing-value
        # representation. The JSONL serializer in particular must not emit
        # the literal "Infinity" / "-Infinity", since neither is valid JSON
        # and most non-Python parsers (JS JSON.parse, jq) reject them.
        self.assertIsNone(_canonical_jsonl_value(-np.inf))
        self.assertIsNone(_canonical_jsonl_value(np.float64(-np.inf)))
        self.assertIsNone(_canonical_jsonl_value(np.float32(-np.inf)))
        self.assertEqual(_canonical_tsv_value(-np.inf), "")
        self.assertEqual(_canonical_tsv_value(np.float32(-np.inf)), "")


class Test_JSONL_Schema_Drift(unittest.TestCase):
    """`prepare_for_append` must reject mid-file schema drift, not only a
    mismatch on the very first record (which a hand-edit or concatenation
    could easily slip past)."""

    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"
        self.path = _TMP_DIR.joinpath("unittest_jsonl_drift.jsonl")
        if self.path.exists():
            os.remove(self.path)

    def tearDown(self) -> None:
        if self.path.exists():
            os.remove(self.path)

    def test_prepare_for_append_rejects_midfile_group_drift(self):
        # First record matches expected schema; second introduces a new group.
        with open(self.path, "w", encoding="utf8") as f:
            f.write(
                json.dumps(
                    {
                        "subject_name": "subj_a",
                        "groups": {"liver": {"dice": 0.9, "tp": 5.0}},
                    }
                )
                + "\n"
            )
            f.write(
                json.dumps(
                    {
                        "subject_name": "subj_b",
                        "groups": {
                            "liver": {"dice": 0.8, "tp": 4.0},
                            "spleen": {"dice": 0.7, "tp": 3.0},
                        },
                    }
                )
                + "\n"
            )
        backend = JSONLBackend(self.path)
        with self.assertRaisesRegex(
            ValueError, "schema of existing file does not match"
        ):
            backend.prepare_for_append(["liver"], ["dice", "tp"])

    def test_prepare_for_append_raises_with_line_number_on_malformed_line(self):
        # Hand-write a file with valid line 1 and a corrupt line 2.
        with open(self.path, "w", encoding="utf8") as f:
            f.write(
                json.dumps(
                    {
                        "subject_name": "subj_a",
                        "groups": {"liver": {"dice": 0.9, "tp": 5.0}},
                    }
                )
                + "\n"
            )
            f.write("{not valid json\n")
        backend = JSONLBackend(self.path)
        with self.assertRaisesRegex(ValueError, "malformed JSON on line 2"):
            backend.prepare_for_append(["liver"], ["dice", "tp"])

    def test_prepare_for_append_rejects_midfile_metric_drift(self):
        # First record matches expected schema; second has a different metric set.
        with open(self.path, "w", encoding="utf8") as f:
            f.write(
                json.dumps(
                    {
                        "subject_name": "subj_a",
                        "groups": {"liver": {"dice": 0.9, "tp": 5.0}},
                    }
                )
                + "\n"
            )
            f.write(
                json.dumps(
                    {
                        "subject_name": "subj_b",
                        "groups": {"liver": {"dice": 0.8, "iou": 0.6}},
                    }
                )
                + "\n"
            )
        backend = JSONLBackend(self.path)
        with self.assertRaisesRegex(
            ValueError, "schema of existing file does not match"
        ):
            backend.prepare_for_append(["liver"], ["dice", "tp"])


class Test_AUTC_Backend_Roundtrip(unittest.TestCase):
    """AUTC writes a different key namespace (``autc_<metric>`` plus per-
    threshold ``t<threshold>_<metric>`` keys), so the round-trip path is
    not covered by the non-AUTC tests above."""

    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"
        self.paths = [
            _TMP_DIR.joinpath("unittest_autc_backend.jsonl"),
            _TMP_DIR.joinpath("unittest_autc_backend.tsv"),
        ]
        for p in self.paths:
            if p.exists():
                os.remove(p)

    def tearDown(self) -> None:
        for p in self.paths:
            if p.exists():
                os.remove(p)

    def _evaluator(self) -> Panoptica_Evaluator:
        return Panoptica_Evaluator(
            expected_input=InputType.SEMANTIC,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=NaiveThresholdMatching(),
        )

    def _run_for_path(self, path: Path) -> None:
        evaluator = self._evaluator()
        aggregator = Panoptica_Aggregator(
            evaluator,
            output_file=path,
            is_autc=True,
            threshold_step_size=0.5,
        )
        a, b = _two_instance_arrays()
        aggregator.evaluate(b, a, "subj_a")

        # Re-open with continue_file=True and append a second subject; the
        # schema validation must accept the AUTC key namespace.
        aggregator2 = Panoptica_Aggregator(
            self._evaluator(),
            output_file=path,
            is_autc=True,
            threshold_step_size=0.5,
            continue_file=True,
        )
        aggregator2.evaluate(b, a, "subj_b")

        stat = Panoptica_Statistic.from_file(path, verbose=False)
        self.assertIn("subj_a", stat.subjectnames)
        self.assertIn("subj_b", stat.subjectnames)
        self.assertIn("autc_pq", stat.metricnames)
        self.assertIn("t0.5_pq", stat.metricnames)
        self.assertIn("t1_pq", stat.metricnames)

    def test_autc_roundtrip_jsonl(self):
        self._run_for_path(self.paths[0])

    def test_autc_roundtrip_tsv(self):
        self._run_for_path(self.paths[1])


class Test_Log_Times_Roundtrip(unittest.TestCase):
    """``log_times=True`` injects ``computation_time`` into the schema.
    Reopening the same file with ``continue_file=True`` and ``log_times=True``
    again must not trip the schema-validation path on either backend."""

    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"
        self.paths = [
            _TMP_DIR.joinpath("unittest_logtimes.jsonl"),
            _TMP_DIR.joinpath("unittest_logtimes.tsv"),
        ]
        for p in self.paths:
            if p.exists():
                os.remove(p)

    def tearDown(self) -> None:
        for p in self.paths:
            if p.exists():
                os.remove(p)

    def _run_for_path(self, path: Path) -> None:
        evaluator = _make_simple_evaluator()
        aggregator = Panoptica_Aggregator(evaluator, output_file=path, log_times=True)
        a, b = _two_instance_arrays()
        aggregator.evaluate(b, a, "subj_a")

        # Reopen with continue_file=True and log_times=True — must accept the
        # existing file (computation_time is part of the schema for both
        # sessions) and append a second subject.
        aggregator2 = Panoptica_Aggregator(
            _make_simple_evaluator(),
            output_file=path,
            log_times=True,
            continue_file=True,
        )
        aggregator2.evaluate(b, a, "subj_b")

        stat = Panoptica_Statistic.from_file(path, verbose=False)
        self.assertIn("subj_a", stat.subjectnames)
        self.assertIn("subj_b", stat.subjectnames)
        self.assertIn("computation_time", stat.metricnames)

    def test_log_times_roundtrip_jsonl(self):
        self._run_for_path(self.paths[0])

    def test_log_times_roundtrip_tsv(self):
        self._run_for_path(self.paths[1])


class Test_JSONL_Write_Full_Instance_Ordering(unittest.TestCase):
    """`write_full` must place instances by their parsed ``inst_idx``, not
    by their encounter order in ``subj_names`` — so a stat built with a
    non-monotonic subject ordering still rewrites with instances in their
    declared index order."""

    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"
        self.path = _TMP_DIR.joinpath("unittest_jsonl_order.jsonl")
        if self.path.exists():
            os.remove(self.path)

    def tearDown(self) -> None:
        if self.path.exists():
            os.remove(self.path)

    def test_write_full_preserves_instance_order_by_inst_idx(self):
        # Master "subj_a" followed by instance rows _inst_2, _inst_0, _inst_1
        # (i.e. non-monotonic encounter order). The metric value encodes the
        # inst_idx so we can assert the written order without parsing names.
        subj_names = [
            "subj_a",
            "subj_a-liver_inst_2",
            "subj_a-liver_inst_0",
            "subj_a-liver_inst_1",
        ]
        value_dict = {
            "liver": {
                "dice": [0.9, 0.2, 0.0, 0.1],
                "tp": [3.0, None, None, None],
            }
        }
        backend = JSONLBackend(self.path)
        backend.write_full(subj_names, value_dict, ["liver"], ["dice", "tp"])

        with open(self.path, "r", encoding="utf8") as f:
            record = json.loads(f.readline())
        inst_list = record["groups"]["liver"]["reference_instances"]
        # Ordered by inst_idx 0, 1, 2 — not by encounter order 2, 0, 1.
        self.assertEqual([r["dice"] for r in inst_list], [0.0, 0.1, 0.2])
