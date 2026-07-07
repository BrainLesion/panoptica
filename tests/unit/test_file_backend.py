"""Tests for panoptica.io.file_backend backends (TSV and JSONL)."""

from __future__ import annotations

from pathlib import Path

import pytest

from panoptica.core.errors import InputValidationError
from panoptica.io.file_backend import derive_file_type, get_backend
from panoptica.io.file_backend.jsonl import JSONLBackend
from panoptica.io.file_backend.tsv import TSVBackend


class MockSerializable:
    """Mock Serializable object for testing."""

    def __init__(self, data: dict[str, float | None]) -> None:
        self.data = data

    def to_dict(self, include_instances: bool = False) -> dict[str, float | None]:
        """Return the data dict (mock Serializable protocol)."""
        result = dict(self.data)
        if include_instances:
            result["reference_instances"] = [
                {"dice": 0.9, "iou": 0.8},
                {"dice": 0.85, "iou": 0.75},
            ]
        return result


class TestFileTypeDetection:
    """Tests for file type detection."""

    def test_tsv_extension_detected(self) -> None:
        """TSV extension is detected."""
        file_type = derive_file_type(Path("results.tsv"))
        assert file_type == "tsv"

    def test_jsonl_extension_detected(self) -> None:
        """JSONL extension is detected."""
        file_type = derive_file_type(Path("results.jsonl"))
        assert file_type == "jsonl"

    def test_case_insensitive_extension(self) -> None:
        """Extensions are case-insensitive."""
        file_type = derive_file_type(Path("results.TSV"))
        assert file_type == "tsv"

    def test_missing_extension_rejected(self) -> None:
        """Missing extension raises error."""
        with pytest.raises(InputValidationError, match="No file extension"):
            derive_file_type(Path("results"))

    def test_unsupported_extension_rejected(self) -> None:
        """Unsupported extension raises error."""
        with pytest.raises(InputValidationError, match="Unsupported"):
            derive_file_type(Path("results.csv"))


class TestGetBackend:
    """Tests for backend instantiation."""

    def test_get_tsv_backend(self, tmp_path: Path) -> None:
        """get_backend() returns TSVBackend for .tsv."""
        path = tmp_path / "results.tsv"
        backend = get_backend(path)
        assert isinstance(backend, TSVBackend)
        assert backend.path == path

    def test_get_jsonl_backend(self, tmp_path: Path) -> None:
        """get_backend() returns JSONLBackend for .jsonl."""
        path = tmp_path / "results.jsonl"
        backend = get_backend(path)
        assert isinstance(backend, JSONLBackend)
        assert backend.path == path


class TestTSVBackend:
    """Tests for TSV backend."""

    def test_tsv_prepare_creates_header(self, tmp_path: Path) -> None:
        """prepare_for_append() creates header in new file."""
        path = tmp_path / "results.tsv"
        backend = TSVBackend(path)

        existing = backend.prepare_for_append(
            group_names=["liver", "spleen"],
            metric_names=["dice", "iou"],
            collect_existing=False,
        )

        assert not existing
        assert path.exists()

        # Check header
        with open(path) as f:
            header = f.readline().strip().split("\t")
        assert header == [
            "subject_name",
            "liver-dice",
            "liver-iou",
            "spleen-dice",
            "spleen-iou",
        ]

    def test_tsv_append_subject(self, tmp_path: Path) -> None:
        """append_subject() appends a row with values."""
        path = tmp_path / "results.tsv"
        backend = TSVBackend(path)
        backend.prepare_for_append(["liver"], ["dice", "iou"])

        result_grouped = {
            "liver": MockSerializable({"dice": 0.9, "iou": 0.8}),
        }
        backend.append_subject(
            subject_name="subj_a",
            result_grouped=result_grouped,
            group_names=["liver"],
            metric_names=["dice", "iou"],
            output_individual_instance_metrics=False,
        )

        # Check file content
        lines = path.read_text().strip().split("\n")
        assert lines[0] == "subject_name\tliver-dice\tliver-iou"
        assert lines[1] == "subj_a\t0.9\t0.8"

    def test_tsv_append_multiple_subjects(self, tmp_path: Path) -> None:
        """Multiple subjects are appended to TSV."""
        path = tmp_path / "results.tsv"
        backend = TSVBackend(path)
        backend.prepare_for_append(["liver"], ["dice"])

        backend.append_subject(
            "subj_a",
            {"liver": MockSerializable({"dice": 0.9})},
            ["liver"],
            ["dice"],
            False,
        )
        backend.append_subject(
            "subj_b",
            {"liver": MockSerializable({"dice": 0.85})},
            ["liver"],
            ["dice"],
            False,
        )

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 3
        assert lines[1] == "subj_a\t0.9"
        assert lines[2] == "subj_b\t0.85"

    def test_tsv_load_raw(self, tmp_path: Path) -> None:
        """load_raw() reads TSV and returns structured data."""
        path = tmp_path / "results.tsv"
        backend = TSVBackend(path)
        backend.prepare_for_append(["liver", "spleen"], ["dice", "iou"])

        backend.append_subject(
            "subj_a",
            {
                "liver": MockSerializable({"dice": 0.9, "iou": 0.8}),
                "spleen": MockSerializable({"dice": 0.7, "iou": 0.6}),
            },
            ["liver", "spleen"],
            ["dice", "iou"],
            False,
        )

        subj_names, value_dict = backend.load_raw(verbose=False)

        assert subj_names == ["subj_a"]
        assert value_dict["liver"]["dice"] == [0.9]
        assert value_dict["liver"]["iou"] == [0.8]
        assert value_dict["spleen"]["dice"] == [0.7]
        assert value_dict["spleen"]["iou"] == [0.6]

    def test_tsv_schema_mismatch_rejected(self, tmp_path: Path) -> None:
        """Reopening with different schema raises error."""
        path = tmp_path / "results.tsv"

        # Create initial file
        backend1 = TSVBackend(path)
        backend1.prepare_for_append(["liver"], ["dice", "iou"])

        # Try to reopen with different schema
        backend2 = TSVBackend(path)
        with pytest.raises(InputValidationError, match="Header mismatch"):
            backend2.prepare_for_append(["liver"], ["dice"])  # Missing iou


class TestJSONLBackend:
    """Tests for JSONL backend."""

    def test_jsonl_prepare_creates_file(self, tmp_path: Path) -> None:
        """prepare_for_append() creates empty file."""
        path = tmp_path / "results.jsonl"
        backend = JSONLBackend(path)

        existing = backend.prepare_for_append(
            group_names=["liver"],
            metric_names=["dice"],
            collect_existing=False,
        )

        assert not existing
        assert path.exists()
        assert path.read_text() == ""

    def test_jsonl_append_subject(self, tmp_path: Path) -> None:
        """append_subject() appends a JSON record."""
        path = tmp_path / "results.jsonl"
        backend = JSONLBackend(path)
        backend.prepare_for_append(["liver"], ["dice", "iou"])

        result_grouped = {
            "liver": MockSerializable({"dice": 0.9, "iou": 0.8}),
        }
        backend.append_subject(
            subject_name="subj_a",
            result_grouped=result_grouped,
            group_names=["liver"],
            metric_names=["dice", "iou"],
            output_individual_instance_metrics=False,
        )

        # Check file content
        import json

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["subject_name"] == "subj_a"
        assert record["groups"]["liver"]["dice"] == 0.9
        assert record["groups"]["liver"]["iou"] == 0.8

    def test_jsonl_append_with_instances(self, tmp_path: Path) -> None:
        """append_subject() includes reference_instances when requested."""
        path = tmp_path / "results.jsonl"
        backend = JSONLBackend(path)
        backend.prepare_for_append(["liver"], ["dice", "iou"])

        result_grouped = {
            "liver": MockSerializable({"dice": 0.9, "iou": 0.8}),
        }
        backend.append_subject(
            subject_name="subj_a",
            result_grouped=result_grouped,
            group_names=["liver"],
            metric_names=["dice", "iou"],
            output_individual_instance_metrics=True,
        )

        # Check file content
        import json

        lines = path.read_text().strip().split("\n")
        record = json.loads(lines[0])
        assert "reference_instances" in record["groups"]["liver"]
        assert len(record["groups"]["liver"]["reference_instances"]) == 2

    def test_jsonl_load_raw(self, tmp_path: Path) -> None:
        """load_raw() reads JSONL and returns structured data."""
        path = tmp_path / "results.jsonl"
        backend = JSONLBackend(path)
        backend.prepare_for_append(["liver"], ["dice", "iou"])

        backend.append_subject(
            "subj_a",
            {"liver": MockSerializable({"dice": 0.9, "iou": 0.8})},
            ["liver"],
            ["dice", "iou"],
            False,
        )

        subj_names, value_dict = backend.load_raw(verbose=False)

        assert subj_names == ["subj_a"]
        assert value_dict["liver"]["dice"] == [0.9]
        assert value_dict["liver"]["iou"] == [0.8]

    def test_jsonl_multiple_subjects(self, tmp_path: Path) -> None:
        """Multiple subjects can be appended to JSONL."""
        path = tmp_path / "results.jsonl"
        backend = JSONLBackend(path)
        backend.prepare_for_append(["liver"], ["dice"])

        backend.append_subject(
            "subj_a",
            {"liver": MockSerializable({"dice": 0.9})},
            ["liver"],
            ["dice"],
            False,
        )
        backend.append_subject(
            "subj_b",
            {"liver": MockSerializable({"dice": 0.85})},
            ["liver"],
            ["dice"],
            False,
        )

        subj_names, _ = backend.load_raw(verbose=False)
        assert subj_names == ["subj_a", "subj_b"]

    def test_jsonl_schema_mismatch_rejected(self, tmp_path: Path) -> None:
        """Reopening with different schema raises error."""
        path = tmp_path / "results.jsonl"

        # Create initial file
        backend1 = JSONLBackend(path)
        backend1.prepare_for_append(["liver"], ["dice", "iou"])
        backend1.append_subject(
            "subj_a",
            {"liver": MockSerializable({"dice": 0.9, "iou": 0.8})},
            ["liver"],
            ["dice", "iou"],
            False,
        )

        # Try to reopen with different schema
        backend2 = JSONLBackend(path)
        with pytest.raises(InputValidationError, match="schema"):
            backend2.prepare_for_append(["liver"], ["dice"])  # Missing iou


class TestRoundTrip:
    """Round-trip tests (write and read back)."""

    def test_tsv_roundtrip(self, tmp_path: Path) -> None:
        """TSV write and read are consistent."""
        path = tmp_path / "results.tsv"
        backend = TSVBackend(path)

        # Write
        backend.prepare_for_append(["liver", "spleen"], ["dice", "iou"])
        backend.append_subject(
            "subj_a",
            {
                "liver": MockSerializable({"dice": 0.9, "iou": 0.8}),
                "spleen": MockSerializable({"dice": 0.7, "iou": 0.6}),
            },
            ["liver", "spleen"],
            ["dice", "iou"],
            False,
        )
        backend.append_subject(
            "subj_b",
            {
                "liver": MockSerializable({"dice": 0.92, "iou": 0.82}),
                "spleen": MockSerializable({"dice": 0.72, "iou": 0.62}),
            },
            ["liver", "spleen"],
            ["dice", "iou"],
            False,
        )

        # Read
        subj_names, value_dict = backend.load_raw(verbose=False)

        assert subj_names == ["subj_a", "subj_b"]
        assert value_dict["liver"]["dice"] == [0.9, 0.92]
        assert value_dict["liver"]["iou"] == [0.8, 0.82]
        assert value_dict["spleen"]["dice"] == [0.7, 0.72]
        assert value_dict["spleen"]["iou"] == [0.6, 0.62]

    def test_jsonl_roundtrip(self, tmp_path: Path) -> None:
        """JSONL write and read are consistent."""
        path = tmp_path / "results.jsonl"
        backend = JSONLBackend(path)

        # Write
        backend.prepare_for_append(["liver", "spleen"], ["dice", "iou"])
        backend.append_subject(
            "subj_a",
            {
                "liver": MockSerializable({"dice": 0.9, "iou": 0.8}),
                "spleen": MockSerializable({"dice": 0.7, "iou": 0.6}),
            },
            ["liver", "spleen"],
            ["dice", "iou"],
            False,
        )

        # Read
        subj_names, value_dict = backend.load_raw(verbose=False)

        assert subj_names == ["subj_a"]
        assert value_dict["liver"]["dice"] == [0.9]
        assert value_dict["liver"]["iou"] == [0.8]
        assert value_dict["spleen"]["dice"] == [0.7]
        assert value_dict["spleen"]["iou"] == [0.6]
