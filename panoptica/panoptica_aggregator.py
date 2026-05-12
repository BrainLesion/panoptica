from panoptica.utils import format_instance_subject_name, validate_subject_name
import numpy as np
from panoptica.panoptica_statistics import Panoptica_Statistic
from panoptica.panoptica_evaluator import Panoptica_Evaluator
from panoptica.panoptica_result import PanopticaAUTCResult, PanopticaResult
from pathlib import Path
from multiprocessing import Lock, set_start_method
import csv
import os
import atexit
from tempfile import NamedTemporaryFile
import warnings
from typing import Optional

from panoptica.utils.file_type import FileType, derive_file_type

# Set start method based on the operating system
try:
    if os.name == "posix":
        set_start_method("fork")
    elif os.name == "nt":
        set_start_method("spawn")
        warnings.warn(
            "The multiprocessing start method has been set to 'spawn' since 'fork' is not available on Windows. This can lead to thread unsafety in the current development state."
        )
except RuntimeError:
    # Start method can only be set once per process, so ignore if already set
    pass

filelock = Lock()
inevalfilelock = Lock()

COMPUTATION_TIME_KEY = "computation_time"

class Panoptica_Aggregator:
    """Aggregator that manages evaluations and saves resulting metrics per sample.

    This class interfaces with the `Panoptica_Evaluator` to perform evaluations,
    store results, and manage file outputs for statistical analysis.
    """

    def __init__(
        self,
        panoptica_evaluator: Panoptica_Evaluator,
        output_file: Path | str,
        log_times: bool = False,
        continue_file: bool = True,
        file_type: FileType = "jsonl",
        output_individual_instance_metrics: bool = False,
        is_autc: bool = False,
        threshold_step_size: Optional[float] = None,
    ):
        """Initializes the Panoptica_Aggregator.

        Args:
            panoptica_evaluator (Panoptica_Evaluator): The evaluator used for performing evaluations.
            output_file (Path | str): Path to the output file for storing results. If the file exists,
                results will be appended. If it doesn't, a new file will be created.
            log_times (bool, optional): If True, computation times will be logged. Defaults to False.
            continue_file (bool, optional): If True, results will continue from existing entries in the file.
                Defaults to True.
            file_type (FileType, optional): Format used when `output_file` has no extension. Ignored if `output_file` already has a supported suffix. Defaults to "jsonl".
            output_individual_instance_metrics (bool, optional): If True, individual instance metrics will be output. Defaults to False.
            is_autc (bool, optional): If True, the aggregator will compute AUTC metrics. Defaults to False.
            threshold_step_size (Optional[float], optional): The step size for thresholding. Defaults to None.

        Raises:
            FileNotFoundError: If the output directory does not exist.
            ValueError: If the file extension is not supported, or if the header of the existing file does not match the expected header based on the evaluator's configuration.
        """
        self.__panoptica_evaluator = panoptica_evaluator
        self.__class_group_names = panoptica_evaluator.segmentation_class_groups_names
        self.__file_type = file_type
        self.__autc = is_autc
        self.__log_times = log_times
        self.__continue_file = continue_file
        self.__output_individual_instance_metrics = output_individual_instance_metrics
        self.__threshold_step_size = threshold_step_size
        self.__output_buffer_file = NamedTemporaryFile(delete=False).name

        if is_autc:
            if self.__threshold_step_size is None:
                raise ValueError(
                    "threshold_step_size must be provided to build AUTC headers"
                )
            self.__evaluation_metrics = panoptica_evaluator.get_autc_metric_keys(
                self.__threshold_step_size
            )
        else:
            self.__evaluation_metrics = panoptica_evaluator.resulting_metric_keys

        if log_times and COMPUTATION_TIME_KEY not in self.__evaluation_metrics:
            self.__evaluation_metrics.append(COMPUTATION_TIME_KEY)

        if isinstance(output_file, str):
            output_file = Path(output_file)

        if not output_file.parent.exists():
            raise FileNotFoundError(
                f"Directory {str(output_file.parent)} does not exist"
            )

        if output_file.suffix:
            # Override preset file type if output file contains suffix
            self.__file_type = derive_file_type(output_file)
        else:
            output_file = output_file.with_suffix(f".{self.__file_type}")

        self.__output_file = output_file

        match self.__file_type:
            case 'tsv':
                self.__write_tsv()
            case 'jsonl':
                self.__write_jsonl()

        atexit.register(self.__exist_handler)

    @property
    def panoptica_evaluator(self):
        return self.__panoptica_evaluator

    @property
    def evaluation_metrics(self):
        return self.__evaluation_metrics

    def __write_tsv(self):
        """Initializes the TSV output file.

        Writes the header if the file is new or empty, validates header
        consistency on continuation, and seeds the buffer file with the
        subject names already present in the file when ``continue_file``
        is True.

        Raises:
            ValueError: If the existing file's header hash differs from
                the header derived from the current evaluator configuration.
        """
        header = ["subject_name"] + [
            f"{g}-{m}"
            for g in self.__class_group_names
            for m in self.__evaluation_metrics
        ]
        header_hash = hash("+".join(header))

        if not self.__output_file.exists():
            _write_content(self.__output_file, [header])
        else:
            header_list = _read_first_row(self.__output_file)
            if len(header_list) == 0:
                print(
                    f"{self.__output_file}: Output file given is empty, will start with header"
                )
                _write_content(self.__output_file, [header])
                self.__continue_file = True
            else:
                # TODO should also hash panoptica_evaluator just to make sure! and then save into header of file
                if header_hash != hash("+".join(header_list)):
                    raise ValueError(
                        f"{self.__output_file}: Hash of header not the same! You are using a different setup!"
                    )

        if self.__continue_file:
            with inevalfilelock:
                with filelock:
                    id_list = _load_first_column_entries(self.__output_file)
                    _write_content(self.__output_buffer_file, [[s] for s in id_list])

    def __write_jsonl(self):
        """Initializes the JSONL output file.

        JSONL counterpart to ``__write_tsv``: validates schema/config
        compatibility with any existing file and seeds the buffer file
        with previously written subject names when ``continue_file`` is
        True.

        Raises:
            NotImplementedError: JSONL writing is not yet implemented.
        """
        raise NotImplementedError()

    def __exist_handler(self):
        """Handles cleanup upon program exit by removing the temporary output buffer file."""
        if Path(self.__output_buffer_file).exists():
            os.remove(str(self.__output_buffer_file))

    def make_statistic(self) -> Panoptica_Statistic:
        """Generates statistics from the aggregated evaluation results.

        Returns:
            Panoptica_Statistic: The statistics object containing the results.
        """
        with filelock:
            obj = Panoptica_Statistic.from_file(self.__output_file)
        return obj

    def evaluate(
        self,
        prediction_arr: np.ndarray,
        reference_arr: np.ndarray,
        subject_name: str,
        voxelspacing: tuple[float, ...] | None = None,
        **kwargs,
    ):
        """Evaluates a single case using the provided prediction and reference arrays.

        Args:
            prediction_arr (np.ndarray): The array containing the predicted segmentation.
            reference_arr (np.ndarray): The array containing the ground truth segmentation.
            subject_name (str): A unique name for the sample being evaluated. If none is provided,
                a name will be generated based on the count.

        Raises:
            ValueError: If the subject name has already been evaluated or is in process.
        """
        validate_subject_name(subject_name)
        # Read tmp file to see which sample names are blocked
        with inevalfilelock:
            id_list = _load_first_column_entries(self.__output_buffer_file)

            if subject_name in id_list:
                print(
                    f"Subject '{subject_name}' evaluated or in process {self.__output_file}, do not add duplicates to your evaluation!",
                    flush=True,
                )
                return
            _write_content(self.__output_buffer_file, [[subject_name]])

        # Run Evaluation (allowed in parallel)
        print(f"Call evaluate on {subject_name}")
        if self.__autc:
            if self.__threshold_step_size is None:
                raise ValueError(
                    "threshold_step_size must be provided to build AUTC headers"
                )
            res = self.__panoptica_evaluator.evaluate_autc(
                prediction_arr,
                reference_arr,
                threshold_step_size=self.__threshold_step_size,
                result_all=True,
                verbose=False,
                log_times=False,
                save_group_times=self.__log_times,
                voxelspacing=voxelspacing,
                **kwargs,
            )
        else:
            res = self.__panoptica_evaluator.evaluate(
                prediction_arr,
                reference_arr,
                result_all=True,
                verbose=False,
                log_times=False,
                save_group_times=self.__log_times,
                voxelspacing=voxelspacing,
                **kwargs,
            )

        # Add to file
        match self.__file_type:
            case 'tsv':
                self._save_one_subject_tsv(subject_name, res)
            case 'jsonl':
                self._save_one_subject_jsonl(subject_name, res)

    def _save_one_subject_tsv(self, subject_name, result_grouped):
        """Saves the evaluation results for a single subject."""
        with filelock:
            if self.__output_individual_instance_metrics:
                all_rows = []
                summary_row = [subject_name]
                group_rows_as_dicts = {}
                for groupname in self.__class_group_names:
                    result: PanopticaResult = result_grouped[groupname]
                    rows_as_dicts = result.to_dict(True)
                    group_rows_as_dicts[groupname] = rows_as_dicts
                    summary_dict = rows_as_dicts[0] if len(rows_as_dicts) > 0 else {}
                    if result.computation_time is not None:
                        summary_dict = dict(summary_dict)
                        summary_dict[COMPUTATION_TIME_KEY] = result.computation_time
                    for e in self.__evaluation_metrics:
                        summary_row.append(summary_dict.get(e, ""))
                all_rows.append(summary_row)
                for groupname in self.__class_group_names:
                    rows_as_dicts = group_rows_as_dicts[groupname]
                    for inst_idx, r_dict in enumerate(rows_as_dicts[1:]):
                        row = [
                            format_instance_subject_name(
                                subject_name, groupname, inst_idx
                            )
                        ]
                        for current_groupname in self.__class_group_names:
                            if current_groupname == groupname:
                                for e in self.__evaluation_metrics:
                                    row.append(r_dict.get(e, ""))
                            else:
                                for _ in self.__evaluation_metrics:
                                    row.append("")
                        all_rows.append(row)

                _write_content(self.__output_file, all_rows)
            else:
                content = [subject_name]
                for groupname in self.__class_group_names:
                    result: PanopticaResult | PanopticaAUTCResult = result_grouped[
                        groupname
                    ]
                    result_dict = result.to_dict(False)

                    if result.computation_time is not None:
                        result_dict[COMPUTATION_TIME_KEY] = result.computation_time

                    for e in self.__evaluation_metrics:
                        content.append(result_dict.get(e, ""))
                _write_content(self.__output_file, [content])

            print(f"Saved entry {subject_name} into {str(self.__output_file)}")

    def _save_one_subject_jsonl(self, subject_name, result_grouped):
        """Appends one nested JSON record for a single subject to the JSONL file.

        Each record carries the per-group summary metrics and, for
        ``PanopticaResult`` (non-AUTC) outputs, a list of matched-instance
        metrics — mirroring the schema described in the JSONL design.

        Args:
            subject_name: Identifier for the evaluated subject.
            result_grouped: Mapping from group name to ``PanopticaResult``
                or ``PanopticaAUTCResult`` produced by the evaluator.

        Raises:
            NotImplementedError: JSONL saving is not yet implemented.
        """
        raise NotImplementedError()


def _read_first_row(file: str | Path):
    """Reads the first row of a TSV file.

    NOT THREAD SAFE BY ITSELF!

    Args:
        file (str | Path): The path to the file from which to read the first row.

    Returns:
        list: The first row of the file as a list of strings.
    """
    if isinstance(file, Path):
        file = str(file)
    #
    with open(str(file), "r", encoding="utf8", newline="") as tsvfile:
        rd = csv.reader(tsvfile, delimiter="\t", lineterminator="\n")

        rows = [row for row in rd]
        if len(rows) == 0:
            row = []
        else:
            row = rows[0]

    return row


def _load_first_column_entries(file: str | Path):
    """Loads the entries from the first column of a TSV file.

    NOT THREAD SAFE BY ITSELF!

    Args:
        file (str | Path): The path to the file from which to load entries.

    Returns:
        list: A list of entries from the first column of the file.

    Raises:
        ValueError: If the file contains duplicate entries.
    """
    if isinstance(file, Path):
        file = str(file)
    with open(str(file), "r", encoding="utf8", newline="") as tsvfile:
        rd = csv.reader(tsvfile, delimiter="\t", lineterminator="\n")

        rows = [row for row in rd]
        if len(rows) == 0:
            id_list = []
        else:
            id_list = list([row[0] for row in rows])

    n_id = len(id_list)
    if n_id != len(list(set(id_list))):
        raise ValueError(f"{file}: file has duplicate entries!")

    return id_list


def _write_content(file: str | Path, content: list[list[str]]):
    """Writes content to a TSV file.

    Args:
        file (str | Path): The path to the file where content will be written.
        content (list[list[str]]): A list of lists containing the rows of data to write.
    """
    if isinstance(file, Path):
        file = str(file)
    # NOT THREAD SAFE BY ITSELF!
    with open(str(file), "a", encoding="utf8", newline="") as tsvfile:
        writer = csv.writer(tsvfile, delimiter="\t", lineterminator="\n")
        for c in content:
            writer.writerow(["" if v is None else v for v in c])
