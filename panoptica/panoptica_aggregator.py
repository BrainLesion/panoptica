import numpy as np
from panoptica.panoptica_statistics import Panoptica_Statistic
from panoptica.panoptica_evaluator import Panoptica_Evaluator
from panoptica.panoptica_result import PanopticaResult
from pathlib import Path
from multiprocessing import Lock, set_start_method
import csv
import os
import atexit
import warnings
import tempfile

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


#
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
    ):
        """Initializes the Panoptica_Aggregator.

        Args:
            panoptica_evaluator (Panoptica_Evaluator): The evaluator used for performing evaluations.
            output_file (Path | str): Path to the output file for storing results. If the file exists,
                results will be appended. If it doesn't, a new file will be created.
            log_times (bool, optional): If True, computation times will be logged. Defaults to False.
            continue_file (bool, optional): If True, results will continue from existing entries in the file.
                Defaults to True.

        Raises:
            AssertionError: If the output directory does not exist or if the file extension is not `.tsv`.
        """
        self.__panoptica_evaluator = panoptica_evaluator
        self.__class_group_names = panoptica_evaluator.segmentation_class_groups_names
        self.__evaluation_metrics = panoptica_evaluator.resulting_metric_keys
        self.__log_times = log_times

        if log_times and COMPUTATION_TIME_KEY not in self.__evaluation_metrics:
            self.__evaluation_metrics.append(COMPUTATION_TIME_KEY)

        if isinstance(output_file, str):
            output_file = Path(output_file)
        # uses tsv
        assert (
            output_file.parent.exists()
        ), f"Directory {str(output_file.parent)} does not exist"

        out_file_path = str(output_file)

        # extension
        if "." in out_file_path:
            # extension exists
            extension = out_file_path.split(".")[-1]
            assert (
                extension == "tsv"
            ), f"You gave the extension {extension}, but currently only .tsv is supported. Either delete it or give .tsv as extension"
        else:
            out_file_path += ".tsv"  # add extension

        # buffer_file = tempfile.NamedTemporaryFile()
        # out_buffer_file: Path = Path(out_file_path).parent.joinpath("panoptica_aggregator_tmp.tsv")
        # self.tmpfile =
        self.__output_buffer_file = tempfile.NamedTemporaryFile(
            delete=False
        ).name  # out_buffer_file
        # print(self.__output_buffer_file)

        Path(out_file_path).parent.mkdir(parents=False, exist_ok=True)
        self.__output_file = out_file_path

        header = ["subject_name"] + [
            f"{g}-{m}"
            for g in self.__class_group_names
            for m in self.__evaluation_metrics
        ]
        header_hash = hash("+".join(header))

        if not output_file.exists():
            # write header
            _write_content(output_file, [header])
        else:
            header_list = _read_first_row(output_file)
            if len(header_list) == 0:
                # empty file
                print("Output file given is empty, will start with header")
                continue_file = True
            else:
                # TODO should also hash panoptica_evaluator just to make sure! and then save into header of file
                assert header_hash == hash(
                    "+".join(header_list)
                ), "Hash of header not the same! You are using a different setup!"

        if continue_file:
            with inevalfilelock:
                with filelock:
                    id_list = _load_first_column_entries(self.__output_file)
                    _write_content(self.__output_buffer_file, [[s] for s in id_list])

        atexit.register(self.__exist_handler)

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
        res = self.__panoptica_evaluator.evaluate(
            prediction_arr,
            reference_arr,
            result_all=True,
            verbose=False,
            log_times=False,
            save_group_times=self.__log_times,
        )

        # Add to file
        self._save_one_subject(subject_name, res)

    def _save_one_subject(self, subject_name, result_grouped):
        """Saves the evaluation results for a single subject.

        Args:
            subject_name (str): The name of the subject whose results are being saved.
            result_grouped (dict): A dictionary of grouped results from the evaluation.
        """
        with filelock:
            #
            content = [subject_name]
            for groupname in self.__class_group_names:
                result: PanopticaResult = result_grouped[groupname]
                result_dict = result.to_dict()
                if result.computation_time is not None:
                    result_dict[COMPUTATION_TIME_KEY] = result.computation_time
                del result

                for e in self.__evaluation_metrics:
                    mvalue = result_dict[e] if e in result_dict else ""
                    content.append(mvalue)
            _write_content(self.__output_file, [content])
            print(f"Saved entry {subject_name} into {str(self.__output_file)}")

    @property
    def panoptica_evaluator(self):
        return self.__panoptica_evaluator

    @property
    def evaluation_metrics(self):
        return self.__evaluation_metrics


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
        AssertionError: If the file contains duplicate entries.
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
    assert n_id == len(list(set(id_list))), "file has duplicate entries!"

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
            writer.writerow(c)
