import numpy as np
from panoptica.panoptica_statistics import Panoptica_Statistic
from panoptica.panoptica_evaluator import Panoptica_Evaluator
from panoptica.panoptica_result import PanopticaResult
from pathlib import Path
from multiprocessing import Lock, set_start_method
import csv
import os
import atexit

set_start_method("fork")
filelock = Lock()
inevalfilelock = Lock()


#
class Panoptica_Aggregator:
    # internal_list_lock = Lock()
    #
    """Aggregator that calls evaluations and saves the resulting metrics per sample. Can be used to create statistics, ..."""

    def __init__(
        self,
        panoptica_evaluator: Panoptica_Evaluator,
        output_file: Path | str,
        continue_file: bool = True,
    ):
        """
        Args:
            panoptica_evaluator (Panoptica_Evaluator): The Panoptica_Evaluator used for the pipeline.
            output_file (Path | None, optional): If given, will stream the sample results into this file. If the file is existent, will append results if not already there. Defaults to None.
        """
        self.__panoptica_evaluator = panoptica_evaluator
        self.__class_group_names = panoptica_evaluator.segmentation_class_groups_names
        self.__output_file = None
        self.__output_buffer_file = None
        self.__evaluation_metrics = panoptica_evaluator.resulting_metric_keys

        if isinstance(output_file, str):
            output_file = Path(output_file)
        # uses tsv
        assert (
            output_file.parent.exists()
        ), f"Directory {str(output_file.parent)} does not exist"

        out_file_path = str(output_file)
        if not out_file_path.endswith(".tsv"):
            out_file_path += ".tsv"

        out_buffer_file: Path = Path(out_file_path).parent.joinpath(
            "panoptica_aggregator_tmp.tsv"
        )
        self.__output_buffer_file = out_buffer_file

        Path(out_file_path).parent.mkdir(parents=True, exist_ok=True)
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
            # TODO should also hash panoptica_evaluator just to make sure! and then save into header of file
            assert header_hash == hash(
                "+".join(header_list)
            ), "Hash of header not the same! You are using a different setup!"

        if out_buffer_file.exists():
            os.remove(out_buffer_file)
        open(out_buffer_file, "a").close()

        if continue_file:
            with inevalfilelock:
                with filelock:
                    id_list = _load_first_column_entries(self.__output_file)
                    _write_content(self.__output_buffer_file, [[s] for s in id_list])

        atexit.register(self.__exist_handler)

    def __exist_handler(self):
        os.remove(self.__output_buffer_file)

    def make_statistic(self) -> Panoptica_Statistic:
        with filelock:
            obj = Panoptica_Statistic.from_file(self.__output_file)
        return obj

    def evaluate(
        self,
        prediction_arr: np.ndarray,
        reference_arr: np.ndarray,
        subject_name: str,
    ):
        """Evaluates one case

        Args:
            prediction_arr (np.ndarray): Prediction array
            reference_arr (np.ndarray): reference array
            subject_name (str | None, optional): Unique name of the sample. If none, will give it a name based on count. Defaults to None.
            skip_already_existent (bool): If true, will skip subjects which were already evaluated instead of crashing. Defaults to False.
            verbose (bool | None, optional): Verbose. Defaults to None.
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
        res = self.__panoptica_evaluator.evaluate(
            prediction_arr,
            reference_arr,
            result_all=True,
            verbose=False,
            log_times=False,
        )

        # Add to file
        self._save_one_subject(subject_name, res)

    def _save_one_subject(self, subject_name, result_grouped):
        with filelock:
            #
            content = [subject_name]
            for groupname in self.__class_group_names:
                result: PanopticaResult = result_grouped[groupname][0]
                result_dict = result.to_dict()
                del result

                for e in self.__evaluation_metrics:
                    mvalue = result_dict[e] if e in result_dict else ""
                    content.append(mvalue)
            _write_content(self.__output_file, [content])
            print(f"Saved entry {subject_name} into {str(self.__output_file)}")


def _read_first_row(file: str):
    # NOT THREAD SAFE BY ITSELF!
    with open(str(file), "r", encoding="utf8", newline="") as tsvfile:
        rd = csv.reader(tsvfile, delimiter="\t", lineterminator="\n")

        rows = [row for row in rd]
        if len(rows) == 0:
            row = []
        else:
            row = rows[0]

    return row


def _load_first_column_entries(file: str):
    # NOT THREAD SAFE BY ITSELF!
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


def _write_content(file: str, content: list[list[str]]):
    # NOT THREAD SAFE BY ITSELF!
    with open(str(file), "a", encoding="utf8", newline="") as tsvfile:
        writer = csv.writer(tsvfile, delimiter="\t", lineterminator="\n")
        for c in content:
            writer.writerow(c)
