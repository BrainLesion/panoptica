import numpy as np
from panoptica.panoptica_evaluator import Panoptica_Evaluator
from panoptica.panoptica_result import PanopticaResult
from dataclasses import dataclass


@dataclass
class NamedPanopticaResultGroup:
    name: str
    group2result: dict[str, PanopticaResult]


# Mean over instances
# mean over subjects
# give below/above percentile of metric (the names)
# make plot with metric dots
# make auc curve as plot
class Panoptica_Aggregator:
    """Aggregator that calls evaluations and saves the resulting metrics per sample. Can be used to create statistics, ..."""

    def __init__(self, panoptica_evaluator: Panoptica_Evaluator):
        self._panoptica_evaluator = panoptica_evaluator
        self._group2named_results: dict[str, list[NamedPanopticaResultGroup]] = {}
        self._n_samples = 0

    def evaluate(
        self,
        prediction_arr: np.ndarray,
        reference_arr: np.ndarray,
        subject_name: str | None = None,
        verbose: bool | None = None,
    ):
        """Evaluates one case

        Args:
            prediction_arr (np.ndarray): Prediction array
            reference_arr (np.ndarray): reference array
            subject_name (str | None, optional): Unique name of the sample. If none, will give it a name based on count. Defaults to None.
            verbose (bool | None, optional): Verbose. Defaults to None.
        """
        if subject_name is None:
            subject_name = f"Sample_{self._n_samples}"

        res = self._panoptica_evaluator.evaluate(
            prediction_arr,
            reference_arr,
            result_all=True,
            verbose=verbose,
        )
        for k, v in res.items():
            if k not in self._group2named_results:
                self._group2named_results[k] = []
            result_obj, _ = v
            self._group2named_results[k].append(NamedPanopticaResultGroup(subject_name, result_obj))

        self._n_samples += 1

    def save_results():
        # save to excel
        pass

    def load_results():
        pass
