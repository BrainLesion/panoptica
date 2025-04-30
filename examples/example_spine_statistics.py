from auxiliary.nifti.io import read_nifti
from auxiliary.turbopath import turbopath

from panoptica import (
    Panoptica_Evaluator,
    Panoptica_Aggregator,
    InputType,
    NaiveThresholdMatching,
    Metric,
)
from panoptica.utils import SegmentationClassGroups, LabelGroup
from panoptica.panoptica_statistics import make_curve_over_setups
from pathlib import Path
from panoptica.utils import NonDaemonicPool
from joblib import delayed, Parallel
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from multiprocessing import set_start_method
import os

# set_start_method("fork")


def main(parallel_opt: str = "future"):  # none, pool, joblib, future
    file_dir = Path(__file__).parent.joinpath("spine_example.tsv")
    try:
        os.remove(str(file_dir))
    except:
        pass

    directory = turbopath(__file__).parent

    reference_mask = read_nifti(directory + "/spine_seg/matched_instance/ref.nii.gz")
    prediction_mask = read_nifti(directory + "/spine_seg/matched_instance/pred.nii.gz")

    evaluator = Panoptica_Aggregator(
        Panoptica_Evaluator.load_from_config("panoptica_evaluator_unmatched_instance"),
        Path(__file__).parent.joinpath("spine_example.tsv"),
        log_times=True,
    )

    evaluator.panoptica_evaluator.set_log_group_times(True)
    #
    parallel_opt = parallel_opt.lower()

    if parallel_opt == "pool":
        args = [
            (prediction_mask, reference_mask, "sample1"),
            (prediction_mask, reference_mask, "sample2"),
            (prediction_mask, reference_mask, "sample3"),
            (prediction_mask, reference_mask, "sample4"),
        ]
        with NonDaemonicPool() as pool:
            pool.starmap(evaluator.evaluate, args)
    elif parallel_opt == "none":
        for i in range(4):
            results = evaluator.evaluate(prediction_mask, reference_mask, f"sample{i}")
    elif parallel_opt == "joblib":
        Parallel(n_jobs=5, backend="threading")(
            delayed(evaluator.evaluate)(prediction_mask, reference_mask, f"sample{i}") for i in range(8)
        )
    elif parallel_opt == "future":
        with ProcessPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(evaluator.evaluate, prediction_mask, reference_mask, f"sample{i}") for i in range(8)}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Panoptica Evaluation"):
                result = future.result()
                if result is not None:
                    print("Done")

    panoptic_statistic = evaluator.make_statistic()
    panoptic_statistic.print_summary()

    fig = panoptic_statistic.get_summary_figure("sq_dsc", horizontal=True)
    out_figure = str(Path(__file__).parent.joinpath("example_sq_dsc_figure.png"))
    # fig.write_image(out_figure)

    fig2 = make_curve_over_setups(
        {
            "t0.5": panoptic_statistic,
            "bad": panoptic_statistic,
            "good classifier": panoptic_statistic,
            2.0: panoptic_statistic,
        },
        groups=None,
        metric="pq",
    )

    out_figure = str(Path(__file__).parent.joinpath("example_multiple_statistics.png"))
    # fig2.savefig(out_figure)


if __name__ == "__main__":
    main()
