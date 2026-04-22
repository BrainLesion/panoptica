import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from panoptica import (
    Panoptica_Aggregator,
    Panoptica_Evaluator,
)
from panoptica.panoptica_statistics import make_autc_plots


def main():
    directory = Path(__file__).parent
    file_dir = directory / "spine_autc_example.tsv"

    try:
        os.remove(str(file_dir))
    except FileNotFoundError:
        pass

    # Setup paths
    reference_mask = directory / "spine_seg/matched_instance/ref.nii.gz"
    prediction_mask = directory / "spine_seg/matched_instance/pred.nii.gz"

    base_evaluator = Panoptica_Evaluator.load_from_config(
        directory / "panoptica_evaluator_unmatched_instance"
    )

    aggregator = Panoptica_Aggregator(
        base_evaluator,
        file_dir,
        is_autc=True,
        threshold_step_size=0.1,
        log_times=True,
    )

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(
                aggregator.evaluate, prediction_mask, reference_mask, f"sample{i}"
            )
            for i in range(5)
        ]
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="AUTC Sweep"
        ):
            try:
                future.result()
            except Exception as e:
                print(f"Worker failed with: {e}")

    stats = aggregator.make_statistic()
    stats.print_summary()

    fig = make_autc_plots(
        statistics_dict={"Spine Model": stats},
        metric="pq",
        figure_title="Panoptic Quality (PQ) across Thresholds",
    )
    # out_figure = str(Path(__file__).parent.joinpath("autc.png"))
    # fig.write_image(out_figure)


if __name__ == "__main__":
    main()
