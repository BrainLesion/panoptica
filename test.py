from panoptica import Panoptica_Statistic
from pathlib import Path

output_dir = Path(
    "/media/hendrik/be5e95dd-27c8-4c31-adc5-7b75f8ebd5c5/data/hendrik/panoptica/verse_results_test/"
)
Panoptica_Statistic.from_file(output_dir.joinpath("christian_payer_1.0.tsv"))
