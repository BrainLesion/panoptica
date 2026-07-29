from pathlib import Path
from sample_generation import main as create_sample

for i in range(9, 13):
    width = 2 ** i
    create_sample(Path(f"{width}_ref.png"), width, seed = 0)
    create_sample(Path(f"{width}_pred.png"), width, seed = 1)