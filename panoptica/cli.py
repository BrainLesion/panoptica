import os
from typing import Optional
import typer
from typing_extensions import Annotated
from importlib.metadata import version
import SimpleITK as sitk
from pathlib import Path
from warnings import warn

from panoptica import Panoptica_Evaluator


def version_callback(value: bool):
    __version__ = version("panoptica")
    if value:
        typer.echo(f"panoptica CLI v{__version__}")
        raise typer.Exit()


app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]}, add_completion=False
)


@app.command()
def main(
    reference: Annotated[
        str,
        typer.Option(
            "-ref",
            "--reference",
            help="The path to the reference/ground-truth image",
        ),
    ],
    prediction: Annotated[
        str,
        typer.Option(
            "-pred",
            "--prediction",
            help="The path to the predicted image",
        ),
    ],
    config: Annotated[
        Optional[str],
        typer.Option(
            "-c",
            "--config",
            help="The path to the configuration file.",
        ),
    ],
    version: Annotated[
        Optional[bool],
        typer.Option(
            "-v",
            "--version",
            callback=version_callback,
            is_eager=True,
            help="Print the version and exit.",
        ),
    ] = None,
):
    cli_main(reference, prediction, config)


def cli_main(reference: str | Path, prediction: str | Path, config: str | Path | None):
    """
    Generate the panoptica evaluation report for the given reference and prediction images.
    """
    if config is None:
        warn(
            "No config file provided. Using default config. For specific projects, please provided a config. Otherwise this might not calculate what you desire.",
            UserWarning,
        )
        # default config
        config = "panoptica/configs/panoptica_evaluator_default.yaml"
    # check if files exist
    for file in [reference, prediction, config]:
        if not os.path.exists(file):
            raise typer.BadParameter(
                f"Error: The file '{file}' does not exist. Please provide a valid path."
            )

    # Load the Panoptica Evaluator object from config
    evaluator = Panoptica_Evaluator.load_from_config(config)

    # Call the evaluate function
    group2result = evaluator.evaluate(prediction, reference)
    for group, result in group2result.items():
        print(f"Group: {group}")
        print(f"Result: {result}")

    return group2result


if __name__ == "__main__":
    app()
