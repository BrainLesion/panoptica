from typing import Optional
from pprint import pprint
import typer
from typing_extensions import Annotated
from importlib.metadata import version
import SimpleITK as sitk


from panoptica import InputType, Panoptica_Evaluator
from panoptica.metrics import Metric


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
    """
    Preprocess the input images according to the BraTS protocol.
    """

    ref_masks = sitk.GetArrayFromImage(sitk.ReadImage(reference))
    pred_masks = sitk.GetArrayFromImage(sitk.ReadImage(prediction))

    evaluator = Panoptica_Evaluator(
        expected_input=InputType.MATCHED_INSTANCE,
        decision_metric=Metric.IOU,
        decision_threshold=0.5,
    )

    pprint(evaluator.evaluate(pred_masks, ref_masks)["ungrouped"])


if __name__ == "__main__":
    app()
