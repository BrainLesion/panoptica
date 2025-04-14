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
    input_type: Annotated[
        str,
        typer.Option(
            "-it",
            "--input-type",
            help="The input type of the images. Can be one of: "
            + ",".join([i.name for i in InputType]),
        ),
    ],
    decision_metric: Annotated[
        str,
        typer.Option(
            "-dm",
            "--decision-metric",
            help="The decision metric to use. Can be one of: "
            + ",".join([i.name for i in Metric]),
        ),
    ],
    threshold: Annotated[
        float,
        typer.Option(
            "-th",
            "--threshold",
            help="The decision threshold to use.",
        ),
    ] = 0.5,
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
    Generate the panoptica evaluation report for the given reference and prediction images.
    """

    ref_masks = sitk.GetArrayFromImage(sitk.ReadImage(reference))
    pred_masks = sitk.GetArrayFromImage(sitk.ReadImage(prediction))

    input_type = input_type.upper()
    for input_type_it in InputType:
        if input_type == input_type_it.name:
            input_type = input_type_it
            break

    decision_metric = decision_metric.upper()
    for decision_metric_it in Metric:
        if decision_metric == decision_metric_it.name:
            decision_metric = decision_metric_it
            break

    evaluator = Panoptica_Evaluator(
        expected_input=input_type,
        decision_metric=decision_metric,
        decision_threshold=threshold,
    )

    print(evaluator.evaluate(pred_masks, ref_masks)["ungrouped"])


if __name__ == "__main__":
    app()
