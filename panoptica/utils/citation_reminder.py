import os

from rich.console import Console

CITATION_LINK = "https://github.com/BrainLesion/panoptica#citation"


def citation_reminder(func):
    """Decorator to remind users to cite panoptica."""

    def wrapper(*args, **kwargs):
        if os.environ.get("PANOPTICA_CITATION_REMINDER", "true").lower() == "true":
            console = Console()
            console.rule("Thank you for using [bold]panoptica[/bold]")
            console.print(
                "Please support our development by citing",
                justify="center",
            )
            console.print(
                f"{CITATION_LINK} -- Thank you!",
                justify="center",
            )
            console.rule()
            console.line()
            os.environ["PANOPTICA_CITATION_REMINDER"] = "false"  # Show only once
        return func(*args, **kwargs)

    return wrapper
