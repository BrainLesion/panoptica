import os

from rich.console import Console

CITATION_LINK = "https://github.com/BrainLesion/panoptica#citation"


def citation_reminder(func):
    def wrapper(*args, **kwargs):
        if os.environ.get("PANOPTICA_CITATION_REMINDER", "true").lower() == "true":
            console = Console()
            console.rule("Thank you for using [bold]panoptica[/bold]")
            console.print(
                f"Please support our development by citing",
                justify="center",
            )
            console.print(
                f"{CITATION_LINK} -- Thank you!",
                justify="center",
            )
            console.rule()
            console.line()
        return func(*args, **kwargs)

    return wrapper
