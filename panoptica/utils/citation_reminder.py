import os
from rich.console import Console

CITATION_LINK = "https://github.com/BrainLesion/panoptica#citation"


def citation_reminder(func):
    def wrapper(*args, **kwargs):
        if os.environ.get("PANOPTICA_CITATION_REMINDER", "true").lower() == "true":
            console = Console()
            console.rule("[bold] Citation reminder [/bold]")
            console.print(
                f"If you use this software in your research, please [bold]cite[/bold] [italic]{CITATION_LINK}[italic]", justify="center")
            console.print("Thank you!", justify="center")
            console.rule()
            console.line()
        return func(*args, **kwargs)
    return wrapper
