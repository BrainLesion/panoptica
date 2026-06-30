import os

from rich.console import Console

CITATION_LINK = "https://github.com/BrainLesion/panoptica#citation"

# Programmatic override, checked in addition to the PANOPTICA_CITATION_REMINDER env var.
_CITATION_REMINDER_DISABLED = False


def disable_citation_reminder() -> None:
    """Suppress the panoptica citation reminder for the rest of this process.

    The runtime equivalent of setting ``PANOPTICA_CITATION_REMINDER`` to a non-"true"
    value -- handy when panoptica is embedded in another package and the reminder is
    just noise.
    """
    global _CITATION_REMINDER_DISABLED
    _CITATION_REMINDER_DISABLED = True


def enable_citation_reminder() -> None:
    """Re-enable the citation reminder (undoes :func:`disable_citation_reminder`)."""
    global _CITATION_REMINDER_DISABLED
    _CITATION_REMINDER_DISABLED = False


def citation_reminder(func):
    """Decorator to remind users to cite panoptica."""

    def wrapper(*args, **kwargs):
        if (
            not _CITATION_REMINDER_DISABLED
            and os.environ.get("PANOPTICA_CITATION_REMINDER", "true").lower() == "true"
        ):
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
